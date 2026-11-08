"""Versioned MemoryBank build, deletion, answering and reporting stages."""
import asyncio
import copy
import hashlib
import json
import re
import tempfile
from pathlib import Path

from memcut.native_memory.contract import digest
from memcut.native_memory.persistence import write_json
from memcut.native_memory.profiles import load_profile, resolve_profile, verify_profile
from . import PROTOCOL as NATIVE_STORAGE_PROTOCOL
from .scoring import evaluation_protocol, select_rotations
from .candidates import build_pool, rank_s2, verify_pool
from .completions import CachedCompletions
from .evaluation import contexts, evaluate, summarize, answer_request, score_response
from .memorybank import SourceMemoryBank, import_source, open_memorybank
from .whole_object_deletion import POLICY, delete_source_turns, deletion_protocol

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = deletion_protocol(evaluation_protocol(NATIVE_STORAGE_PROTOCOL))


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_identity():
    paths = list(Path(__file__).parent.glob('*.py')) + [ROOT / 'memcut/native_memory/memorybank.py',
        ROOT / 'tools/memorybank_s012.py', ROOT / 'configs/native_memory_profiles.json']
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def sealed_write(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError(f'Frozen artifact differs: {path.name}')
        return
    write_json(path, value)


def normalize_claim(row):
    cid = row['claim_id']
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', cid):
        raise ValueError('Unsafe claim ID')
    choices = row['choices']
    ids = [c['id'] for c in choices]
    if len(ids) not in {4, 5} or len(set(ids)) != len(ids) or row['target_choice_id'] not in ids:
        raise ValueError('Invalid actual choice set')
    normalized = select_rotations(row.get('rotations'), ids)
    return {'claim_id': cid, 'stem': row.get('neutral_question') or row.get('stem') or row['question'],
            'choices': choices, 'target_choice_id': row['target_choice_id'], 'rotations': normalized,
            'category': row.get('category')}


def prepare(folder, *, source, contracts, annotations, write_model, answer_model,
            selector_model, endpoint, compatibility='openai', claim_ids=None, k_list=(10,),
            user_id=None, residual_plus=False, controls=False, retention=None, reuse_full=None,
            answer_endpoint=None, answer_compatibility=None, selector_endpoint=None, selector_compatibility=None):
    source, contracts, annotations = map(lambda p: Path(p).resolve(), (source, contracts, annotations))
    episode, contract, annotation = read(source), read(contracts), read(annotations)
    turns, groups, order = import_source(episode)
    if any(x.get('source_sha256') != sha(source) for x in (contract, annotation)):
        raise ValueError('Contracts/annotations are not bound to this source')
    selected = set(claim_ids) if claim_ids is not None else {r['claim_id'] for r in contract['claims']}
    if not selected or not selected <= {r['claim_id'] for r in contract['claims']}:
        raise ValueError('Unknown/empty planned claim cohort')
    ks = sorted(set(k_list))
    if not ks or any(type(k) is not int or k not in PROTOCOL['k_list'] for k in ks):
        raise ValueError('Use documented k values')
    claims = []
    for row in contract['claims']:
        if row['claim_id'] not in selected:
            continue
        c = normalize_claim(row)
        c['deletion_request'] = row['deletion_request']
        for field, key in [('evidence_turns', 'carriers'), ('disclosure_turns', 'direct'), ('distractor_turns', 'distractors')]:
            ids = set()
            for a in annotation[key]:
                if a['claim_id'] != c['claim_id']:
                    continue
                t = a['record_id']
                if t not in turns or a['role'] != turns[t]['role']:
                    raise ValueError('Annotation does not match original turn')
                if 'text' in a and a['text'] != turns[t]['text']:
                    raise ValueError('Annotation text differs from source')
                text_hash = hashlib.sha256(turns[t]['text'].encode()).hexdigest()
                if a.get('text_sha256') != text_hash:
                    raise ValueError('Annotation source-text hash differs')
                ids.add(t)
            c[field] = [t for t in order if t in ids]
        if not c['evidence_turns'] or len(c['evidence_turns']) > 10 or not c['disclosure_turns']:
            raise ValueError('Require E<=10 and nonempty E/D')
        if 'direct_record_ids' in row and set(row['direct_record_ids']) != set(c['disclosure_turns']):
            raise ValueError('Direct fields disagree')
        claims.append(c)
    profile = load_profile('memorybank_faiss_en')
    verification = verify_profile('memorybank_faiss_en')
    if not verification['ok']:
        raise ValueError(verification['errors'])
    models = {'write': write_model, 'answer': answer_model, 'selector': selector_model}
    providers = {
        'write': {'endpoint': endpoint, 'compatibility': compatibility, 'key_env': 'MEMCUT_MODEL_API_KEY'},
        'answer': {'endpoint': answer_endpoint or endpoint, 'compatibility': answer_compatibility or compatibility,
                   'key_env': 'MEMCUT_RECOVERY_API_KEY' if answer_endpoint else 'MEMCUT_MODEL_API_KEY'},
        'selector': {'endpoint': selector_endpoint or endpoint, 'compatibility': selector_compatibility or compatibility,
                     'key_env': 'MEMCUT_SELECTOR_API_KEY' if selector_endpoint else 'MEMCUT_MODEL_API_KEY'},
    }
    resolved = {k: resolve_profile('memorybank_faiss_en', model=m, base_url=providers[k]['endpoint']) for k, m in models.items()}
    retention_rows = []
    if retention:
        pack = read(retention)
        if pack['source_sha256'] != sha(source):
            raise ValueError('Retention source mismatch')
        for row in pack['questions']:
            q = normalize_claim({**row, 'claim_id': row['question_id']})
            q['for_claim_ids'] = row['for_claim_ids']
            q['support_turns'] = row['support_turns']
            if not set(q['for_claim_ids']) <= selected or not set(q['support_turns']) <= set(turns):
                raise ValueError('Retention question is outside planned source/cohort')
            retention_rows.append(q)
    if any(p['compatibility'] not in {'openai', 'deepseek', 'mimo'} for p in providers.values()):
        raise ValueError('Unknown provider compatibility')
    reuse = None
    if reuse_full:
        p = Path(reuse_full).resolve()
        frozen = read(p.with_suffix('.frozen.json'))
        if sha(p) != frozen['sha256']:
            raise ValueError('Reusable Full snapshot changed')
        reuse = {'path': str(p), 'sha256': sha(p)}
    plan = {'protocol': PROTOCOL, 'protocol_document_sha256': sha(ROOT / 'PROTOCOL.md'),
            'code': code_identity(), 'profile': profile, 'resolved_profiles': resolved,
            'inputs': {k: {'path': str(p), 'sha256': sha(p)} for k, p in
                       [('source', source), ('contracts', contracts), ('annotations', annotations)]},
            'models': models, 'endpoint': endpoint, 'compatibility': compatibility, 'providers': providers,
            'user_id': user_id or turns[order[0]].get('speaker') or episode['id'],
            'claims': claims, 'k_list': ks, 'residual_plus': residual_plus,
            'controls': controls, 'retention': retention_rows, 'reuse_full': reuse,
            'conditions': ['Full', 'S0'] + [f'{s}@{k}' for s in ('S1', 'S2') for k in ks]}
    folder = Path(folder)
    if folder.exists():
        raise FileExistsError('Use a new versioned run directory')
    folder.mkdir(parents=True)
    sealed_write(folder / 'PLAN.json', plan)
    return plan


def load_plan(folder):
    plan = read(Path(folder) / 'PLAN.json')
    if plan['protocol'] != PROTOCOL or plan['code'] != code_identity():
        raise ValueError('Frozen implementation changed; use a new run version')
    for item in plan['inputs'].values():
        if sha(item['path']) != item['sha256']:
            raise ValueError('Frozen input changed')
    if plan['profile'] != load_profile('memorybank_faiss_en'):
        raise ValueError('Native reference profile changed')
    return plan


def gateway(folder, plan):
    import os
    from openai import OpenAI
    class Gateway:
        def __init__(self):
            self.clients, self.caches = {}, {}

        def for_role(self, role):
            if role not in self.caches:
                provider = plan['providers'][role]
                def ensure_client():
                    if role not in self.clients:
                        key = os.environ.get(provider['key_env'])
                        if not key:
                            raise ValueError(f"{provider['key_env']} is required for {role}")
                        self.clients[role] = OpenAI(api_key=key, base_url=provider['endpoint'], max_retries=0, timeout=180)
                def create(**kwargs):
                    return self.clients[role].chat.completions.create(**kwargs)
                self.caches[role] = CachedCompletions(Path(folder) / 'requests', create,
                    endpoint=provider['endpoint'], compatibility=provider['compatibility'], before_dispatch=ensure_client)
            return self.caches[role]

        @property
        def calls(self):
            return [r for cache in self.caches.values() for r in cache.calls]

        def close(self):
            for client in self.clients.values():
                client.close()
    gateway = Gateway()
    return gateway, gateway


def role_cache(completions, role):
    return completions.for_role(role) if hasattr(completions, 'for_role') else completions


def template(folder, plan):
    return open_memorybank(ROOT / plan['profile']['upstream']['repository'], plan['user_id'],
                           Path(folder) / '_unused_factory.pkl', plan['models']['write'])


def branch(seed, path, snapshot=None):
    bank = SourceMemoryBank(seed.native, seed.prompts, copy.copy(seed.qa), seed.user_id,
                            path, seed.summaries, seed.model)
    bank.restore(snapshot)
    return bank


async def build(folder, plan, completions, *, seed=None):
    folder = Path(folder)
    completions = role_cache(completions, 'write')
    frozen = folder / 'full' / 'full.pkl'
    if frozen.exists() and frozen.with_suffix('.frozen.json').exists():
        if sha(frozen) != read(frozen.with_suffix('.frozen.json'))['sha256']:
            raise ValueError('Full snapshot changed')
        if (folder / 'full' / 'IDENTITY.json').exists():
            if read(folder / 'full' / 'IDENTITY.json')['plan_sha256'] != digest(plan):
                raise ValueError('Full belongs to a different plan')
            return str(frozen)
    seed = seed or template(folder, plan)
    path = folder / 'formation' / 'state.pkl'
    bank = SourceMemoryBank(seed.native, seed.prompts, copy.copy(seed.qa), seed.user_id, path,
                            seed.summaries, seed.model)
    if path.exists():
        bank.restore()
    elif plan.get('reuse_full'):
        reuse = plan['reuse_full']
        if sha(reuse['path']) != reuse['sha256']:
            raise ValueError('Reusable Full changed')
        bank.restore(reuse['path'])
    else:
        await bank.form(read(plan['inputs']['source']['path']), completions)
    if (bank.meta['source_digest'] != digest(read(plan['inputs']['source']['path'])) or
            bank.meta['deleted_turns'] or any(e['op'] != 'create' for e in bank.meta['history'])):
        raise ValueError('Full must be an unmodified complete build of this exact episode')
    expected_transport = {'endpoint': completions.endpoint, 'compatibility': completions.compatibility}
    if bank.meta['write_transport'] != expected_transport:
        raise ValueError('Full was built with a different writing provider/compatibility')
    bank.snapshot(frozen)
    sealed_write(folder / 'full' / 'IDENTITY.json', {'plan_sha256': digest(plan), 'state_identity': bank.state_identity()})
    return str(frozen)


def stage_claims(plan, claim_ids=None):
    """Run a subset without changing the frozen cohort or plan identity."""
    if claim_ids is None:
        return plan['claims']
    selected = set(claim_ids)
    if not selected or not selected <= {c['claim_id'] for c in plan['claims']}:
        raise ValueError('Unknown/empty stage claim subset')
    return [c for c in plan['claims'] if c['claim_id'] in selected]


async def delete(folder, plan, completions, *, seed=None, claim_ids=None):
    if plan['protocol'].get('deletion_policy') != POLICY:
        raise ValueError('Whole-object deletion requires a new explicitly frozen plan')
    folder = Path(folder)
    selected_claims = stage_claims(plan, claim_ids)
    selector = role_cache(completions, 'selector')
    seed = seed or template(folder, plan)
    full = folder / 'full' / 'full.pkl'
    if read(folder / 'full' / 'IDENTITY.json')['plan_sha256'] != digest(plan):
        raise ValueError('Full does not belong to this frozen plan')
    with tempfile.TemporaryDirectory(prefix='memorybank-s012-') as temp:
        counter = 0
        def fresh(snap):
            nonlocal counter
            counter += 1
            return branch(seed, Path(temp) / f'{counter}.pkl', snap)
        async def save_state(cid, condition, bank, claim, receipt, pool_sha=None):
            target = folder / 'claims' / cid / condition
            target.mkdir(parents=True, exist_ok=True)
            if not (target / 'state.pkl').exists():
                bank.snapshot(target / 'state.pkl')
            ctx = await contexts(bank, claim, residual_plus=plan['residual_plus'])
            artifact = {'claim_id': cid, 'condition': condition, 'plan_sha256': digest(plan),
                        'snapshot_sha256': sha(target / 'state.pkl'), 'state_identity': bank.state_identity(),
                        'pool_sha256': pool_sha, 'deletion': receipt, 'contexts': ctx}
            sealed_write(target / 'STATE.json', artifact)
            return target / 'state.pkl'
        for claim in selected_claims:
            cid = claim['claim_id']
            base = folder / 'claims' / cid
            full_bank = fresh(full)
            if not (base / 'Full' / 'STATE.json').exists():
                await save_state(cid, 'Full', full_bank, claim, {})
            s0path = base / 'S0' / 'state.pkl'
            if s0path.exists():
                s0 = fresh(s0path)
                receipt = read(base / 'S0' / 'RECEIPT.json')
            else:
                s0 = full_bank
                receipt = await delete_source_turns(s0, claim['disclosure_turns'])
                sealed_write(base / 'S0' / 'RECEIPT.json', receipt)
            if not (base / 'S0' / 'STATE.json').exists():
                await save_state(cid, 'S0', s0, claim, receipt)
            poolpath = base / 'POOL.json'
            pool = read(poolpath) if poolpath.exists() else await build_pool(s0, claim['deletion_request'], claim['disclosure_turns'])
            verify_pool(pool)
            if pool['state_identity'] != s0.state_identity():
                raise ValueError('Pool does not match shared S0')
            sealed_write(poolpath, pool)
            rankpath = base / 'S2_ORDER.json'
            ranking = read(rankpath) if rankpath.exists() else rank_s2(pool, selector, plan['models']['selector'])
            sealed_write(rankpath, ranking)
            for strategy in ('S1', 'S2'):
                order = [r['turn_id'] for r in pool['turns']] if strategy == 'S1' else ranking['order']
                if len(set(order)) != len(order) or set(order) != {r['turn_id'] for r in pool['turns']}:
                    raise ValueError('Ordering is not a permutation of the same pool')
                work = fresh(s0path)
                for k in plan['k_list']:
                    condition = f'{strategy}@{k}'
                    statepath = base / condition / 'state.pkl'
                    expected = set(claim['disclosure_turns']) | set(order[:k])
                    if statepath.exists():
                        work = fresh(statepath)
                        receipt = read(base / condition / 'RECEIPT.json')
                    else:
                        receipt = await delete_source_turns(work, order[:k])
                        receipt.update({'requested_k': k, 'actual_k': min(k, len(order)),
                                        'budget_complete': len(order) >= k, 'extra_deleted_turns': order[:k],
                                        'direct_turns': claim['disclosure_turns'],
                                        'all_deleted_turns': work.meta['deleted_turns']})
                        sealed_write(base / condition / 'RECEIPT.json', receipt)
                    if set(work.meta['deleted_turns']) != expected:
                        raise ValueError('Condition did not delete exactly its selected turn prefix')
                    if not (base / condition / 'STATE.json').exists():
                        await save_state(cid, condition, work, claim, receipt, pool['pool_sha256'])


def validate_state(folder, plan, claim, condition):
    target = Path(folder) / 'claims' / claim['claim_id'] / condition
    state = read(target / 'STATE.json')
    if state['plan_sha256'] != digest(plan) or state['snapshot_sha256'] != sha(target / 'state.pkl'):
        raise ValueError('Wrong plan or changed condition snapshot')
    for row in state['contexts'].values():
        if row['state_identity'] != state['state_identity'] or row['context_sha256'] != digest(row['context']):
            raise ValueError('Readout context does not match its condition')
    return state


def answer(folder, plan, completions, *, claim_ids=None):
    folder = Path(folder)
    completions = role_cache(completions, 'answer')
    selected_claims = stage_claims(plan, claim_ids)
    for claim in selected_claims:
        for condition in plan['conditions']:
            state = validate_state(folder, plan, claim, condition)
            for family, ctx in state['contexts'].items():
                result = evaluate(claim, ctx, completions, plan['models']['answer'])
                sealed_write(folder / 'claims' / claim['claim_id'] / condition / f'{family}.json', {
                    'claim_id': claim['claim_id'], 'condition': condition, 'family': family,
                    'plan_sha256': digest(plan), 'result': result})
    if plan['controls']:
        turns, _, _ = import_source(read(plan['inputs']['source']['path']))
        for claim in selected_claims:
            for name, text in [('no_memory', ''), ('complete_evidence', '\n'.join(
                    turns[t]['text'] for t in dict.fromkeys(claim['evidence_turns'] + claim['disclosure_turns'])))]:
                ctx = {'context': text, 'context_sha256': digest(text), 'state_identity': name, 'empty': False}
                result = evaluate(claim, ctx, completions, plan['models']['answer'])
                sealed_write(folder / 'controls' / claim['claim_id'] / f'{name}.json', result)


async def retention(folder, plan, completions, *, seed=None):
    """Two fixed Full probes define Q+, then the first probe measures retention."""
    if not plan['retention']:
        return
    folder = Path(folder)
    completions = role_cache(completions, 'answer')
    seed = seed or template(folder, plan)
    with tempfile.TemporaryDirectory(prefix='memorybank-retention-') as temp:
        for claim in plan['claims']:
            questions = [q for q in plan['retention'] if claim['claim_id'] in q['for_claim_ids']]
            full = branch(seed, Path(temp) / f'{claim["claim_id"]}-full.pkl', folder / 'full/full.pkl')
            stable = []
            for q in questions:
                native = await full.native_context(q['stem'])
                ctx = {'context': native.native_context['prompt']}
                answers = [score_response(completions.complete(answer_request(q, ctx, r, plan['models']['answer'])),
                                          r, q['target_choice_id'], q['choices']) for r in q['rotations'][:2]]
                sealed_write(folder / 'retention' / claim['claim_id'] / f'{q["claim_id"]}-Full.json', {'answers': answers})
                if all(a['target_hit'] for a in answers):
                    stable.append(q)
            for condition in plan['conditions']:
                if condition == 'Full':
                    continue
                state = validate_state(folder, plan, claim, condition)
                bank = branch(seed, Path(temp) / f'{claim["claim_id"]}-{condition}.pkl',
                              folder / 'claims' / claim['claim_id'] / condition / 'state.pkl')
                records = []
                affected = {e['obj_id'] for e in bank.meta['history'] if e['op'] in {'delete', 'rewrite'}}
                for q in stable:
                    native = await bank.native_context(q['stem'])
                    ctx = {'context': native.native_context['prompt']}
                    a = score_response(completions.complete(answer_request(q, ctx, q['rotations'][0], plan['models']['answer'])),
                                       q['rotations'][0], q['target_choice_id'], q['choices'])
                    support = {oid for t in q['support_turns'] for oid in full.meta['der'].get(t, [])}
                    records.append({'question_id': q['claim_id'], 'answer': a,
                                    'support_object_touched': bool(support & affected)})
                sealed_write(folder / 'retention' / claim['claim_id'] / f'{condition}.json', {
                    'Q_plus': [q['claim_id'] for q in stable], 'records': records,
                    'denominator': len(stable), 'ur': sum(r['answer']['target_hit'] for r in records)/len(stable) if stable else None})


def report(folder, plan):
    folder = Path(folder)
    families = ['native', 'residual'] + (['residual_plus'] if plan['residual_plus'] else [])
    rows = []
    for claim in plan['claims']:
        for condition in plan['conditions']:
            for family in families:
                path = folder / 'claims' / claim['claim_id'] / condition / f'{family}.json'
                if not path.exists():
                    continue
                row = read(path)
                state = validate_state(folder, plan, claim, condition)
                result, ctx = row['result'], state['contexts'][family]
                if (row['plan_sha256'] != digest(plan) or result['context_sha256'] != ctx['context_sha256']
                        or result['state_identity'] != state['state_identity']):
                    raise ValueError('Saved score has a different execution identity')
                for a in result['answers']:
                    rotation = next(r for r in claim['rotations'] if r['trial'] == a['trial'])
                    rescored = score_response({'text': json.dumps(a['parsed']), 'request_id': a['request_id'],
                                               'finish_reason': 'stop'}, rotation, claim['target_choice_id'], claim['choices'])
                    if rescored != a:
                        raise ValueError('Saved answer score does not match fixed options')
                rows.append(row)
    result = {'plan_sha256': digest(plan), **summarize(rows, plan['claims'], plan['conditions'], families)}
    result['budgets'] = {}
    for condition in plan['conditions']:
        if '@' not in condition:
            continue
        receipts = []
        for claim in plan['claims']:
            path = folder / 'claims' / claim['claim_id'] / condition / 'RECEIPT.json'
            if path.exists():
                r = read(path)
                receipts.append({'claim_id': claim['claim_id'], 'requested_k': r['requested_k'],
                                 'actual_k': r['actual_k'], 'budget_complete': r['budget_complete']})
        result['budgets'][condition] = receipts
        for family in families:
            result['metrics'][f'{family}-{condition}']['all_claims_reached_requested_budget'] = (
                len(receipts) == len(plan['claims']) and all(r['budget_complete'] for r in receipts))
    if 'S2@10' in plan['conditions']:
        result['named_metrics'] = {name: result['metrics'][key] for name, key in
                                  [('native-S2', 'native-S2@10'), ('residual-S2', 'residual-S2@10')]}
    if plan.get('retention'):
        result['utility'] = []
        for claim in plan['claims']:
            for condition in plan['conditions']:
                path = folder / 'retention' / claim['claim_id'] / f'{condition}.json'
                if path.exists():
                    result['utility'].append({'claim_id': claim['claim_id'], 'condition': condition, **read(path)})
    write_json(folder / 'SUMMARY.json', result)
    return result
