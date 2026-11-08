"""Raw-turn lifecycle; shared S012 candidates, readouts, scoring and receipts."""
import copy
import hashlib
import os
from pathlib import Path
import shutil

from memcut.native_memory.contract import digest
from memcut.native_memory.persistence import write_json
from memcut.source_anchored_systems import runner as shared
from memcut.source_anchored_v2.runner import (read, sha, sealed_write, normalize_claim,
    stage_claims, validate_state, role_cache, gateway)
from memcut.source_anchored_v2.candidates import build_pool, rank_s2, verify_pool, candidate_protocol
from . import VERSION
from .bank import RawTurnBank
from .embedding import LocalEmbeddings, DEFAULT_EMBEDDING, REGISTRY, embedding_settings

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'configs/raw_turn_rag_v1.json'
PROTOCOL = {**copy.deepcopy(shared.PROTOCOL), 'version': VERSION + '-whole-object-eval2of2',
    'candidate_search': 'exact cosine top20 live raw-turn objects on shared S0; source-turn cap20',
    'provenance': 'one original single-role turn per stored object; exact one-to-one source table',
    'memory_generation_finish': 'no generation; store original text and local embedding',
    'formation_llm_calls': 0}


def code_identity():
    paths = list(Path(__file__).parent.glob('*.py')) + [CONFIG, REGISTRY, ROOT/'tools/raw_turn_rag.py']
    # Freeze every reused implementation without adding files to old globbed packages.
    paths += [ROOT/p for p in shared.code_identity()]
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(paths))}


def formation_identity(plan):
    paths = ['memcut/raw_turn_rag/bank.py', 'memcut/raw_turn_rag/embedding.py',
             'memcut/raw_turn_rag/__init__.py', 'memcut/source_anchored_systems/base.py',
             'memcut/source_anchored_systems/lineage.py', 'memcut/source_anchored_v2/provenance.py',
             'memcut/native_memory/contract.py']
    return digest({'format': VERSION, 'settings': plan['settings'],
                   'source': plan['inputs']['source']['sha256'],
                   'code': {p: plan['code'][p] for p in paths}})


def load_plan(folder):
    folder = Path(folder)
    plan = read(folder/'PLAN.json')
    expected = candidate_protocol(PROTOCOL, plan['protocol']['n_obj'], plan['protocol']['n_turn'])
    if plan['system'] != 'raw_turn_rag' or plan['protocol'] != expected or plan['code'] != code_identity():
        raise ValueError('Frozen implementation changed; use a new RAG run version')
    expected_settings = {**read(CONFIG), 'native_eval_k': plan['settings']['native_eval_k'],
                         'embedding': embedding_settings(plan['settings']['embedding']['name'])}
    if plan['settings'] != expected_settings:
        raise ValueError('Frozen RAG settings differ')
    if sha(folder/'PROTOCOL.md') != plan['protocol_document_sha256']:
        raise ValueError('Frozen protocol changed')
    for entry in plan['inputs'].values():
        if sha(entry['path']) != entry['sha256']:
            raise ValueError('Frozen source or annotations changed')
    return plan


def prepare(folder, *, source, contracts, annotations, answer_model, endpoint,
            selector_model=None, compatibility='openai', claim_ids=None, k_list=(10,),
            candidate_objects=20, candidate_turns=20, native_eval_k=16, embedding=DEFAULT_EMBEDDING,
            controls=False, residual_plus=False, reuse_full=None,
            selector_endpoint=None, selector_compatibility=None):
    if type(native_eval_k) is not int or native_eval_k <= 0:
        raise ValueError('native_eval_k must be positive')
    protocol = candidate_protocol(PROTOCOL, candidate_objects, candidate_turns)
    inputs = {k: {'path': str(Path(p).resolve()), 'sha256': sha(p)} for k, p in
              [('source', source), ('contracts', contracts), ('annotations', annotations)]}
    turns, _, order = shared.import_source(read(source))
    contract, annotation = read(contracts), read(annotations)
    if any(x.get('source_sha256') != inputs['source']['sha256'] for x in (contract, annotation)):
        raise ValueError('Contracts/annotations do not match source')
    all_ids = [c['claim_id'] for c in contract['claims']]
    selected = set(all_ids if claim_ids is None else claim_ids)
    if len(all_ids) != len(set(all_ids)) or not selected or not selected <= set(all_ids):
        raise ValueError('Unknown, duplicate or empty claim cohort')
    ks = sorted(set(k_list))
    if not ks or any(type(k) is not int or k not in PROTOCOL['k_list'] for k in ks):
        raise ValueError('Use documented deletion budgets')
    claims = []
    for row in contract['claims']:
        if row['claim_id'] not in selected:
            continue
        claim = {**normalize_claim(row), 'deletion_request': row['deletion_request']}
        for field, key in [('evidence_turns', 'carriers'), ('disclosure_turns', 'direct'), ('distractor_turns', 'distractors')]:
            ids = set()
            for a in annotation[key]:
                if a['claim_id'] != claim['claim_id']:
                    continue
                tid = a['record_id']
                if tid not in turns or a['role'] != turns[tid]['role']:
                    raise ValueError('Annotation turn or role differs')
                text = turns[tid]['text']
                if a.get('text_sha256') != hashlib.sha256(text.encode()).hexdigest() or ('text' in a and a['text'] != text):
                    raise ValueError('Annotation text differs')
                ids.add(tid)
            claim[field] = [t for t in order if t in ids]
        if not 0 < len(claim['evidence_turns']) <= 10 or not claim['disclosure_turns']:
            raise ValueError('Require E<=10 and nonempty E/D')
        if 'direct_record_ids' in row and set(row['direct_record_ids']) != set(claim['disclosure_turns']):
            raise ValueError('Direct fields disagree')
        claims.append(claim)
    if compatibility not in {'openai', 'deepseek', 'mimo'} or (selector_compatibility or compatibility) not in {'openai', 'deepseek', 'mimo'}:
        raise ValueError('Unknown provider compatibility')
    reuse = None
    if reuse_full:
        p = Path(reuse_full).resolve()
        if sha(p) != read(p.with_suffix('.frozen.json'))['sha256']:
            raise ValueError('Reusable Full changed')
        reuse = {'path': str(p), 'sha256': sha(p)}
    settings = {**read(CONFIG), 'native_eval_k': native_eval_k, 'embedding': embedding_settings(embedding)}
    plan = {'system': 'raw_turn_rag', 'protocol': protocol, 'settings': settings,
            'code': code_identity(), 'inputs': inputs, 'claims': claims, 'k_list': ks,
            'models': {'answer': answer_model, 'selector': selector_model or answer_model},
            'providers': {
                'answer': {'endpoint': endpoint, 'compatibility': compatibility, 'key_env': 'MEMCUT_MODEL_API_KEY'},
                'selector': {'endpoint': selector_endpoint or endpoint,
                             'compatibility': selector_compatibility or compatibility,
                             'key_env': 'MEMCUT_SELECTOR_API_KEY' if selector_endpoint else 'MEMCUT_MODEL_API_KEY'}},
            'endpoint': endpoint, 'compatibility': compatibility, 'reuse_full': reuse,
            'controls': controls, 'residual_plus': residual_plus, 'retention': [],
            'protocol_document_sha256': sha(ROOT/'PROTOCOL.md'),
            'conditions': ['Full', 'S0'] + [f'{s}@{k}' for s in ('S1', 'S2') for k in ks]}
    folder = Path(folder)
    if folder.exists():
        raise FileExistsError('Use a new versioned run directory')
    folder.mkdir(parents=True)
    shutil.copyfile(ROOT/'PROTOCOL.md', folder/'PROTOCOL.md')
    sealed_write(folder/'PLAN.json', plan)
    return plan


class Factory:
    def __init__(self, plan, *, embedder=None, cache_dir=None):
        self.plan = plan
        self.embedder = embedder or LocalEmbeddings(plan['settings']['embedding'],
            cache_dir or ROOT/'runtime/raw-turn-rag/embeddings')
        self.turns, _, _ = shared.import_source(read(plan['inputs']['source']['path']))

    def create(self, snapshot=None):
        bank = RawTurnBank(self.plan['settings'], self.turns, self.embedder)
        bank.progress['formation_identity'] = formation_identity(self.plan)
        if snapshot:
            bank.restore(snapshot)
        return bank


def validate_full(bank, plan):
    if (not bank.frozen or bank.progress['stage'] != 'FULL' or bank.meta['deleted_turns']
            or bank.progress['formation_identity'] != formation_identity(plan)
            or set(bank.turns) != {r['turn_id'] for r in bank.rows.values()}
            or any(bank.trace(oid) != {row['turn_id']} for oid, row in bank.rows.items())):
        raise ValueError('Require complete undeleted one-to-one Raw-turn Full')


def build(folder, plan, *, factory=None):
    """No completion transport is accepted or constructed during formation."""
    folder = Path(folder)
    factory = factory or Factory(plan)
    bank = factory.create()
    full = folder/'full/full.pkl'
    if full.exists():
        bank.restore(full)
    elif plan['reuse_full']:
        reuse = plan['reuse_full']
        if sha(reuse['path']) != reuse['sha256']:
            raise ValueError('Reusable Full changed')
        bank.restore(reuse['path'])
    else:
        bank.form()
    validate_full(bank, plan)
    if not full.exists():
        bank.snapshot(full)
    sealed_write(folder/'full/IDENTITY.json', {'plan_sha256': digest(plan), 'state_identity': bank.state_identity()})
    write_json(folder/'PROGRESS.json', {'stage': 'full_complete', 'objects': len(bank.rows),
        'completed_turns': len(bank.turns), 'full': str(full), 'formation_llm_calls': 0})
    return str(full)


async def delete(folder, plan, completions=None, *, factory=None, claim_ids=None, strategies=('S0', 'S1', 'S2')):
    if not strategies or not set(strategies) <= {'S0', 'S1', 'S2'}:
        raise ValueError('Choose S0/S1/S2')
    folder = Path(folder)
    factory = factory or Factory(plan)
    full = folder/'full/full.pkl'
    if read(folder/'full/IDENTITY.json')['plan_sha256'] != digest(plan):
        raise ValueError('Full belongs to another plan')

    async def save(claim, condition, bank, receipt, pool_sha=None):
        target = folder/'claims'/claim['claim_id']/condition
        path = target/'state.pkl'
        if path.exists():
            bank.restore(path)
            receipt = bank.progress.get('last_receipt', {})
        else:
            target.mkdir(parents=True, exist_ok=True)
            if condition == 'Full':
                os.link(full, path)
                shutil.copyfile(full.with_suffix('.frozen.json'), path.with_suffix('.frozen.json'))
            else:
                bank.progress['last_receipt'] = receipt
                bank.snapshot(path)
        if condition != 'Full':
            sealed_write(target/'RECEIPT.json', receipt)
        if (target/'STATE.json').exists():
            state = validate_state(folder, plan, claim, condition)
            if state['state_identity'] != bank.state_identity() or state['pool_sha256'] != pool_sha:
                raise ValueError('Saved branch identity differs')
        else:
            ctx = await shared.contexts(bank, claim, residual_plus=plan['residual_plus'])
            sealed_write(target/'STATE.json', {'claim_id': claim['claim_id'], 'condition': condition,
                'plan_sha256': digest(plan), 'snapshot_sha256': sha(path),
                'state_identity': bank.state_identity(), 'pool_sha256': pool_sha,
                'deletion': receipt, 'contexts': ctx})
        return path

    for claim in stage_claims(plan, claim_ids):
        base = folder/'claims'/claim['claim_id']
        bank = factory.create(full)
        validate_full(bank, plan)
        await save(claim, 'Full', bank, {})
        bank = factory.create(full)
        receipt = await bank.delete_turns(claim['disclosure_turns'])
        s0 = await save(claim, 'S0', bank, receipt)
        if set(bank.meta['deleted_turns']) != set(claim['disclosure_turns']):
            raise ValueError('S0 did not delete exactly Direct')
        if set(strategies) == {'S0'}:
            continue
        pool_path = base/'POOL.json'
        pool = read(pool_path) if pool_path.exists() else await build_pool(bank, claim['deletion_request'],
            claim['disclosure_turns'], n_obj=plan['protocol']['n_obj'], n_turn=plan['protocol']['n_turn'])
        verify_pool(pool)
        if (pool['state_identity'] != bank.state_identity() or pool['requested_objects'] != plan['protocol']['n_obj']
                or pool['requested_candidate_turns'] != plan['protocol']['n_turn']):
            raise ValueError('Candidate pool differs from frozen shared S0')
        sealed_write(pool_path, pool)
        orders = {'S1': [r['turn_id'] for r in pool['turns']]}
        if 'S2' in strategies:
            path = base/'S2_ORDER.json'
            if path.exists():
                ranking = read(path)
            else:
                if completions is None and pool['turns']:
                    raise ValueError('S2 needs a selector completion transport')
                ranking = {**rank_s2(pool, role_cache(completions, 'selector'), plan['models']['selector']),
                    'pool_sha256': pool['pool_sha256'], 'plan_sha256': digest(plan)}
            if ranking['pool_sha256'] != pool['pool_sha256'] or ranking['plan_sha256'] != digest(plan):
                raise ValueError('S2 ranking belongs to another pool or plan')
            sealed_write(path, ranking)
            orders['S2'] = ranking['order']
        for strategy in ('S1', 'S2'):
            if strategy not in strategies:
                continue
            order = orders[strategy]
            if len(order) != len(set(order)) or set(order) != set(orders['S1']):
                raise ValueError('Ordering must cover the pool exactly once')
            for k in plan['k_list']:
                bank = factory.create(s0)
                receipt = await bank.delete_turns(order[:k])
                receipt.update(requested_k=k, actual_k=min(k, len(order)), budget_complete=len(order)>=k,
                    extra_deleted_turns=order[:k], direct_turns=claim['disclosure_turns'])
                await save(claim, f'{strategy}@{k}', bank, receipt, pool['pool_sha256'])
                expected = set(claim['disclosure_turns']) | set(order[:k])
                if set(bank.meta['deleted_turns']) != expected or any(bank.reverse_trace(t) for t in expected):
                    raise ValueError('Saved condition did not erase the selected turns')
        write_json(folder/'PROGRESS.json', {'stage': 'claim_deleted', 'claim_id': claim['claim_id']})


def answer(folder, plan, completions, *, claim_ids=None, conditions=None):
    if conditions is None:
        return shared.answer(folder, plan, completions, claim_ids=claim_ids)
    if not conditions or not set(conditions) <= set(plan['conditions']):
        raise ValueError('Unknown/empty stage conditions')
    recovery = role_cache(completions, 'answer')
    for claim in stage_claims(plan, claim_ids):
        for condition in plan['conditions']:
            if condition not in conditions:
                continue
            state = validate_state(folder, plan, claim, condition)
            for family, context in state['contexts'].items():
                result = shared.evaluate(claim, context, recovery, plan['models']['answer'])
                sealed_write(Path(folder)/'claims'/claim['claim_id']/condition/(family+'.json'),
                    {'claim_id': claim['claim_id'], 'condition': condition, 'family': family,
                     'plan_sha256': digest(plan), 'result': result})


def report(folder, plan):
    return shared.report(folder, plan)
