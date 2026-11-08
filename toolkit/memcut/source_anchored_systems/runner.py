"""Source-turn S0/S1/S2 lifecycle for pinned Nemori and LightMem profiles.

Native formation/deletion is adapter-owned. Candidate selection, frozen contexts,
rotations and fixed denominators reuse the tested MemoryBank protocol code.
"""
import asyncio
import copy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import time
import shutil

from memcut.native_memory.contract import digest
from memcut.native_memory.persistence import write_json
from . import PROTOCOL as NATIVE_STORAGE_PROTOCOL
from memcut.source_anchored_v2.scoring import evaluation_protocol, ROTATION_COUNT, TRIAL_IDS
from memcut.source_anchored_v2.candidates import build_pool, rank_leakcut, verify_pool, candidate_protocol
from memcut.source_anchored_v2.evaluation import contexts as primary_contexts, evaluate, answer_request, score_response
from memcut.source_anchored_v2.runner import (read, sha, sealed_write, normalize_claim, stage_claims,
                                            gateway, role_cache, validate_state, report as core_report)
from .profiles import ROOT, profile as load_profile, load_assets
from .lightmem_sources import POLICY as LIGHTMEM_SOURCE_POLICY, source_protocol
from memcut.source_anchored_v2.whole_object_deletion import POLICY, delete_source_turns, deletion_protocol

PROTOCOL = deletion_protocol(evaluation_protocol(NATIVE_STORAGE_PROTOCOL))


async def contexts(bank, claim, *, residual_plus=False):
    result = await primary_contexts(bank, claim)
    if residual_plus:
        ids = set(result['residual']['obj_ids'])
        for t in claim['evidence_turns']:
            found = await bank.search(bank.turns[t]['text'], top_k=3)
            ids.update(o.address for o in found.ranked['retrieved_document'])
        # Native search is only the ID discovery step. Use current stored
        # bodies even when native-visible episode excerpts are shorter.
        live = {o.address:o for o in await bank.list_all()}
        ordered = sorted(ids, key=bank.meta['birth'].__getitem__)
        text = '\n'.join(live[oid].text for oid in ordered)
        result['residual_plus'] = {'context':text, 'context_sha256':digest(text),
            'state_identity':bank.state_identity(), 'obj_ids':ordered, 'empty':not ordered}
    return result


def import_source(source):
    turns, groups, order = {}, {}, []
    for session in source['sessions']:
        for row in session['turns']:
            tid = row['id']
            if tid in turns or row['role'] not in {'user', 'assistant'} or not isinstance(row['text'], str):
                raise ValueError('Invalid original single-role turn')
            timestamp = row.get('timestamp', session['timestamp'])
            datetime.fromisoformat(timestamp)
            turns[tid] = {**row, 'turn_id': tid, 'timestamp': timestamp, 'session_id': session['id']}
            order.append(tid)
    if not turns:
        raise ValueError('Empty episode')
    return turns, groups, order


def code_identity():
    paths = list(Path(__file__).parent.glob('*.py'))
    paths += list((Path(__file__).parent / 'formation_runtime').glob('*.py'))
    paths += list((ROOT/'memcut/source_anchored_v2').glob('*.py'))
    paths += [ROOT/p for p in ('memcut/native_memory/lightmem_v1.py',
        'memcut/native_memory/lightmem_formation_v1.py', 'memcut/native_memory/contract.py',
        'memcut/native_memory/persistence.py', 'tools/source_anchored_s012.py',
        'configs/source_anchored_systems_v1.json', 'configs/lightmem_native_v1.json')]
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def load_plan(folder):
    plan = read(Path(folder)/'PLAN.json')
    if plan.get('build_execution'):
        from .formation_runtime import execution
        e = plan['build_execution']
        if e != execution(e['workers'], e['checkpoint_turns']):
            raise ValueError('Unknown formation execution policy')
    expected = source_protocol(candidate_protocol(PROTOCOL,plan['protocol']['n_obj'],plan['protocol']['n_turn']),plan['system'])
    if plan['protocol'] != expected or plan['code'] != code_identity():
        raise ValueError('Frozen implementation changed; create a new run version')
    if plan['profile'] != load_profile(plan['system']):
        raise ValueError('Native profile differs')
    if plan['system'] == 'lightmem' and plan['settings'].get('source_policy') != LIGHTMEM_SOURCE_POLICY:
        raise ValueError('LightMem requires the frozen fine-source formation policy')
    if sha(Path(folder)/'PROTOCOL.md') != plan['protocol_document_sha256']:
        raise ValueError('Frozen protocol document changed')
    for entry in plan['inputs'].values():
        if sha(entry['path']) != entry['sha256']:
            raise ValueError('Frozen input changed')
    return plan


def formation_identity(plan):
    modules = {'__init__.py','base.py','lineage.py','profiles.py',f'{plan["system"]}.py'}
    if plan['system'] == 'lightmem':
        modules.add('lightmem_sources.py')
    code = {p:h for p,h in plan['code'].items() if
            (Path(p).parent == Path('memcut/source_anchored_systems') and Path(p).name in modules) or
            p in {'memcut/source_anchored_v2/completions.py','memcut/source_anchored_v2/provenance.py',
                  'memcut/native_memory/contract.py'} or
            (plan['system']=='lightmem' and p in {'memcut/native_memory/lightmem_v1.py',
                'memcut/native_memory/lightmem_formation_v1.py','configs/lightmem_native_v1.json'})}
    return digest({'code':code, 'settings':plan['settings'], 'profile':plan['profile'],
                   'source':plan['inputs']['source']['sha256']})


class Factory:
    """One set of local weights, independent native stores for every branch."""
    def __init__(self, plan):
        self.plan, self.assets, self.shared, self.embedder = plan, load_assets(plan['system']), None, None
        self.seed = None

    def create(self, snapshot=None):
        turns, _, _ = import_source(read(self.plan['inputs']['source']['path']))
        settings = self.plan['settings']
        execution = self.plan.get('build_execution')
        if self.plan['system'] == 'nemori':
            from .nemori import NemoriBank, LocalEmbedding
            if execution:
                from .formation_runtime.nemori import FastNemoriBank
                bank_type = FastNemoriBank
            else:
                bank_type = NemoriBank
            if self.embedder is None:
                self.embedder = LocalEmbedding(self.assets['embedder'])
            bank = bank_type(settings, turns, embedder=self.embedder,
                             **({'execution': execution} if execution else {}))
        else:
            from .lightmem import LightMemBank
            if execution:
                from .formation_runtime.lightmem import FastLightMemBank
                bank_type = FastLightMemBank
            else:
                bank_type = LightMemBank
            if self.seed is None:
                self.seed = LightMemBank(settings, assets=self.assets)
            bank = bank_type(settings, turns, assets=self.assets, shared=self.seed.shared,
                             **({'execution': execution} if execution else {}))
        if snapshot is not None:
            try:
                bank.restore(snapshot)
            except BaseException:
                bank.close()
                raise
        else:
            bank.progress['formation_identity'] = formation_identity(self.plan)
        return bank

    def close(self):
        if self.seed is not None:
            self.seed.close()


def checkpoint(path, bank, plan):
    """One bounded, atomic formation checkpoint; final/claim states immutable."""
    path = Path(path)
    if hasattr(bank, 'flush_pending'):
        bank.flush_pending()
    bank.lineage.validate(observed=bank.observation())
    payload = pickle.dumps(bank.export(), protocol=5)
    envelope = pickle.dumps({'payload': payload, 'sha256': hashlib.sha256(payload).hexdigest(),
                             'plan_sha256': digest(plan)}, protocol=5)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix('.pending')
    with pending.open('wb') as f:
        f.write(envelope)
        f.flush()
        os.fsync(f.fileno())
    os.replace(pending, path)


def restore_checkpoint(path, bank, plan):
    value = pickle.loads(Path(path).read_bytes())
    if value['plan_sha256'] != digest(plan) or hashlib.sha256(value['payload']).hexdigest() != value['sha256']:
        raise ValueError('Formation checkpoint changed or belongs to another plan')
    bank.restore_value(pickle.loads(value['payload']))


def validate_full(bank, plan):
    turns, _, order = import_source(read(plan['inputs']['source']['path']))
    if (bank.turns != turns or not bank.frozen or bank.progress['stage'] != 'FULL'
            or bank.progress['next_turn'] != len(order) or bank.meta['deleted_turns']
            or bank.progress.get('formation_identity') != formation_identity(plan)
            or any(e['op'] == 'rewrite' for e in bank.meta['history'])):
        raise ValueError('Full must be a complete, undeleted build of this exact source')


async def build(folder, plan, completions, *, factory=None):
    folder = Path(folder)
    own = factory is None
    factory = factory or Factory(plan)
    bank = factory.create()
    full = folder/'full/full.pkl'
    state = folder/'formation/checkpoint.pkl'
    writer = role_cache(completions, 'write')
    started = time.perf_counter()
    checkpoint_turns = plan.get('build_execution', {}).get('checkpoint_turns', 20)
    def timings():
        if hasattr(bank, 'timings'):
            write_json(folder / 'formation/TIMINGS.json', {
                'elapsed_seconds': time.perf_counter() - started,
                'execution': plan['build_execution'], 'stages': bank.timings.report(),
                'note': 'Worker seconds overlap; stage totals must not be summed as wall time.'})
    try:
        if full.exists():
            bank.restore(full)
            validate_full(bank, plan)
            sealed_write(folder/'full/IDENTITY.json', {'plan_sha256': digest(plan), 'state_identity': bank.state_identity()})
            return str(full)
        if state.exists():
            restore_checkpoint(state, bank, plan)
        elif plan.get('reuse_full'):
            reuse = plan['reuse_full']
            if sha(reuse['path']) != reuse['sha256']:
                raise ValueError('Reusable Full changed')
            bank.restore(reuse['path'])
            validate_full(bank, plan)
        else:
            checkpoint(state, bank, plan)
        order = list(bank.turns)
        plan_sha256 = digest(plan)
        committed_turns = bank.progress['next_turn']
        if not bank.frozen:
            for i in range(bank.progress['next_turn'], len(order)):
                bank.ingest(order[i], writer, final=i==len(order)-1)
                bank.progress.update(stage='INGEST', next_turn=i+1)
                # Immutable source + exact request receipts permit replay of
                # at most one uncommitted batch without repeating paid calls.
                # Never save a partially failed native mutation as committed.
                if (i+1) % checkpoint_turns == 0 or i == len(order)-1:
                    if hasattr(bank, 'timings'):
                        with bank.timings.measure('checkpoint_drain_audit_and_save'):
                            checkpoint(state, bank, plan)
                    else:
                        checkpoint(state, bank, plan)
                    committed_turns = i+1
                if (i+1) % 20 == 0 or i == len(order)-1:
                    write_json(folder/'PROGRESS.json', {'stage': 'formation', 'completed_turns': i+1,
                        'checkpoint_turns': committed_turns,
                        'planned_turns': len(order), 'objects': len(bank.meta['src']),
                        'plan_sha256': plan_sha256})
                    timings()
            bank.finish(writer)
            checkpoint(state, bank, plan)
        validate_full(bank, plan)
        bank.snapshot(full)
        sealed_write(folder/'full/IDENTITY.json', {'plan_sha256': digest(plan), 'state_identity': bank.state_identity()})
        write_json(folder/'PROGRESS.json', {'stage': 'full_complete', 'completed_turns': len(order),
                   'objects': len(bank.meta['src']), 'full': str(full)})
        return str(full)
    finally:
        bank.close()
        timings()
        if own:
            factory.close()


async def delete(folder, plan, completions, *, factory=None, claim_ids=None):
    if plan['protocol'].get('deletion_policy') != POLICY:
        raise ValueError('Whole-object deletion requires a new explicitly frozen plan')
    folder = Path(folder)
    own = factory is None
    factory = factory or Factory(plan)
    selector = role_cache(completions, 'selector')
    full = folder/'full/full.pkl'
    if read(folder/'full/IDENTITY.json')['plan_sha256'] != digest(plan):
        raise ValueError('Full belongs to a different plan')

    async def save(cid, condition, bank, claim, receipt, pool_sha=None):
        target = folder/'claims'/cid/condition
        target.mkdir(parents=True, exist_ok=True)
        if not (target/'state.pkl').exists():
            if condition == 'Full':
                # The frozen Full is identical for every claim. Share bytes;
                # every mutation still happens in a separate native store.
                os.link(full, target/'state.pkl')
                shutil.copyfile(full.with_suffix('.frozen.json'), target/'state.frozen.json')
            else:
                # Receipt and bank commit together. A crash before STATE.json
                # resumes from the receipt already inside this snapshot.
                bank.progress['last_receipt'] = receipt
                bank.snapshot(target/'state.pkl')
        saved = bank.progress.get('last_receipt', receipt)
        if condition != 'Full':
            sealed_write(target/'RECEIPT.json', saved)
        if (target/'STATE.json').exists():
            validate_state(folder, plan, claim, condition)
            return target/'state.pkl'
        ctx = await contexts(bank, claim, residual_plus=plan['residual_plus'])
        sealed_write(target/'STATE.json', {'claim_id': cid, 'condition': condition,
            'plan_sha256': digest(plan), 'snapshot_sha256': sha(target/'state.pkl'),
            'state_identity': bank.state_identity(), 'pool_sha256': pool_sha,
            'deletion': saved, 'contexts': ctx})
        return target/'state.pkl'

    try:
        for claim in stage_claims(plan, claim_ids):
            cid = claim['claim_id']
            base = folder/'claims'/cid
            bank = factory.create(full)
            try:
                validate_full(bank, plan)
                if not (base/'Full/STATE.json').exists():
                    if (base/'Full/state.pkl').exists():
                        bank.restore(base/'Full/state.pkl')
                    await save(cid, 'Full', bank, claim, {})
                s0path = base/'S0/state.pkl'
                if s0path.exists():
                    bank.restore(s0path)
                    receipt = bank.progress['last_receipt']
                else:
                    bank.restore(full)
                    receipt = await delete_source_turns(bank, claim['disclosure_turns'])
                await save(cid, 'S0', bank, claim, receipt)
                poolpath = base/'POOL.json'
                pool = read(poolpath) if poolpath.exists() else await build_pool(bank, claim['deletion_request'], claim['disclosure_turns'],
                    n_obj=plan['protocol']['n_obj'],n_turn=plan['protocol']['n_turn'])
                verify_pool(pool)
                if (pool['requested_objects']!=plan['protocol']['n_obj'] or
                        pool.get('requested_candidate_turns',20)!=plan['protocol']['n_turn']):
                    raise ValueError('Saved candidate pool uses a different frozen budget')
                if pool['state_identity'] != bank.state_identity():
                    raise ValueError('Candidate pool differs from shared S0')
                sealed_write(poolpath, pool)
                rankpath = base/'S2_ORDER.json'
                ranking = read(rankpath) if rankpath.exists() else {
                    **rank_leakcut(pool, selector, plan['models']['selector']),
                    'pool_sha256':pool['pool_sha256'], 'plan_sha256':digest(plan)}
                if ranking.get('pool_sha256')!=pool['pool_sha256'] or ranking.get('plan_sha256')!=digest(plan):
                    raise ValueError('Saved S2 ordering belongs to another pool or plan')
                sealed_write(rankpath, ranking)
                for strategy in ('S1', 'S2'):
                    order = [r['turn_id'] for r in pool['turns']] if strategy == 'S1' else ranking['order']
                    if len(order) != len(set(order)) or set(order) != {r['turn_id'] for r in pool['turns']}:
                        raise ValueError('Ordering must cover the shared pool exactly once')
                    bank.restore(s0path)
                    for k in plan['k_list']:
                        condition = f'{strategy}@{k}'
                        path = base/condition/'state.pkl'
                        if path.exists():
                            bank.restore(path)
                            receipt = bank.progress['last_receipt']
                        else:
                            receipt = await delete_source_turns(bank, order[:k])
                            receipt.update(requested_k=k, actual_k=min(k, len(order)),
                                budget_complete=len(order)>=k, extra_deleted_turns=order[:k],
                                direct_turns=claim['disclosure_turns'])
                        expected = set(claim['disclosure_turns']) | set(order[:k])
                        if set(bank.meta['deleted_turns']) != expected or any(bank.reverse_trace(t) for t in expected):
                            raise ValueError('Condition did not process the exact source-turn closure')
                        await save(cid, condition, bank, claim, receipt, pool['pool_sha256'])
                write_json(folder/'PROGRESS.json', {'stage': 'claim_deleted', 'claim_id': cid})
            finally:
                bank.close()
    finally:
        if own:
            factory.close()


async def retention(folder, plan, completions, *, factory=None):
    if not plan['retention']:
        return
    folder = Path(folder)
    own = factory is None
    factory = factory or Factory(plan)
    recovery = role_cache(completions, 'answer')
    full = factory.create(folder/'full/full.pkl')
    try:
        for claim in plan['claims']:
            cid, stable = claim['claim_id'], []
            for q in plan['retention']:
                if cid not in q['for_claim_ids']:
                    continue
                result = await full.native_context(q['stem'])
                ctx = {'context': result.native_context['prompt']}
                answers = [score_response(recovery.complete(answer_request(q, ctx, r, plan['models']['answer'])),
                           r, q['target_choice_id'], q['choices']) for r in q['rotations'][:2]]
                sealed_write(folder/'retention'/cid/f'{q["claim_id"]}-Full.json',
                    {'answers': answers, 'context': ctx['context'], 'state_identity': full.state_identity()})
                if all(a['target_hit'] for a in answers):
                    stable.append(q)
            for condition in plan['conditions']:
                if condition == 'Full':
                    continue
                validate_state(folder, plan, claim, condition)
                bank = factory.create(folder/'claims'/cid/condition/'state.pkl')
                try:
                    records = []
                    affected = {e['obj_id'] for e in bank.meta['history'] if e['op'] in {'delete', 'rewrite'}}
                    for q in stable:
                        native = await bank.native_context(q['stem'])
                        ctx = {'context': native.native_context['prompt']}
                        a = score_response(recovery.complete(answer_request(q, ctx, q['rotations'][0], plan['models']['answer'])),
                                           q['rotations'][0], q['target_choice_id'], q['choices'])
                        support = {oid for t in q['support_turns'] for oid in full.reverse_trace(t)}
                        records.append({'question_id': q['claim_id'], 'answer': a,
                            'context': ctx['context'], 'support_object_touched': bool(support & affected)})
                    sealed_write(folder/'retention'/cid/f'{condition}.json', {'Q_plus': [q['claim_id'] for q in stable],
                        'records': records, 'denominator': len(stable), 'state_identity': bank.state_identity(),
                        'ur': sum(r['answer']['target_hit'] for r in records)/len(stable) if stable else None})
                finally:
                    bank.close()
    finally:
        full.close()
        if own:
            factory.close()

def prepare(folder, *, system, source, contracts, annotations, write_model, answer_model,
            selector_model, endpoint, compatibility='openai', claim_ids=None, k_list=(10,),
            user_id=None, residual_plus=False, controls=False, retention=None, reuse_full=None,
            answer_endpoint=None, answer_compatibility=None, selector_endpoint=None, selector_compatibility=None,
            candidate_objects=20, candidate_turns=20, build_workers=4, checkpoint_turns=80):
    from .formation_runtime import execution
    build_execution = execution(build_workers, checkpoint_turns)
    protocol = source_protocol(candidate_protocol(PROTOCOL,candidate_objects,candidate_turns),system)
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
    profile = load_profile(system)
    models = {'write': write_model, 'answer': answer_model, 'selector': selector_model}
    providers = {
        'write': {'endpoint': endpoint, 'compatibility': compatibility, 'key_env': 'MEMCUT_MODEL_API_KEY'},
        'answer': {'endpoint': answer_endpoint or endpoint, 'compatibility': answer_compatibility or compatibility,
                   'key_env': 'MEMCUT_RECOVERY_API_KEY' if answer_endpoint else 'MEMCUT_MODEL_API_KEY'},
        'selector': {'endpoint': selector_endpoint or endpoint, 'compatibility': selector_compatibility or compatibility,
                     'key_env': 'MEMCUT_SELECTOR_API_KEY' if selector_endpoint else 'MEMCUT_MODEL_API_KEY'},
    }
    resolved = {**profile['settings'], 'model': write_model, 'write_transport': providers['write']}
    if system == 'lightmem':
        resolved['source_policy'] = LIGHTMEM_SOURCE_POLICY
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
    plan = {'protocol': protocol, 'protocol_document_sha256': sha(ROOT / 'PROTOCOL.md'),
            'build_execution': build_execution,
            'code': code_identity(), 'system': system, 'profile': profile, 'settings': resolved,
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
    shutil.copyfile(ROOT/'PROTOCOL.md', folder/'PROTOCOL.md')
    sealed_write(folder / 'PLAN.json', plan)
    return plan

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


def report(folder, plan):
    """Main metrics keep shared scoring; include planned controls and coverage."""
    result = core_report(folder, plan)
    folder = Path(folder)
    result['system'], result['protocol_version'] = plan['system'], plan['protocol']['version']
    result['controls'] = {}
    if plan['controls']:
        for name in ('no_memory','complete_evidence'):
            rows = []
            for claim in plan['claims']:
                path = folder/'controls'/claim['claim_id']/f'{name}.json'
                if not path.exists():
                    continue
                row = read(path)
                if (len(row['answers']) != ROTATION_COUNT or {a['trial'] for a in row['answers']} != TRIAL_IDS
                        or row['recovered'] != all(a['target_hit'] for a in row['answers'])):
                    raise ValueError('Incomplete or inconsistent control result')
                for a,rotation in zip(row['answers'],claim['rotations']):
                    rescore = score_response({'text':json.dumps(a['parsed']),'request_id':a['request_id'],
                        'finish_reason':'stop'},rotation,claim['target_choice_id'],claim['choices'])
                    if rescore != a:
                        raise ValueError('Control score differs from frozen choices')
                rows.append(row['recovered'])
            result['controls'][name] = {'denominator':len(plan['claims']), 'completed_claims':len(rows),
                'rate':sum(rows)/len(plan['claims']) if len(rows)==len(plan['claims']) else None}
    if plan['residual_plus']:
        result['residual_plus_gap'] = {}
        for condition in plan['conditions']:
            base = result['metrics'][f'residual-{condition}']['rate']
            extra = result['metrics'][f'residual_plus-{condition}']['rate']
            gap = extra-base if base is not None and extra is not None else None
            result['residual_plus_gap'][condition] = {'gap':gap,
                'needs_provenance_review':gap>.05 if gap is not None else None}
    result['main_complete'] = all(v['completed_claims']==len(plan['claims']) for v in result['metrics'].values())
    write_json(folder/'SUMMARY.json',result)
    return result
