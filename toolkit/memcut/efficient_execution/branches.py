"""Explicit reuse of completed, audited Nemori/LightMem deletion branches.

No retrieval, generation or scoring runs here. Only answer identities may vary;
every imported context retains its original bytes and object/state identities.
"""
import copy
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

from memcut.native_memory.contract import digest
from memcut.model_adapters import experiment
from memcut.model_adapters.glm import GLMCompletions, MODEL
from memcut.source_anchored_v2 import runner as core
from memcut.source_anchored_v2.candidates import rank_s2, verify_pool
from memcut.source_anchored_v2.completions import CachedCompletions, parse_json
from memcut.execution_optimization.receipts import request_identity

POLICY = 'four-system-audited-branches-v1'


def _binding(plan):
    if system_of(plan) not in {'memorybank', 'raw_turn_rag', 'nemori', 'lightmem'}:
        raise ValueError('Branch reuse currently supports Nemori/LightMem only')
    if plan.get('controls') or plan.get('retention') or plan.get('residual_plus'):
        raise ValueError('Branch reuse requires the two-readout experiment without extra controls')
    result = copy.deepcopy(plan)
    for key in ('claims', 'reuse_full', 'build_execution', 'execution_optimization', 'answer_policy', 'answer_execution', 'efficient_execution'):
        result.pop(key, None)
    result['models'].pop('answer')
    result['providers'].pop('answer')
    if 'model_adapter' in result:
        result['model_adapter']['roles'] = [r for r in result['model_adapter']['roles'] if r != 'answer']
        if not result['model_adapter']['roles']:
            result.pop('model_adapter')  # An answer-only adapter cannot affect stored branches.
    return result


def _compatible(source, target):
    if _binding(source) != _binding(target):
        raise ValueError('Formation, selector, retrieval or deletion identity differs')
    source_claims = {c['claim_id']: c for c in source['claims']}
    target_claims = {c['claim_id']: c for c in target['claims']}
    if not target_claims or len(target_claims) != len(target['claims']):
        raise ValueError('Empty or duplicate claim selection')
    for cid, claim in target_claims.items():
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', cid) or source_claims.get(cid) != claim:
            raise ValueError('Source claim identity differs or is missing')
    return target_claims


def _source_plan(folder):
    raw = core.read(folder / 'PLAN.json')
    # Original frozen core/profile/source/model-adapter checks remain mandatory.
    plan = experiment.load_plan(folder) if 'model_adapter' in raw else experiment.runner(system_of(raw)).load_plan(folder)
    adapter = plan.get('settings', {}).get('response_schema_adapter')
    if adapter and core.sha(adapter['path']) != adapter['sha256']:
        raise ValueError('Source Full formation adapter changed')
    for relative, expected in plan.get('execution_optimization', {}).get('code', {}).items():
        if core.sha(experiment.ROOT / relative) != expected:
            raise ValueError('Source frozen execution implementation changed')
    for key in ('efficient_execution', 'answer_execution'):
        for relative, expected in plan.get(key, {}).get('code', {}).items():
            if core.sha(experiment.ROOT/relative)!=expected:raise ValueError('Inherited execution code changed')
    return plan


def _seal(path):
    seal = core.read(path.with_suffix('.frozen.json'))
    if core.sha(path) != seal['sha256']:
        raise ValueError('Changed frozen native snapshot')
    if 'state_identity' not in seal and 'native_state' in seal:
        seal={**seal,'state_identity':seal['native_state']}
    return seal


def system_of(plan):
    return plan.get('system', plan.get('model_adapter', {}).get('system', 'memorybank'))


def _selector(folder, plan, pool, ranking):
    from .replay import selector_receipts
    return selector_receipts(folder, plan, pool, ranking)


def _claim_files(folder, plan, claim, full_seal):
    cid = claim['claim_id']
    root = folder / 'claims' / cid
    audit = core.read(root / 'STATE_AUDIT.json')
    if (audit.get('status') != 'PASS' or audit.get('claim_id') != cid or
            [r['condition'] for r in audit.get('states', [])] != plan['conditions']):
        raise ValueError('Missing complete successful native-state audit')
    pool, ranking = core.read(root / 'POOL.json'), core.read(root / 'S2_ORDER.json')
    verify_pool(pool)
    if (pool['request'] != claim['deletion_request'] or
            pool['requested_objects'] != plan['protocol']['n_obj'] or
            pool.get('requested_candidate_turns', 20) != plan['protocol']['n_turn'] or
            (system_of(plan) != 'memorybank' and (ranking.get('pool_sha256') != pool['pool_sha256'] or ranking.get('plan_sha256') != digest(plan)))):
        raise ValueError('Candidate or selector budget/binding differs')
    turn_order = [r['turn_id'] for r in pool['turns']]
    if len(set(turn_order)) != len(turn_order) or set(ranking['order']) != set(turn_order) or len(ranking['order']) != len(turn_order):
        raise ValueError('Incomplete source-turn ordering')
    files = [root / p for p in ('STATE_AUDIT.json', 'POOL.json', 'S2_ORDER.json')]
    files += _selector(folder, plan, pool, ranking)
    for condition in plan['conditions']:
        state = core.validate_state(folder, plan, claim, condition)
        path = root / condition
        seal = _seal(path / 'state.pkl')
        if (seal['state_identity'] != state['state_identity'] or state['claim_id'] != cid or
                state['condition'] != condition or set(state['contexts']) != {'native', 'residual'}):
            raise ValueError('Frozen state/readout identity differs')
        for family, context in state['contexts'].items():
            if type(context.get('empty')) is not bool or len(context['obj_ids']) != len(set(context['obj_ids'])):
                raise ValueError('Malformed frozen readout')
            if family == 'native' and (context['empty'] or context['retrieval']['query'] != claim['stem']):
                raise ValueError('Native readout differs from the claim question')
            if family == 'residual' and context['empty'] != (not context['obj_ids']):
                raise ValueError('Residual structural-empty flag differs from its objects')
        if condition == 'Full':
            if seal != full_seal or state['deletion']:
                raise ValueError('Claim Full differs from the source Full')
        else:
            receipt = core.read(path / 'RECEIPT.json')
            if receipt != state['deletion'] or receipt.get('deletion_llm_calls') != 0 or receipt.get('rewritten_objs') or receipt.get('created_objs'):
                raise ValueError('Missing or incompatible whole-object deletion receipt')
            if receipt.get('deletion_policy') != plan['protocol']['deletion_policy'] or not receipt.get('selected_turns_have_empty_der'):
                raise ValueError('Deletion closure was not verified')
            expected = set(claim['disclosure_turns'])
            if condition != 'S0':
                strategy, size = condition.split('@')
                k = int(size)
                order = turn_order if strategy == 'S1' else ranking['order']
                if strategy not in {'S1', 'S2'} or k not in plan['k_list']:
                    raise ValueError('Unplanned deletion condition')
                expected.update(order[:k])
                if (receipt['extra_deleted_turns'] != order[:k] or receipt['requested_k'] != k or
                        receipt['actual_k'] != min(k, len(order)) or receipt['budget_complete'] != (len(order) >= k) or
                        state['pool_sha256'] != pool['pool_sha256']):
                    raise ValueError('Saved branch differs from the selected deletion budget')
            elif state['state_identity'] != pool['state_identity']:
                raise ValueError('Shared S0 candidate pool differs')
            if set(receipt['all_deleted_turns']) != expected:
                raise ValueError('Saved branch selected different source turns')
            files.append(path / 'RECEIPT.json')
        files.extend(path / p for p in ('STATE.json', 'state.pkl', 'state.frozen.json'))
    from .replay import validate_source_answers
    files += validate_source_answers(folder, plan, claim)
    return files


def prepare_catalog(source, plan):
    """Freeze verified source assets for the target's complete planned claims."""
    source = Path(source).resolve()
    old = _source_plan(source)
    claims = _compatible(old, plan)
    complete = core.read(source / 'COMPLETE.json') if (source / 'COMPLETE.json').exists() else None
    audit = core.read(source / 'RESULT_AUDIT.json')
    if audit.get('status') != 'PASS' or audit.get('claims') != len(old['claims']):
        raise ValueError('Require a completed independently audited source run')
    if audit.get('states', audit.get('state_audits')) != len(old['claims']) * len(old['conditions']):
        raise ValueError('Source audit does not cover every state')
    if audit.get('full_sha256') != core.sha(source/'full/full.pkl') and not audit.get('full_unchanged'):
        raise ValueError('Source audit does not bind the Full')
    summary = core.read(source/'SUMMARY.json')
    if len(summary['metrics']) != 8 or any(v['completed_claims'] != len(old['claims']) or
            v['denominator'] != len(old['claims']) or v['rate'] is None for v in summary['metrics'].values()):
        raise ValueError('Source experiment is incomplete')
    full_seal = _seal(source / 'full/full.pkl')
    files = [source / p for p in ('PLAN.json', 'SUMMARY.json', 'RESULT_AUDIT.json',
                                  'full/full.pkl', 'full/full.frozen.json')]
    if (source/'PROTOCOL.md').exists():files.append(source/'PROTOCOL.md')
    files += [p for p in source.glob('*.json') if p not in files]
    files += [p for p in (source/'full').glob('*.json') if p not in files]
    claim_files = {}
    for cid, claim in claims.items():
        selected = _claim_files(source, old, claim, full_seal)
        claim_files[cid] = [str(p.relative_to(source)) for p in selected]
        files.extend(selected)
    hashes = {str(p.relative_to(source)): core.sha(p) for p in files}
    # Hashes must cover the same version that passed the independent checks.
    # This is a one-time catalog operation, never a new native retrieval.
    if digest(_source_plan(source)) != digest(old):
        raise ValueError('Source plan changed during catalog preparation')
    for cid, claim in claims.items():
        checked = [str(p.relative_to(source)) for p in _claim_files(source, old, claim, full_seal)]
        if checked != claim_files[cid]:
            raise ValueError('Source branch manifest changed during catalog preparation')
    if any(core.sha(source / p) != h for p, h in hashes.items()):
        raise ValueError('Source branch changed during catalog preparation')
    return {'policy': POLICY, 'source': str(source), 'source_plan_sha256': digest(old),
            'target_binding': digest({'identity': _binding(plan), 'claims': list(claims.values())}),
            'source_full': full_seal, 'files': hashes, 'claims': claim_files}


def _import_selector_receipts(folder, source, catalog, cid):
    groups = {}
    allowed = {'request.json', 'response.json', 'usage.json', 'NUMERIC_SCORE_NORMALIZATION.json', 'RETRY_PROOF.json'}
    for relative in catalog['claims'][cid]:
        parts = Path(relative).parts
        if parts and parts[0] == 'requests':
            if len(parts) != 3 or not re.fullmatch(r'[0-9a-f]{64}', parts[1]) or parts[2] not in allowed:
                raise ValueError('Invalid selector receipt manifest')
            groups.setdefault(parts[1], {})[parts[2]] = catalog['files'][relative]
    root = folder / 'requests'
    locks = root / '.receipt_reuse_locks'
    if groups:
        locks.mkdir(parents=True, exist_ok=True)
    for rid, files in groups.items():
        if not {'request.json', 'response.json'} <= files.keys():
            raise ValueError('Incomplete source selector receipt')
        with (locks / rid).open('a+b') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            target, original = root / rid, source / 'requests' / rid
            if target.exists():
                # Never conceal an unknown dispatch or replace another sample.
                if not (target / 'request.json').is_file() or not (target / 'response.json').is_file():
                    raise RuntimeError('Unresolved local selector request; branch reuse cannot repair it')
                if any(not (target / name).is_file() or core.sha(target / name) != expected
                       for name, expected in files.items()):
                    raise ValueError('Existing selector receipt differs or is incomplete')
                continue
            pending = Path(tempfile.mkdtemp(prefix='.import-branch-', dir=root))
            try:
                for name, expected in files.items():
                    if core.sha(original / name) != expected:
                        raise ValueError('Source selector receipt changed')
                    shutil.copyfile(original / name, pending / name)
                    if core.sha(pending / name) != expected or core.sha(original / name) != expected:
                        raise ValueError('Source selector receipt changed during import')
                core.write_json(pending / 'REUSED_RECEIPT.json', {
                    'directory': str(original), 'files': files,
                    'branch_catalog_sha256': digest(catalog), 'policy': POLICY})
                os.rename(pending, target)
            finally:
                if pending.exists():
                    shutil.rmtree(pending)
    return list(groups)


def import_claims(folder, plan, catalog, claim_ids=None):
    """Atomically import each whole claim after the destination Full is frozen.

    Existing claim directories are accepted only when exactly imported by this
    catalog already. Unknown partial runs are never overwritten or repaired.
    """
    folder, source = Path(folder).resolve(), Path(catalog['source']).resolve()
    if catalog.get('policy') != POLICY or folder == source:
        raise ValueError('Invalid branch reuse policy or destination')
    old = _source_plan(source)
    claims = _compatible(old, plan)
    if (digest(old) != catalog['source_plan_sha256'] or
            digest({'identity': _binding(plan), 'claims': list(claims.values())}) != catalog['target_binding']):
        raise ValueError('Frozen branch reuse catalog belongs to another plan')
    if catalog['files'].get('full/full.pkl') != catalog['source_full']['sha256']:
        raise ValueError('Invalid catalog Full binding')
    for relative, expected in catalog['files'].items():
        path = source / relative
        if not path.resolve().is_relative_to(source) or core.sha(path) != expected:
            raise ValueError('Reusable branch artifact changed after catalog preparation')
    full = _seal(folder / 'full/full.pkl')
    if (full['state_identity'] != catalog['source_full']['state_identity'] or
            core.read(folder / 'full/IDENTITY.json')['plan_sha256'] != digest(plan) or
            (full['sha256'] != catalog['source_full']['sha256'] and
             plan.get('reuse_full', {}).get('sha256') != catalog['source_full']['sha256'])):
        raise ValueError('Destination Full differs from the audited source Full')
    selected = list(claims) if claim_ids is None else list(claim_ids)
    if len(selected) != len(set(selected)) or not set(selected) <= claims.keys() or not set(selected) <= catalog['claims'].keys():
        raise ValueError('Unknown or duplicate claim import selection')
    imported, selector_receipts = [], set()
    for cid in selected:
        selector_receipts.update(_import_selector_receipts(folder, source, catalog, cid))
        target = folder / 'claims' / cid
        provenance = {'policy': POLICY, 'source': str(source), 'claim_id': cid,
                      'source_plan_sha256': digest(old), 'plan_sha256': digest(plan),
                      'catalog_sha256': digest(catalog),
                      'source_files': {r: catalog['files'][r] for r in catalog['claims'][cid]}}
        if target.exists():
            note = core.read(target / 'REUSED_BRANCH.json')
            if {k: v for k, v in note.items() if k != 'imported_files'} != provenance:
                raise ValueError('Destination already contains another or partial claim')
            if any(core.sha(target / p) != h for p, h in note['imported_files'].items()):
                raise ValueError('Previously imported branch changed')
            for condition in plan['conditions']:
                core.validate_state(folder, plan, claims[cid], condition)
            imported.append(cid)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        pending = Path(tempfile.mkdtemp(prefix='.branch-import-', dir=target.parent))
        try:
            prefix = f'claims/{cid}/'
            for relative in catalog['claims'][cid]:
                if not relative.startswith(prefix) or Path(relative).name in {'native.json', 'residual.json'}:
                    continue  # Selector receipts were imported atomically above.
                old_path, new_path = source / relative, pending / relative[len(prefix):]
                new_path.parent.mkdir(parents=True, exist_ok=True)
                if core.sha(old_path) != catalog['files'][relative]:
                    raise ValueError('Source branch changed during import')
                if old_path.name in {'STATE.json', 'S2_ORDER.json'}:
                    value = core.read(old_path)
                    if old_path.name=='STATE.json' or 'plan_sha256' in value:value['plan_sha256'] = digest(plan)
                    core.write_json(new_path, value)
                else:
                    # Independent copies prevent destination writes from modifying old assets.
                    shutil.copyfile(old_path, new_path)
                    if core.sha(new_path) != catalog['files'][relative]:
                        raise ValueError('Source branch changed while copying')
                if core.sha(old_path) != catalog['files'][relative]:
                    raise ValueError('Source branch changed during import')
            imported_hashes = {str(p.relative_to(pending)): core.sha(p) for p in pending.rglob('*') if p.is_file()}
            core.write_json(pending / 'REUSED_BRANCH.json', {**provenance, 'imported_files': imported_hashes})
            os.rename(pending, target)
        finally:
            if pending.exists():
                shutil.rmtree(pending)
        for condition in plan['conditions']:
            core.validate_state(folder, plan, claims[cid], condition)
        imported.append(cid)
    return {'policy': POLICY, 'claims': imported, 'api_calls': 0, 'answers_copied': 0,
            'reused_selector_request_ids': sorted(selector_receipts)}
