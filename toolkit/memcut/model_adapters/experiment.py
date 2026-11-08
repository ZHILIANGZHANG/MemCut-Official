"""Additive GLM experiment entry; original plans, banks and hashes stay valid."""
import copy
from contextlib import contextmanager
import importlib
from pathlib import Path

from memcut.native_memory.persistence import write_json
from memcut.source_anchored_v2.runner import read, sha
from .glm import ENDPOINT, MODEL, PROFILE

ROOT = Path(__file__).resolve().parents[2]
MODULES = {'memorybank': 'memcut.source_anchored_v2.runner',
           'nemori': 'memcut.source_anchored_systems.runner',
           'lightmem': 'memcut.source_anchored_systems.runner',
           'raw_turn_rag': 'memcut.raw_turn_rag.runner'}


def runner(system):
    return importlib.import_module(MODULES[system])


def identity(system, plan):
    files = list(Path(__file__).parent.glob('*.py')) + [ROOT/'tools/glm_s012.py']
    return {'system': system, 'request_profile': copy.deepcopy(PROFILE),
            'roles': [role for role, model in plan['models'].items() if model == MODEL],
            'code': {str(p.relative_to(ROOT)): sha(p) for p in sorted(files)}}


def validate_reuse(plan):
    """A DS-written bank may be a DS-write/GLM-answer run, never a GLM Full."""
    reuse = plan.get('reuse_full')
    if not reuse or 'write' not in plan['models']:
        return
    source = Path(reuse['path']).parent.parent
    old = read(source/'PLAN.json')
    if old['models']['write'] != plan['models']['write']:
        raise ValueError('Full writing model differs; keep its original write model or build a new Full')
    for field in ('endpoint', 'compatibility'):
        if old['providers']['write'][field] != plan['providers']['write'][field]:
            raise ValueError('Reusable Full writing provider differs')
    if plan['models']['write'] == MODEL:
        extension = old.get('model_adapter', {})
        if extension.get('request_profile') != PROFILE or 'write' not in extension.get('roles', []):
            raise ValueError('Reusable GLM Full has no matching frozen request policy')


def prepare(folder, *, system, **kwargs):
    base = runner(system)
    native_eval_k = kwargs.pop('native_eval_k', 10) if system == 'memorybank' else None
    if system == 'memorybank' and (type(native_eval_k) is not int or native_eval_k <= 0):
        raise ValueError('MemoryBank native QA k must be a positive integer')
    defaults = dict(answer_model=MODEL, selector_model=MODEL, endpoint=ENDPOINT, compatibility='openai')
    if system != 'raw_turn_rag':
        defaults['write_model'] = MODEL
    if system in {'nemori', 'lightmem'}:
        defaults['system'] = system
    values = {**defaults, **kwargs}
    # Preflight before creating a run directory or loading any native weights.
    for role in ('write', 'answer', 'selector'):
        if values.get(role+'_model') == MODEL:
            endpoint = values.get(role+'_endpoint') or values['endpoint']
            compatibility = values.get(role+'_compatibility') or values['compatibility']
            if endpoint.rstrip('/') != ENDPOINT or compatibility != 'openai':
                raise ValueError('GLM model requires the official endpoint with compatibility=openai')
    if not any(values.get(r+'_model') == MODEL for r in ('write', 'answer', 'selector')):
        raise ValueError('Use the original entrypoint for runs without GLM')
    if values.get('reuse_full') and system != 'raw_turn_rag':
        validate_reuse({'models': {'write': values['write_model']},
            'providers': {'write': {'endpoint': values['endpoint'], 'compatibility': values['compatibility']}},
            'reuse_full': {'path': str(Path(values['reuse_full']).resolve())}})
    plan = base.prepare(folder, **values)
    if system == 'memorybank':
        plan['native_qa'] = {'top_k': native_eval_k, 'method': 'native FAISS search, adjacency expansion and renderer'}
    # This is a new plan before any build/dispatch; never rewrite an existing run.
    plan['model_adapter'] = identity(system, plan)
    write_json(Path(folder)/'PLAN.json', plan)
    return plan


def load_plan(folder):
    raw = read(Path(folder)/'PLAN.json')
    extension = raw.get('model_adapter')
    if not extension:
        raise ValueError('This is not a GLM adapter plan; use its original runner')
    system = extension['system']
    plan = runner(system).load_plan(folder)
    if extension != identity(system, plan):
        raise ValueError('Frozen GLM adapter changed; create a new versioned experiment')
    if system == 'memorybank':
        qa = plan.get('native_qa', {})
        if type(qa.get('top_k')) is not int or qa['top_k'] <= 0 or qa.get('method') != 'native FAISS search, adjacency expansion and renderer':
            raise ValueError('Missing or invalid frozen MemoryBank native QA setting')
    validate_reuse(plan)
    return plan


@contextmanager
def runtime(plan):
    """Apply the frozen QA k to every new/restored branch in this process.

    Native storage and formation identity remain unchanged; deletion retrieval
    still passes its explicit top20. The original class default is restored on exit.
    """
    if plan['model_adapter']['system'] != 'memorybank':
        yield
        return
    from unittest.mock import patch
    from memcut.source_anchored_v2.memorybank import SourceMemoryBank
    with patch.object(SourceMemoryBank, 'native_answer_top_k', plan['native_qa']['top_k']):
        yield


def usage_report(folder):
    """Count durable real receipts, not local cache reads; never guess GLM prices."""
    groups, pending = {}, []
    for request_file in sorted((Path(folder)/'requests').glob('*/request.json')):
        request = read(request_file)
        response_file = request_file.with_name('response.json')
        if not response_file.exists():
            pending.append(request_file.parent.name)
            continue
        response = read(response_file)
        usage = response.get('usage') or {}
        key = request['request']['model']
        row = groups.setdefault(key, {'requests': 0, 'prompt_tokens': 0, 'completion_tokens': 0,
            'cached_prompt_tokens': 0, 'reasoning_tokens': 0, 'missing_usage': 0,
            'missing_cached_detail': 0, 'missing_reasoning_detail': 0, 'non_stop_responses': 0})
        row['requests'] += 1
        for field in ('prompt_tokens', 'completion_tokens'):
            if type(usage.get(field)) is int:
                row[field] += usage[field]
            else:
                row['missing_usage'] += 1
        for group, field, total, missing in (
            ('prompt_tokens_details', 'cached_tokens', 'cached_prompt_tokens', 'missing_cached_detail'),
            ('completion_tokens_details', 'reasoning_tokens', 'reasoning_tokens', 'missing_reasoning_detail')):
            value = (usage.get(group) or {}).get(field)
            if type(value) is int:
                row[total] += value
            else:
                row[missing] += 1
        if any(c.get('finish_reason') != 'stop' for c in response.get('choices', [])):
            row['non_stop_responses'] += 1
    result = {'by_requested_model': groups, 'unresolved_request_ids': pending,
              'cost_cny': None, 'price_status': 'No verified official GLM price card; token usage only, not zero cost.',
              'reasoning_accounting': 'reasoning_tokens is a detail within completion_tokens, never added twice',
              'cache_accounting': 'one saved response per real request; local reuse adds no usage'}
    write_json(Path(folder)/'MODEL_USAGE.json', result)
    return result
