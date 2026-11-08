"""GLM request identity, failure handling and current S012 pipeline integration."""
import asyncio
import copy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from memcut.model_adapters import experiment
from memcut.model_adapters.glm import ENDPOINT, MODEL, PROFILE, GLMCompletions, Gateway, normalize
from memcut.native_memory.contract import digest
from memcut.native_memory.persistence import write_json
from memcut.source_anchored_systems.formation_runtime.concurrency import ConcurrentCompletions
from memcut.source_anchored_v2.completions import CachedCompletions
from test_raw_turn_rag import prepared, response as fixture_response


def request():
    return {'model': MODEL, 'temperature': 0, 'max_tokens': 400,
            'response_format': {'type': 'json_object'},
            'messages': [{'role': 'user', 'content': 'fixture'}]}


def response(finish='stop', content='{"answer":"a"}'):
    return {'model': MODEL, 'choices': [{'finish_reason': finish,
        'message': {'content': content, 'reasoning_content': 'Private reasoning is never context.'}}],
        'usage': {'prompt_tokens': 100, 'completion_tokens': 30,
                  'prompt_tokens_details': {'cached_tokens': 40},
                  'completion_tokens_details': {'reasoning_tokens': 20}}}


def test_actual_params_cached_identity_and_no_reasoning_leak(tmp_path):
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return response()
    cache = GLMCompletions(tmp_path/'requests', create)
    original = request()
    result = cache.complete(original)
    assert original == request()
    assert calls == [normalize(original)]
    assert calls[0]['max_tokens'] == 400
    assert calls[0]['extra_body'] == {'thinking': {'type': 'enabled'}, 'reasoning_effort': 'low'}
    assert result['text'] == '{"answer":"a"}'
    identity = json.loads((tmp_path/'requests'/result['request_id']/'request.json').read_text())
    assert identity['request'] == calls[0] and digest(identity) == result['request_id']
    assert cache.complete(original)['cached']
    report = experiment.usage_report(tmp_path)
    assert report['by_requested_model'][MODEL]['requests'] == 1
    assert report['by_requested_model'][MODEL]['completion_tokens'] == 30
    assert report['by_requested_model'][MODEL]['reasoning_tokens'] == 20
    assert report['cost_cny'] is None
    # A historical unconfigured OpenAI request cannot reuse the GLM-low receipt.
    old = CachedCompletions(tmp_path/'requests', create, endpoint=ENDPOINT)
    assert not old.complete(original)['cached']
    assert len(calls) == 2


@pytest.mark.parametrize('change', [
    {'model': 'glm-5-flash'}, {'reasoning_effort': 'max'},
    {'extra_body': {'thinking': {'type': 'disabled'}}}])
def test_conflicting_model_or_reasoning_fails_before_api(tmp_path, change):
    def never(**kwargs):
        pytest.fail('Must reject before dispatch')
    with pytest.raises(ValueError):
        GLMCompletions(tmp_path, never).complete({**request(), **change})


def test_concurrent_aliases_share_one_real_request(tmp_path):
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return response()
    cache = GLMCompletions(tmp_path, create)
    wrapper = ConcurrentCompletions(cache, 4)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(wrapper.complete, [request(), normalize(request())]*3))
    assert len(calls) == 1 and sum(not r['cached'] for r in results) == 1


def test_incomplete_or_unknown_dispatch_never_becomes_zero_or_retry(tmp_path):
    incomplete = GLMCompletions(tmp_path/'length', lambda **kw: response('length'))
    with pytest.raises(RuntimeError, match='Incomplete'):
        incomplete.complete(request())
    assert incomplete.complete(request(), allow_length=True)['finish_reason'] == 'length'
    empty = GLMCompletions(tmp_path/'empty', lambda **kw: response(content=''))
    with pytest.raises(RuntimeError):
        empty.complete(request(), allow_length=True)
    calls = []
    def fail(**kwargs):
        calls.append(1)
        raise TimeoutError()
    cache = GLMCompletions(tmp_path/'unknown', fail)
    with pytest.raises(TimeoutError):
        cache.complete(request())
    with pytest.raises(RuntimeError, match='Unresolved'):
        cache.complete(request())
    assert len(calls) == 1


def test_rag_complete_protocol_with_glm_gateway_and_reuse(prepared):
    old_folder, old_plan, old_factory, kwargs = prepared
    base = experiment.runner('raw_turn_rag')
    full = base.build(old_folder, old_plan, factory=old_factory)
    folder = old_folder.parent/'glm'
    kwargs.update(answer_model=MODEL, endpoint=ENDPOINT, selector_model=MODEL,
                  compatibility='openai', reuse_full=full)
    plan = experiment.prepare(folder, system='raw_turn_rag', **kwargs)
    assert experiment.load_plan(folder) == plan
    assert base.load_plan(old_folder) == old_plan
    factory = base.Factory(plan, embedder=old_factory.embedder)
    calls = []
    def fake(role, sent):
        assert sent['extra_body']['reasoning_effort'] == 'low'
        calls.append(role)
        return fixture_response(**sent)
    gateway = Gateway(folder, plan, create_factory=fake)
    embedded = len(factory.embedder.inputs)
    base.build(folder, plan, factory=factory)
    assert len(factory.embedder.inputs) == embedded
    asyncio.run(base.delete(folder, plan, gateway, factory=factory))
    base.answer(folder, plan, gateway)
    report = base.report(folder, plan)
    assert report['main_complete'] and report['scoring']['rotation_count'] == 2
    assert set(calls) == {'answer', 'selector'}
    for cid in ('c1', 'c2'):
        receipt = base.read(folder/f'claims/{cid}/S2@10/RECEIPT.json')
        assert receipt['actual_k'] == 10 and receipt['deletion_llm_calls'] == 0
        pool = base.read(folder/f'claims/{cid}/POOL.json')
        assert pool['requested_objects'] == pool['requested_candidate_turns'] == 20
    before = len(calls)
    base.answer(folder, plan, gateway)
    assert len(calls) == before
    changed = copy.deepcopy(plan)
    changed['model_adapter']['request_profile']['reasoning_effort'] = 'max'
    write_json(folder/'PLAN.json', changed)
    with pytest.raises(ValueError, match='Frozen GLM'):
        experiment.load_plan(folder)
    gateway.close()


def test_ds_full_cannot_be_claimed_as_glm_full(tmp_path):
    full = tmp_path/'ds'/'full'/'full.pkl'
    full.parent.mkdir(parents=True)
    old = {'models': {'write': 'deepseek-v4-flash'},
           'providers': {'write': {'endpoint': 'https://api.deepseek.com', 'compatibility': 'deepseek'}}}
    write_json(full.parent.parent/'PLAN.json', old)
    plan = {'models': {'write': MODEL}, 'providers': {'write': {'endpoint': ENDPOINT, 'compatibility': 'openai'}},
            'reuse_full': {'path': str(full)}}
    with pytest.raises(ValueError, match='writing model differs'):
        experiment.validate_reuse(plan)
    plan.update(models=old['models'], providers=old['providers'])
    experiment.validate_reuse(plan)


@pytest.mark.parametrize('system', list(experiment.MODULES))
def test_all_runners_prepare_freezes_adapter_without_changing_native_protocol(tmp_path, system):
    base = experiment.runner(system)
    path = tmp_path/system
    def prepare(folder, **kwargs):
        assert kwargs['endpoint'] == ENDPOINT and kwargs['compatibility'] == 'openai'
        roles = ['answer', 'selector'] + ([] if system == 'raw_turn_rag' else ['write'])
        assert all(kwargs[r+'_model'] == MODEL for r in roles)
        plan = {'models': {r: MODEL for r in roles}, 'protocol': copy.deepcopy(base.PROTOCOL)}
        write_json(Path(folder)/'PLAN.json', plan)
        return plan
    with patch.object(base, 'prepare', prepare):
        plan = experiment.prepare(path, system=system)
    assert plan['protocol'] == base.PROTOCOL
    assert plan['model_adapter'] == experiment.identity(system, plan)
    if system == 'memorybank':
        from memcut.source_anchored_v2.memorybank import SourceMemoryBank
        before = SourceMemoryBank.native_answer_top_k
        assert plan['native_qa']['top_k'] == 10
        with experiment.runtime(plan):
            assert SourceMemoryBank.native_answer_top_k == 10
        assert SourceMemoryBank.native_answer_top_k == before
