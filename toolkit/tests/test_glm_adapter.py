"""GLM request identity, failure handling and current S012 pipeline integration."""
import asyncio
import copy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from unittest.mock import patch

import pytest

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






