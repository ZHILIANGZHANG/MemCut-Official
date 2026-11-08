"""Offline regressions for the aligned entry's v2 behaviour: no remote clients, models or upstream checkouts."""
import json
import pickle
import re
import uuid
from types import SimpleNamespace

import pytest

from memcut.aligned_s012 import answers, transport
from memcut.aligned_s012.runtime import ANSWER_POLICY, runtime
from memcut.model_adapters import qwen
from memcut.native_memory.contract import digest
from memcut.source_anchored_v2 import evaluation, runner as core
from memcut.source_anchored_v2.completions import CachedCompletions
from memcut.source_anchored_v2.scoring import select_rotations
from tools import memcut_s012 as entry
from test_aligned_s012 import inputs, profile_fixture


@pytest.fixture(autouse=True)
def private_paid_lock(tmp_path, monkeypatch):
    monkeypatch.setenv('MEMCUT_PAID_LOCK', str(tmp_path / 'paid.lock'))


# ---------------------------------------------------------------- prepare

def named_trials(args):
    """Rewrite the fixture contract the way EP14/EP15 write theirs: rotations named "trial1"/"trial2"."""
    contract = core.read(args.contracts)
    ids = [c['id'] for c in contract['claims'][0]['choices']]
    contract['claims'][0]['rotations'] = [{**r, 'trial': f"trial{r['trial']}"} for r in select_rotations(None, ids)]
    core.write_json(args.contracts, contract)


@pytest.mark.parametrize('system', ['memorybank', 'nemori', 'lightmem'])
def test_string_trial_names_prepare_for_every_system(tmp_path, system, monkeypatch):
    profile_fixture(monkeypatch, system)
    args = inputs(tmp_path, system)
    named_trials(args)
    plan = entry.prepare(args)
    assert [r['trial'] for r in plan['claims'][0]['rotations']] == [1, 2]
    assert [c['before'] for c in plan['input_normalization']['changes']] == ['trial1', 'trial2']
    assert plan['aligned_answer_policy'] == ANSWER_POLICY and ANSWER_POLICY['max_tokens'] == 2048
    with runtime(system) as base:
        assert base.load_plan(args.output) == plan


def test_the_qwen_profile_is_checked_before_a_plan_is_frozen(tmp_path, monkeypatch):
    profile_fixture(monkeypatch, 'memorybank')
    args = inputs(tmp_path, 'memorybank')
    args.model, args.endpoint = qwen.MODEL, 'https://elsewhere.invalid/v1'
    with pytest.raises(ValueError, match='served from'):
        entry.prepare(args)
    assert not args.output.exists()


def test_an_interrupted_prepare_leaves_no_half_plan(tmp_path, monkeypatch):
    profile_fixture(monkeypatch, 'memorybank')
    args = inputs(tmp_path, 'memorybank')
    original = core.prepare

    def crash(folder, **kwargs):
        original(folder, **kwargs)
        raise KeyboardInterrupt
    monkeypatch.setattr(core, 'prepare', crash)
    with pytest.raises(KeyboardInterrupt):
        entry.prepare(args)
    assert not args.output.exists() and not list(tmp_path.glob('.run.preparing-*'))
    monkeypatch.setattr(core, 'prepare', original)
    assert entry.prepare(args)['alignment']['system'] == 'memorybank'     # the same command then works


# ---------------------------------------------------------------- transport

class StatusError(Exception):
    def __init__(self, status, body='{}'):
        super().__init__(f'HTTP {status}')
        self.status_code, self.body = status, body


class APIConnectionError(Exception):
    pass


class ConnectError(Exception):
    pass


def unsent():
    error = APIConnectionError('Connection error.')
    error.__cause__ = ConnectError('refused')
    return error


def ok(text='{"answer": "a"}'):
    return {'choices': [{'message': {'content': text}, 'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 3, 'completion_tokens': 2}}


def scripted(*steps):
    steps = list(steps)
    sent = []

    def create(**request):
        sent.append(request)
        step = steps.pop(0)
        if isinstance(step, BaseException):
            raise step
        return step
    return create, sent


def test_unbilled_errors_are_retried_inside_one_dispatch(tmp_path):
    create, sent = scripted(StatusError(429), unsent(), StatusError(503), ok())
    waits = []
    send = transport.policy(create, tmp_path, sleep=waits.append)
    assert send(model='m', messages=[]) == ok() and len(sent) == 4 and waits == [2.0, 4.0, 8.0]
    assert len((tmp_path / 'TRANSPORT_RETRIES.jsonl').read_text().splitlines()) == 3


@pytest.mark.parametrize('error,kind', [
    (StatusError(400, '{"error": {"code": "data_inspection_failed"}}'), 'refused'),
    (StatusError(400, 'Range of input length should be [1, 98304]'), 'refused'),
    (StatusError(401, '{"error": "invalid api key"}'), 'rejected'),
    (StatusError(400, '{"error": "unknown parameter"}'), 'rejected'),
    (StatusError(429, '{"error": {"code": "Arrearage"}}'), 'rejected'),
    (StatusError(503), 'retry'), (unsent(), 'retry'),
    (APIConnectionError('Connection error.'), 'unknown'), (TimeoutError('read'), 'unknown')])
def test_each_failure_has_one_outcome(error, kind):
    assert transport.outcome(error) == kind


def gateway(folder, create):
    cache = CachedCompletions(folder / 'requests', create, endpoint='offline-fixture')
    inner = SimpleNamespace(for_role=lambda role: cache, calls=cache.calls, close=lambda: None)
    return transport.TransportGateway(inner, sleep=lambda s: None), cache


def test_a_refusal_is_a_saved_reply_the_cache_rejects_and_replays_unsent(tmp_path):
    create, sent = scripted(StatusError(400, '{"error": {"code": "data_inspection_failed"}}'))
    g, cache = gateway(tmp_path, create)
    request = {'model': 'm', 'messages': [{'role': 'user', 'content': 'q'}], 'max_tokens': 9}
    for _ in range(2):
        with pytest.raises(RuntimeError, match='Incomplete/empty'):
            g.for_role('answer').complete(request)
    assert len(sent) == 1 and cache.calls[-1]['finish_reason'] == 'content_filter'
    saved = json.loads(next(tmp_path.glob('requests/*/response.json')).read_text())
    assert saved['memcut_refusal']['billed'] is False and not transport.pending(tmp_path)


def test_unbilled_stops_are_archived_and_unknown_ones_wait_for_resolve(tmp_path):
    create, sent = scripted(StatusError(401), TimeoutError('read timed out'), ok(), ok())
    g, cache = gateway(tmp_path, create)
    first = {'model': 'm', 'messages': [{'role': 'user', 'content': 'one'}]}
    second = {'model': 'm', 'messages': [{'role': 'user', 'content': 'two'}]}
    with pytest.raises(transport.NotBilled):
        g.for_role('answer').complete(first)
    with pytest.raises(TimeoutError):
        g.for_role('answer').complete(second)
    assert {r['error'] for r in transport.pending(tmp_path)} == {'NotBilled', 'TimeoutError'}
    assert len(transport.archive_not_billed(tmp_path)) == 1           # what execute() does before a stage
    assert [r['error'] for r in transport.pending(tmp_path)] == ['TimeoutError']
    with pytest.raises(RuntimeError, match='Unresolved model dispatch'):
        cache.complete(second)                                        # still blocked: its outcome is unknown
    result = entry.resolve(tmp_path, retry=True)
    assert len(result['archived']) == 1 and not transport.pending(tmp_path)
    assert cache.complete(first)['text'] and cache.complete(second)['text'] and len(sent) == 4
    log = [json.loads(line) for line in (tmp_path / 'requests/ARCHIVED.jsonl').read_text().splitlines()]
    assert [row['reason'].split(',')[0] for row in log] == ['provider error or unsent request', 'resolve --retry: sent']


# ---------------------------------------------------------------- answers

def claim(cid):
    ids = ['yes', 'n0', 'n1', 'n2']
    return {'claim_id': cid, 'stem': f'Question {cid}?', 'category': None, 'target_choice_id': 'yes',
            'choices': [{'id': 'yes', 'text': 'Known option'}] + [{'id': f'n{i}', 'text': f'Wrong {i}'} for i in range(3)],
            'rotations': select_rotations(None, ids)}


def answer_plan(tmp_path, claims, conditions=('Full', 'S0', 'S1@10', 'S2@10')):
    plan = {'claims': claims, 'conditions': list(conditions), 'controls': False, 'residual_plus': False,
            'models': {'answer': 'offline'}, 'aligned_answer_policy': ANSWER_POLICY}
    for c in claims:
        for condition in conditions:
            target = tmp_path / 'claims' / c['claim_id'] / condition
            target.mkdir(parents=True)
            (target / 'state.pkl').write_bytes(b'state')
            contexts = {}
            for family in ('native', 'residual'):
                text = f'{family} memory for {c["claim_id"]} {condition}'
                contexts[family] = {'context': text, 'context_sha256': digest(text), 'state_identity': condition,
                                    'empty': False}
            core.write_json(target / 'STATE.json', {'plan_sha256': digest(plan), 'state_identity': condition,
                                                    'snapshot_sha256': core.sha(target / 'state.pkl'),
                                                    'contexts': contexts})
    return plan


def answering(cut_off):
    """Answers every question correctly, except that a native readout of a claim in `cut_off` runs into its cap
    on its second rotation."""
    sent, seen = [], {}

    def create(**request):
        sent.append(request)
        body = request['messages'][-1]['content']
        assert request['max_tokens'] == 2048
        label = re.search(r'^([a-e])\. Known option$', body, re.M)[1]
        context = body.split('\n\nQuestion:')[0]
        seen[context] = seen.get(context, 0) + 1
        cut = context.startswith('native') and any(f'for {c} ' in context for c in cut_off) and seen[context] == 2
        text = '{"reason": "a long' if cut else json.dumps({'reason': 'ok', 'answer': label})
        return {'choices': [{'message': {'content': text}, 'finish_reason': 'length' if cut else 'stop'}]}
    return create, sent


def test_an_unusable_answer_is_unavailable_and_the_stage_goes_on(tmp_path):
    claims = [claim('c1'), claim('c2')]
    plan = answer_plan(tmp_path, claims)
    cache = CachedCompletions(tmp_path / 'requests', lambda **r: None, endpoint='offline-fixture')
    create, sent = answering(cut_off=['c2'])
    cache.create = create
    g = SimpleNamespace(for_role=lambda role: cache)
    result = answers.answer(tmp_path, plan, g, evaluate=evaluation.evaluate)
    assert result == {'unavailable_readouts': 4}                      # c2's native readout, in 4 conditions
    for condition in plan['conditions']:
        base = tmp_path / 'claims'
        assert (base / 'c1' / condition / 'native.json').exists()
        gone = core.read(base / 'c2' / condition / 'native.unavailable.json')['unavailable']
        assert gone['finish_reason'] == 'length' and gone['trial'] == 2 and gone['trials_answered_before'] == 1
        assert (base / 'c2' / condition / 'residual.json').exists()
    n = len(sent)
    assert answers.answer(tmp_path, plan, g, evaluate=evaluation.evaluate) == result and len(sent) == n
    cohort = answers.available_cohort(tmp_path, plan)
    assert cohort['families']['native']['claims'] == 1 and cohort['families']['native']['excluded_claims'] == ['c2']
    assert cohort['families']['residual']['claims'] == 2
    assert cohort['families']['native']['rates']['S0']['rate'] == 1.0
    assert cohort['families']['native']['paired_differences']['S2@10-minus-S1@10']['mean'] == 0
    assert len(cohort['unavailable_readouts']) == 4


def test_memorybank_early_stop_keeps_a_valid_miss_and_marks_an_unusable_first_trial(tmp_path):
    """With MemoryBank's early stop, a valid wrong trial 1 is a scored miss (one request); an unusable trial 1 is an
    unavailable readout, never a miss."""
    from memcut.answer_execution import early_stop
    plan = answer_plan(tmp_path, [claim('c1'), claim('c2')], conditions=('Full',))

    def create(**request):
        body = request['messages'][-1]['content']
        wrong = re.search(r'^([a-e])\. Wrong 0$', body, re.M)[1]
        if 'native memory for c1' in body:
            return {'choices': [{'message': {'content': json.dumps({'reason': 'x', 'answer': wrong})},
                                 'finish_reason': 'stop'}]}
        if 'native memory for c2' in body:
            return {'choices': [{'message': {'content': '{"reason": "x", "answer": "z"}'}, 'finish_reason': 'stop'}]}
        label = re.search(r'^([a-e])\. Known option$', body, re.M)[1]
        return {'choices': [{'message': {'content': json.dumps({'reason': 'x', 'answer': label})},
                             'finish_reason': 'stop'}]}
    cache = CachedCompletions(tmp_path / 'requests', create, endpoint='offline-fixture')
    result = answers.answer(tmp_path, plan, SimpleNamespace(for_role=lambda role: cache), evaluate=early_stop.evaluate)
    assert result == {'unavailable_readouts': 1}
    miss = core.read(tmp_path / 'claims/c1/Full/native.json')['result']
    assert miss['recovered'] is False and len(miss['answers']) == 1
    early_stop.validate_result(miss, plan['claims'][0], 'native')
    gone = core.read(tmp_path / 'claims/c2/Full/native.unavailable.json')['unavailable']
    assert gone['trial'] == 1 and 'Invalid/incomplete answer' in gone['error']


def test_a_cache_integrity_error_on_a_later_trial_still_stops_the_stage(tmp_path):
    plan = answer_plan(tmp_path, [claim('c1')], conditions=('Full',))
    good = {'choices': [{'message': {'content': '{"reason": "x", "answer": "a"}'}, 'finish_reason': 'stop'}]}

    class Cache:
        calls = []

        def complete(self, request, **kwargs):
            if self.calls:
                raise ValueError('Cached request identity changed')
            self.calls.append({'request_id': 'r1', 'finish_reason': 'stop'})
            return {'text': good['choices'][0]['message']['content'], 'request_id': 'r1', 'finish_reason': 'stop'}
    with pytest.raises(ValueError, match='identity changed'):
        answers.answer(tmp_path, plan, SimpleNamespace(for_role=lambda role: Cache()), evaluate=evaluation.evaluate)
    assert not list(tmp_path.glob('claims/c1/Full/*.unavailable.json'))


def test_a_frozen_state_error_still_stops_the_stage(tmp_path):
    plan = answer_plan(tmp_path, [claim('c1')], conditions=('Full',))
    state = core.read(tmp_path / 'claims/c1/Full/STATE.json')
    state['contexts']['native']['context'] = 'changed after freezing'
    core.write_json(tmp_path / 'claims/c1/Full/STATE.json', state)
    cache = CachedCompletions(tmp_path / 'requests', lambda **r: pytest.fail('no request'), endpoint='offline')
    with pytest.raises(ValueError):
        answers.answer(tmp_path, plan, SimpleNamespace(for_role=lambda role: cache), evaluate=evaluation.evaluate)


# ---------------------------------------------------------------- systems

@pytest.mark.parametrize('system,value,ok', [
    ('information requirement analyst', {}, True),
    ('information requirement analyst', {'minimal_queries_needed': '2', 'key_entities': []}, True),
    ('information requirement analyst', {'key_entities': 'Mira'}, False),
    ('query analysis assistant', {'keywords': ["Lin Yue's"]}, True),
    ('query analysis assistant', {'keywords': 'Lin Yue'}, False),
    ('query analysis assistant', {'time_expression': 5}, False),
    ('query generation specialist', {'reasoning': 'x'}, True),
    ('query generation specialist', {'queries': 'one'}, False),
    ('information completeness evaluator', {'assessment': 'partial'}, True),
    ('missing information query generator', {'targeted_queries': ['a', 'b', 'c', 'd']}, True),
    ('missing information query generator', {'targeted_queries': [{'q': 1}]}, False)])
def test_simplemem_planner_replies_default_like_upstream(system, value, ok):
    from memcut.aligned_s012.simplemem import validate_result
    if ok:
        validate_result(value, system)
    else:
        with pytest.raises(ValueError):
            validate_result(value, system)


def test_nemori_ids_have_the_native_uuid_format_and_bm25_needs_lemmas():
    from memcut.aligned_s012.nemori import lemmatizing, native_id
    ident = native_id('aligned-episode', ['t1', 't2'], 'Title', 'Body')
    assert uuid.UUID(ident).version == 5 and ident == native_id('aligned-episode', ['t1', 't2'], 'Title', 'Body')
    assert lemmatizing(SimpleNamespace(pipe_names=['tok2vec', 'tagger', 'lemmatizer']))
    assert not lemmatizing(SimpleNamespace(pipe_names=['sentencizer'])) and not lemmatizing(None)


def test_lightmem_records_the_turns_its_final_forced_cut_drops(monkeypatch):
    from memcut.aligned_s012.lightmem import AlignedLightMemBank
    from memcut.source_anchored_systems.formation_runtime.lightmem import FastLightMemBank

    class Buffers:
        def add_messages(self, messages, segmenter, embedder):
            return [[{'role': 'user', '_memcut_source_id': 't8'}, {'role': 'assistant', '_memcut_source_id': 't8'}],
                    [{'role': 'user', '_memcut_source_id': 't9'}]]

        def cut_with_segmenter(self, segmenter, embedder, force_segment=False):
            return [[{'role': 'user', '_memcut_source_id': 't7'}]]

    def upstream_final_call(self, turn_id, completions, *, final=False):
        manager = self.memory.senmem_buffer_manager
        manager.add_messages([], None, None)
        return manager.cut_with_segmenter(None, None, force_segment=final)   # replaces what was just produced

    monkeypatch.setattr(FastLightMemBank, 'ingest', upstream_final_call)
    bank = AlignedLightMemBank.__new__(AlignedLightMemBank)
    bank.memory, bank.progress = SimpleNamespace(senmem_buffer_manager=Buffers()), {}
    bank.ingest('t9', None, final=True)
    assert bank.progress['diagnostics']['final_segments_dropped_turns'] == ['t8', 't9']
    assert 'add_messages' not in vars(bank.memory.senmem_buffer_manager)    # the native methods are back


def test_formation_diagnostics_name_unencoded_turns_and_the_claims_they_touch(tmp_path):
    full = tmp_path / 'full/full.pkl'
    full.parent.mkdir()
    full.write_bytes(pickle.dumps({'turns': {'t1': {}, 't2': {}, 't3': {}},
                                   'lineage': {'der': {'t1': ['o1'], 't2': [], 't3': ['o2']}},
                                   'progress': {'diagnostics': {'final_segments_dropped_turns': ['t2']}}}))
    plan = {'alignment': {'system': 'lightmem'},
            'claims': [{'claim_id': 'c1', 'evidence_turns': ['t2'], 'disclosure_turns': ['t3']},
                       {'claim_id': 'c2', 'evidence_turns': ['t1'], 'disclosure_turns': ['t3']}]}
    result = entry.write_diagnostics(tmp_path, plan)
    assert result['turns_without_memory'] == 1 and result['claims_with_unencoded_turns'] == {'c1': ['t2']}
    assert result['system'] == {'final_segments_dropped_turns': ['t2']}
    assert core.read(tmp_path / 'full/DIAGNOSTICS.json') == result


# ---------------------------------------------------------------- native (opt in: MEMCUT_ALIGNED_NATIVE_TESTS=1)

def native_environment(dependency):
    import os
    if os.environ.get('MEMCUT_ALIGNED_NATIVE_TESTS') != '1':
        pytest.skip('Opt in with MEMCUT_ALIGNED_NATIVE_TESTS=1 in a prepared native runtime')
    pytest.importorskip(dependency)


def test_nemori_flush_semantic_times_and_a_buffered_checkpoint(tmp_path):
    """The final partial batch becomes one "Force episode creation" episode without a segmentation or merge
    request; a checkpoint holding buffered messages restores; semantic lines show their stored time."""
    native_environment('chromadb')
    import copy
    from memcut.aligned_s012.nemori import AlignedNemoriBank
    from memcut.aligned_s012.runtime import native_profile
    from memcut.source_anchored_systems.profiles import profile
    from aligned_fixtures import source_turns

    class Embedding:
        model, embedding_dim = 'fixture', 384

        def embed_text(self, text):
            return [1.0] + [0.0] * 383

        def embed_texts(self, texts):
            return SimpleNamespace(embeddings=[self.embed_text(t) for t in texts], usage={})

        def cosine_similarity(self, a, b):
            return 1.0

    categories = []

    def create(**q):
        prompt = '\n'.join(m['content'] for m in q['messages'])
        if 'response_format' not in q:
            categories.append('prediction')
            return dict(choices=[dict(message=dict(content='Predicted receipt.'), finish_reason='stop')])
        if '"indices"' in prompt:
            categories.append('segmentation')
            value = {'episodes': [{'indices': [1, 2, 3, 4]}]}
        elif 'merge_target_id' in prompt:
            categories.append('merge')
            value = dict(decision='new', reason='Different')
        elif 'statements' in prompt:
            categories.append('statements')
            value = {'statements': [f'Receipt statement {len(categories)}']}
        else:
            categories.append('episode')
            value = dict(title='Receipts', content=f'Receipt codes {len(categories)}', timestamp='2026-01-01T12:00:01')
        return dict(choices=[dict(message=dict(content=json.dumps(value)), finish_reason='stop')])

    settings = {**native_profile(profile, 'nemori')['settings'], 'model': 'fixture', 'batch_threshold': 4}
    turns = source_turns(6)
    cache = CachedCompletions(tmp_path / 'requests', create, endpoint='offline-fixture')
    bank = AlignedNemoriBank(settings, turns, embedder=Embedding())
    try:
        bank._is_duplicate = lambda *_: False
        for tid in ('t1', 't2', 't3', 't4', 't5'):
            bank.ingest(tid, cache)
        buffered = copy.deepcopy(bank.export())          # t5 still in the buffer
        assert buffered['native']['buffer']['messages']
        bank.restore_value(buffered)                      # used to fail: last_updated restamped
        bank.ingest('t6', cache)
        before = categories.count('segmentation')
        bank.finish(cache)
        assert categories.count('segmentation') == before == 1 and 'merge' not in categories
        episodes = bank.episodes.list_by_user(bank.user)
        assert [e.boundary_reason for e in episodes][-1] == 'Force episode creation'
        assert [m['message_id'] for m in episodes[-1].original_messages] == ['t5', 't6']
        assert all(uuid.UUID(e.episode_id).version == 5 for e in episodes)
        assert bank.trace(bank.oid('episodic', episodes[-1].episode_id)) == {'t5', 't6'}
        import asyncio
        prompt = asyncio.run(bank.native_context('Which receipt code?')).native_context['prompt']
        assert '- [] ' not in prompt and '- [2026-01-01' in prompt
    finally:
        bank.close()


def test_simplemem_rejected_keyword_queries_and_a_reproducible_index(tmp_path):
    native_environment('lancedb')
    import numpy as np
    from memcut.aligned_s012.simplemem import native as simplemem_native
    from memcut.simplemem_s012.client import NativeAbort
    from memcut.simplemem_s012.native import Store
    from simplemem.core.models.memory_entry import MemoryEntry

    class Embedding:
        dimension = 8

        def encode_documents(self, texts):
            return np.asarray([np.eye(8, dtype=np.float32)[i % 8] for i, _ in enumerate(texts)])

        def encode_single(self, text, is_query=False):
            return self.encode_documents([text])[0]

    store = Store(tmp_path, Embedding(), 'memory_entries')
    store.add_entries([MemoryEntry(entry_id=f'e{i:02}', lossless_restatement=f"Lin Yue's trip note {i % 3}")
                       for i in range(40)])
    with simplemem_native():
        store.backend.build_fts()
        assert store.keyword_search(["Lin Yue's"], top_k=5) == []
        assert store.calls[-1]['query_rejected'].startswith('Syntax Error')
        orders = set()
        for _ in range(8):
            store.backend.build_fts()
            orders.add(tuple(e.entry_id for e in store.keyword_search(['trip'], top_k=5)))
        assert len(orders) == 1 and len(next(iter(orders))) == 5
    with pytest.raises(NativeAbort):                       # the frozen entries keep their strict behaviour
        store.keyword_search(["Lin Yue's"], top_k=5)
