"""RAG branches, vector deletion, exact provenance, caching and shared scoring."""
import asyncio
import copy
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np
import pytest

from memcut.native_memory.persistence import write_json
from memcut.raw_turn_rag import runner
from memcut.raw_turn_rag.bank import RawTurnBank
from memcut.raw_turn_rag.embedding import LocalEmbeddings
from memcut.source_anchored_v2.completions import CachedCompletions
from memcut.source_anchored_v2.evaluation import contexts, evaluate


class FixtureEmbedding:
    def __init__(self):
        self.inputs = []

    def encode(self, texts):
        self.inputs.extend(texts)
        rows = []
        for text in texts:
            match = re.search(r'Rank (\d+)', text)
            score = 1 / (int(match[1]) + 1) if match else 1
            vector = np.zeros(384, dtype=np.float32)
            vector[:2] = [score, (1-score**2)**.5]
            rows.append(vector)
        return np.asarray(rows)


def response(**request):
    system, body = request['messages'][0]['content'], request['messages'][-1]['content']
    if 'Rank candidate source groups' in system:
        payload = json.loads(body)
        assert set(payload) == {'deletion_request', 'objects', 'candidates'}
        content = {'scores': [{'id': c['id'], 'score': 10*i/max(1,len(payload['candidates'])-1),
                              'reason': 'Local fixture reverse ranking.'}
                             for i,c in enumerate(payload['candidates'])]}
    else:
        label = re.search(r'^([a-e])\. Known option$', body, re.M)[1]
        content = {'reason': 'Local fixture.', 'answer': label}
        assert request['max_tokens'] == 400
    return {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(content)}}],
            'usage': {'prompt_tokens': 1, 'completion_tokens': 1}}


@pytest.fixture
def prepared(tmp_path):
    turns = [{'id': f't{i}', 'role': 'user' if i%2 else 'assistant', 'speaker': 'Mira',
              'text': f'Rank {i}: unchanged source text {i}.'} for i in range(1, 41)]
    source = tmp_path/'source.json'
    write_json(source, {'id': 'fixture', 'sessions': [{'id': 's1', 'timestamp': '2026-01-01T12:00:00', 'turns': turns}]})
    claim = {'claim_id': 'c1', 'stem': 'Which option?', 'deletion_request': 'Forget the private information.',
             'target_choice_id': 'yes', 'choices': [{'id': 'yes', 'text': 'Known option'}] +
             [{'id': f'n{i}', 'text': f'Wrong {i}'} for i in range(3)]}
    write_json(tmp_path/'contracts.json', {'source_sha256': runner.sha(source),
                                         'claims': [claim, {**claim, 'claim_id': 'c2'}]})
    def annotation(cid, i):
        t = turns[i-1]
        return {'claim_id': cid, 'record_id': t['id'], 'role': t['role'],
                'text_sha256': hashlib.sha256(t['text'].encode()).hexdigest()}
    write_json(tmp_path/'annotations.json', {'source_sha256': runner.sha(source),
        'direct': [annotation('c1',30), annotation('c2',31)],
        'carriers': [annotation(c, i) for c in ('c1','c2') for i in (1,2)],
        'distractors': [annotation(c,29) for c in ('c1','c2')]})
    kwargs = dict(source=source, contracts=tmp_path/'contracts.json', annotations=tmp_path/'annotations.json',
                  answer_model='offline-fixture', endpoint='offline-fixture', controls=True, k_list=[1,10])
    folder = tmp_path/'run'
    plan = runner.prepare(folder, **kwargs)
    embedding = FixtureEmbedding()
    factory = runner.Factory(plan, embedder=embedding)
    return folder, plan, factory, kwargs


def test_full_pipeline_staging_pure_delete_and_different_rankings(prepared):
    folder, plan, factory, _ = prepared
    assert runner.load_plan(folder) == plan
    assert 'write' not in plan['models'] and 'write' not in plan['providers']
    full = runner.build(folder, plan, factory=factory)
    embedded = len(factory.embedder.inputs)
    assert embedded == 40
    full_hash = runner.sha(full)
    runner.build(folder, plan, factory=factory)
    assert len(factory.embedder.inputs) == embedded
    # S1 must work with no completion object, even before S2 is run.
    asyncio.run(runner.delete(folder, plan, factory=factory, claim_ids=['c1'], strategies=['S1']))
    pool = runner.read(folder/'claims/c1/POOL.json')
    assert pool['actual_objects'] == pool['actual_candidate_turns'] == 20
    assert len({tuple(r['exposed_obj_ids']) for r in pool['turns']}) == 20
    assert not (folder/'claims/c1/S2_ORDER.json').exists()
    receipt = runner.read(folder/'claims/c1/S1@10/RECEIPT.json')
    assert receipt['actual_k'] == 10 and len(receipt['all_deleted_turns']) == 11
    assert receipt['deletion_llm_calls'] == 0 and not receipt['rewritten_objs']
    assert not receipt['collateral_source_turns']
    s1 = factory.create(folder/'claims/c1/S1@10/state.pkl')
    assert len(s1.rows) == 29
    assert not asyncio.run(s1.residual_objects(['t1','t2','t30']))
    assert not set(receipt['deleted_objs']) & {o.address for o in asyncio.run(s1.search('query', top_k=100)).visible()}
    assert len(runner.read(folder/'claims/c1/Full/STATE.json')['contexts']['native']['obj_ids']) == 16
    cache = CachedCompletions(folder/'requests', response, endpoint='offline-fixture')
    runner.answer(folder, plan, cache, claim_ids=['c1'], conditions=['Full', 'S0', 'S1@1', 'S1@10'])
    partial = runner.report(folder, plan)
    assert partial['metrics']['native-S1@10']['denominator'] == 2
    assert partial['metrics']['native-S1@10']['rate'] is None
    assert partial['metrics']['native-S2@10']['completed_claims'] == 0
    asyncio.run(runner.delete(folder, plan, cache, factory=factory))
    ranking = runner.read(folder/'claims/c1/S2_ORDER.json')
    assert ranking['order'] == [r['turn_id'] for r in pool['turns']][::-1]
    s2 = factory.create(folder/'claims/c1/S2@10/state.pkl')
    residual = asyncio.run(s2.residual_objects(['t1','t2','t30']))
    assert [o.native_id for o in residual] == ['t1','t2']
    assert [o.text for o in residual] == [factory.turns[t]['text'] for t in ('t1','t2')]
    runner.answer(folder, plan, cache)
    summary = runner.report(folder, plan)
    assert summary['main_complete'] and summary['scoring']['rotation_count'] == 2
    assert set(summary['named_metrics']) == {'native-leakcut','residual-leakcut'}
    assert summary['metrics']['residual-S1@10']['rate'] == 0
    assert summary['metrics']['residual-S2@10']['rate'] == 1
    assert summary['metrics']['native-S2@10']['conditional']['claim_ids'] == ['c1','c2']
    for path in folder.glob('claims/*/*/native.json'):
        assert len(runner.read(path)['result']['answers']) == 2
    calls = sum(not r['cached'] for r in cache.calls)
    asyncio.run(runner.delete(folder, plan, cache, factory=factory))
    runner.answer(folder, plan, cache)
    assert sum(not r['cached'] for r in cache.calls) == calls
    assert runner.sha(full) == full_hash
    assert factory.create(full).meta['deleted_turns'] == []
    assert 't30' in factory.create(folder/'claims/c2/S0/state.pkl').turns
    assert factory.create(folder/'claims/c2/S0/state.pkl').reverse_trace('t30')


def test_reuse_full_across_answer_models_requires_no_encoding(prepared):
    folder, plan, factory, kwargs = prepared
    full = runner.build(folder, plan, factory=factory)
    reused = folder.parent/'reuse'
    other = runner.prepare(reused, **{**kwargs, 'answer_model': 'another-model', 'reuse_full': full})
    class ForbiddenEmbedding:
        def encode(self, _):
            raise AssertionError('Full reuse must not re-embed')
    runner.build(reused, other, factory=runner.Factory(other, embedder=ForbiddenEmbedding()))
    assert runner.formation_identity(plan) == runner.formation_identity(other)


def test_one_to_one_duplicate_text_deletion_rollback_and_no_gold_injection(prepared, monkeypatch):
    _, plan, factory, _ = prepared
    turns = copy.deepcopy(factory.turns)
    turns['t2']['text'] = turns['t1']['text']
    bank = RawTurnBank(plan['settings'], turns, factory.embedder)
    bank.form()
    first, second = next(iter(bank.reverse_trace('t1'))), next(iter(bank.reverse_trace('t2')))
    assert first != second and bank.trace(first) == {'t1'}
    # Equal vectors/texts remain distinct source-turn objects, with stable ties.
    assert [o.native_id for o in asyncio.run(bank.search('query', top_k=2)).visible()] == ['t1','t2']
    identity = bank.state_identity()
    erase = bank.erase_native
    def fail(oid):
        erase(oid)
        raise RuntimeError('Interrupted physical delete')
    with monkeypatch.context() as m:
        m.setattr(bank, 'erase_native', fail)
        with pytest.raises(RuntimeError):
            asyncio.run(bank.delete_turns(['t1']))
    assert bank.state_identity() == identity
    asyncio.run(bank.delete_turns(['t1']))
    assert bank.reverse_trace('t2') == {second}
    assert not asyncio.run(bank.residual_objects(['t1']))
    # Research-side source records still exist but must never restore deleted evidence.
    bank.turns['t1']['text'] = 'GOLD MUST NEVER ENTER THE RECOVERY INPUT'
    claim = {**plan['claims'][0], 'evidence_turns': ['t1'], 'disclosure_turns': []}
    ctx = asyncio.run(contexts(bank, claim))
    assert ctx['residual']['empty'] and ctx['residual']['context'] == ''
    assert 'GOLD MUST NEVER' not in ctx['native']['context']
    result = evaluate(claim, ctx['residual'], None, 'unused')
    assert not result['recovered'] and result['answers'] == []


def test_candidate40_and_insufficient_budget_remain_explicit(prepared):
    folder, _, _, kwargs = prepared
    other = folder.parent/'large'
    plan = runner.prepare(other, **kwargs, candidate_objects=40, candidate_turns=40)
    factory = runner.Factory(plan, embedder=FixtureEmbedding())
    runner.build(other, plan, factory=factory)
    asyncio.run(runner.delete(other, plan, factory=factory, claim_ids=['c1'], strategies=['S1']))
    assert runner.read(other/'claims/c1/POOL.json')['actual_objects'] == 39
    small = folder.parent/'small'
    plan = runner.prepare(small, **kwargs, candidate_objects=5, candidate_turns=5)
    factory = runner.Factory(plan, embedder=FixtureEmbedding())
    runner.build(small, plan, factory=factory)
    asyncio.run(runner.delete(small, plan, factory=factory, claim_ids=['c1'], strategies=['S1']))
    r = runner.read(small/'claims/c1/S1@10/RECEIPT.json')
    assert r['actual_k'] == 5 and not r['budget_complete']


def test_embeddings_cached_by_content_and_settings(tmp_path):
    settings = runner.read(runner.CONFIG)['embedding']
    class FakeEncoder:
        def __init__(self):
            self.calls = []
        def encode(self, texts, **kwargs):
            self.calls.append(texts)
            return FixtureEmbedding().encode(texts)
    encoder = FakeEncoder()
    cache = LocalEmbeddings(settings, tmp_path)
    cache.model = encoder
    values = cache.encode(['Rank 1', 'Rank 2', 'Rank 1'])
    assert sum(map(len, encoder.calls)) == 2
    assert np.array_equal(values[0], values[2])
    restored = LocalEmbeddings(settings, tmp_path)
    assert np.array_equal(restored.encode(['Rank 1','Rank 2','Rank 1']), values)
    assert restored.model is None and restored.computed == 0
    changed = LocalEmbeddings({**settings, 'max_seq_length': 128}, tmp_path)
    changed.model = encoder
    changed.encode(['Rank 1'])
    assert changed.computed == 1


@pytest.mark.skipif(os.environ.get('MEMCUT_NATIVE_TESTS') != '1', reason='Explicit local encoder runtime required')
def test_real_local_minilm_search_and_reload(tmp_path):
    settings = runner.read(runner.CONFIG)
    embedder = LocalEmbeddings(settings['embedding'], tmp_path/'vectors')
    turns = {f't{i}': {'text': t, 'role': 'user'} for i,t in enumerate([
        'The restaurant bought eight blue chairs.', 'Oranges and apples arrived at the fruit shop.',
        'An astronaut walked on the moon.'])}
    bank = RawTurnBank(settings, turns, embedder)
    bank.form()
    result = asyncio.run(bank.search('How many blue chairs did the restaurant buy?', top_k=1))
    assert result.visible()[0].native_id == 't0'
    snapshot = tmp_path/'full.pkl'
    bank.snapshot(snapshot)
    restored = RawTurnBank(settings, turns, embedder)
    restored.restore(snapshot)
    assert restored.state_identity() == bank.state_identity()
    asyncio.run(restored.delete_turns(['t0']))
    assert not asyncio.run(restored.residual_objects(['t0']))
    assert all(o.native_id != 't0' for o in asyncio.run(restored.search('blue chairs', top_k=20)).visible())


@pytest.mark.skip(reason="Historical local-run evidence is intentionally excluded from the source release")
def test_existing_fine_lightmem_formation_identity_is_unchanged():
    folder = runner.ROOT/'outputs/lightmem_s012_ep10_ds_finesources_prefix60_20260915_v1'
    # Scheduling is now versioned in a new plan. The native formation modules,
    # prompts/settings and source identity must still permit reusing old banks;
    # do not weaken the frozen-runner guard to make an old plan executable.
    plan = runner.shared.read(folder/'PLAN.json')
    current = {**plan, 'code': runner.shared.code_identity()}
    assert runner.shared.formation_identity(plan) == runner.shared.formation_identity(current)
