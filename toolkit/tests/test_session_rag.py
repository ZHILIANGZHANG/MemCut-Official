"""Offline contracts for whole-session retrieval and exact raw-turn deletion.

No live completion or embedding API is used here. Deterministic unit vectors
make the expected cosine ranks independent of platform BLAS behavior.
"""
import asyncio
import copy
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pytest

from memcut.native_memory.persistence import write_json
from memcut.raw_turn_rag.bank import RawTurnBank
from memcut.session_rag.bank import SessionBank
from memcut.source_anchored_v2.candidates import selector_input, verify_pool
from memcut.source_anchored_v2.completions import CachedCompletions
from memcut.source_anchored_v2.evaluation import contexts as raw_contexts


class WordTokenizer:
    """HF-shaped whitespace tokenizer with exact character offsets."""
    def __init__(self):
        self.vocab = {}
        self.inverse = {}

    def _parts(self, text):
        return list(re.finditer(r'\S+', text))

    def encode(self, text, add_special_tokens=True, **kwargs):
        assert not kwargs.get('truncation'), 'The baseline must not truncate input.'
        ids = []
        for match in self._parts(text):
            token = match.group()
            if token not in self.vocab:
                value = len(self.vocab) + 3
                self.vocab[token] = value
                self.inverse[value] = token
            ids.append(self.vocab[token])
        return [1, *ids, 2] if add_special_tokens else ids

    def decode(self, ids, **kwargs):
        return ' '.join(self.inverse[i] for i in ids if i > 2)

    def num_special_tokens_to_add(self, pair=False):
        assert not pair
        return 2

    def __call__(self, text, add_special_tokens=True, return_offsets_mapping=False, **kwargs):
        result = {'input_ids': self.encode(text, add_special_tokens=add_special_tokens, **kwargs)}
        if return_offsets_mapping:
            offsets = [m.span() for m in self._parts(text)]
            result['offset_mapping'] = [(0, 0), *offsets, (0, 0)] if add_special_tokens else offsets
        return result


class FixtureEmbedding:
    def __init__(self, dimensions=2, tokenizer=None, max_tokens=None):
        self.dimensions = dimensions
        self.inputs = []
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.fail = False

    def encode(self, texts):
        texts = list(texts)
        if self.fail:
            raise RuntimeError('Injected local embedding failure')
        if self.max_tokens is not None:
            assert all(len(self.tokenizer.encode(t)) <= self.max_tokens for t in texts)
        self.inputs.extend(texts)
        rows = []
        for text in texts:
            explicit = re.findall(r'Score:(0\.\d+)', text)
            rank = re.search(r'Rank (\d+)', text)
            score = max(map(float, explicit)) if explicit else (1 / (int(rank[1]) + 1) if rank else 1.)
            vector = np.zeros(self.dimensions, dtype=np.float32)
            vector[:2] = [score, np.sqrt(1 - score ** 2)]
            rows.append(vector)
        return np.asarray(rows, dtype=np.float32).reshape(-1, self.dimensions)


def settings(*, native_eval_k=2, max_seq_length=8):
    return {
        'system': 'session_rag', 'native_eval_k': native_eval_k,
        'embedding': {'model': 'offline-fixture', 'revision': 'fixture-v1',
                      'dimensions': 2, 'max_seq_length': max_seq_length,
                      'normalize_embeddings': True, 'dtype': 'float32', 'device': 'cpu', 'batch_size': 64},
        'retrieval': 'exact cosine; maximum chunk score per session; stable original session order',
    }


def turns_fixture():
    return {
        'a1': {'session_id': 'A', 'text': 'Score:0.90 alpha evidence first', 'role': 'user'},
        'a2': {'session_id': 'A', 'text': 'Score:0.10 alpha background second', 'role': 'assistant'},
        'b1': {'session_id': 'B', 'text': 'Score:0.80 beta first item', 'role': 'user'},
        'b2': {'session_id': 'B', 'text': 'Score:0.80 beta second item', 'role': 'assistant'},
        'b3': {'session_id': 'B', 'text': 'Score:0.80 beta third item', 'role': 'user'},
        'c1': {'session_id': 'C', 'text': 'Score:0.90 gamma tied score', 'role': 'user'},
    }


def new_bank(turns=None, **kwargs):
    config = settings(**kwargs)
    tokenizer = WordTokenizer()
    embedding = FixtureEmbedding(tokenizer=tokenizer, max_tokens=config['embedding']['max_seq_length'])
    bank = SessionBank(config, turns or turns_fixture(), embedding, tokenizer=tokenizer)
    bank.form()
    return bank, embedding, tokenizer


def run(value):
    return asyncio.run(value)


def test_session_max_ranking_deduplicates_chunks_with_stable_ties():
    bank, embedder, tokenizer = new_bank()
    result = run(bank.search('query', top_k=2))
    # B has three .8 chunks; A has .9 and .1. Sum/mean aggregation gives a
    # different winner, so this verifies the frozen maximum aggregation.
    assert [o.native_id for o in result.visible()] == ['A', 'C']
    assert len({o.address for o in result.visible()}) == 2
    assert bank.system_render(result) == '\n'.join(turns_fixture()[t]['text'] for t in ('a1', 'a2', 'c1'))
    assert len(bank.rows) == 3
    assert len(bank.reverse_trace('a1')) == 1
    assert bank.reverse_trace('a1') == bank.reverse_trace('a2')
    for oid in bank.reverse_trace('a1'):
        assert bank.trace(oid) == {'a1', 'a2'}
    assert all(len(tokenizer.encode(t)) <= 8 for t in embedder.inputs)
    assert result.costs['llm_calls'] == 0


@pytest.mark.parametrize('top_k', [0, -1, True, 1.5])
def test_invalid_retrieval_budget_is_rejected(top_k):
    bank, _, _ = new_bank()
    with pytest.raises(ValueError):
        run(bank.search('query', top_k=top_k))


def test_long_turn_is_fully_encoded_but_preserved_as_one_source_turn():
    text = '  ' + '\n  '.join(f'café{i}，片段{i}' for i in range(25)) + '  '
    turns = {'long': {'session_id': 'long-session', 'role': 'user', 'text': text}}
    bank, embedder, tokenizer = new_bank(turns)
    # Every word is embedded, once, in sequence; no tail is silently truncated.
    assert ''.join(embedder.inputs) == text
    assert all(len(tokenizer.encode(t)) <= 8 for t in embedder.inputs)
    assert len(bank.rows) == len(bank.reverse_trace('long')) == 1
    assert bank.system_render(run(bank.search('query', top_k=1))) == text
    # Deletion-side single-turn scoring deliberately matches RawTurn: pass the
    # whole turn to the same encoder, whose length limit may truncate it.
    # This compatibility exception must be explicit in the candidate audit.
    embedder.max_tokens = None
    offset = len(embedder.inputs)
    pool = run(bank.build_pool('query', [], n_obj=20, n_turn=20))
    assert [r['turn_id'] for r in pool['turns']] == ['long']
    assert text in embedder.inputs[offset:]
    assert pool['candidate_embedding_truncated_turns_before_cap'] == 1


def test_multiline_delete_changes_only_affected_session_and_rechunks():
    turns = turns_fixture()
    turns['a1']['text'] = 'Score:0.90 alpha\nSECRET SECOND LINE'
    bank, embedder, _ = new_bank(turns)
    before = copy.deepcopy(bank.dump_native())
    a_oid = next(iter(bank.reverse_trace('a1')))
    b_oid = next(iter(bank.reverse_trace('b1')))
    b_original = copy.deepcopy(bank.rows[b_oid])
    call_offset = len(embedder.inputs)
    receipt = run(bank.delete_turns(['a1']))
    assert not bank.reverse_trace('a1')
    assert bank.trace(a_oid) == {'a2'}
    assert bank.rows[b_oid] == b_original
    assert bank.rows[a_oid]['text'] == turns['a2']['text']
    assert all('SECRET' not in t for t in embedder.inputs[call_offset:])
    assert 'SECRET' not in json.dumps(bank.dump_native())
    assert receipt['deletion_llm_calls'] == 0
    assert receipt['selected_turns_have_empty_der']
    # Rebuilding from the surviving turns gives exactly the same live chunks.
    survivor = {t: row for t, row in turns.items() if t != 'a1'}
    fresh, _, _ = new_bank(survivor)
    assert bank.rows[a_oid] == fresh.rows[a_oid]
    assert bank.dump_native() != before
    run(bank.delete_turns(['a2']))
    assert a_oid not in bank.rows
    assert not bank.reverse_trace('a2')
    assert all(o.native_id != 'A' for o in run(bank.search('query', top_k=10)).visible())


def test_delete_is_transactional_on_embedding_failure_and_idempotent():
    bank, embedder, _ = new_bank()
    before = copy.deepcopy(bank.export())
    identity = bank.state_identity()
    embedder.fail = True
    with pytest.raises(RuntimeError, match='Injected local embedding failure'):
        run(bank.delete_turns(['a1']))
    assert bank.export() == before
    assert bank.state_identity() == identity
    embedder.fail = False
    run(bank.delete_turns(['a1']))
    after = bank.state_identity()
    count = len(embedder.inputs)
    run(bank.delete_turns(['a1']))
    assert bank.state_identity() == after
    assert len(embedder.inputs) == count
    with pytest.raises(ValueError):
        run(bank.delete_turns(['not-an-input-turn']))
    assert bank.state_identity() == after



def test_failed_formation_leaves_no_partial_bank_or_provenance():
    config = settings()
    tokenizer = WordTokenizer()
    embedder = FixtureEmbedding(tokenizer=tokenizer, max_tokens=8)
    bank = SessionBank(config, turns_fixture(), embedder, tokenizer=tokenizer)
    before = copy.deepcopy(bank.export())
    embedder.fail = True
    with pytest.raises(RuntimeError, match='Injected local embedding failure'):
        bank.form()
    assert bank.export() == before
    assert not bank.rows and not bank.meta['src'] and not bank.frozen
    embedder.fail = False
    bank.form()
    assert bank.frozen and len(bank.rows) == 3
    with pytest.raises(ValueError):
        bank.form()


def test_restore_rejects_foreign_source_without_changing_live_branch():
    bank, embedder, tokenizer = new_bank()
    different = turns_fixture()
    different['a1']['text'] = 'A different immutable source body'
    foreign = SessionBank(bank.settings, different, embedder, tokenizer=tokenizer)
    foreign.form()
    identity = bank.state_identity()
    with pytest.raises(ValueError, match='Snapshot source differs'):
        bank.restore_value(foreign.export())
    assert bank.state_identity() == identity

def test_snapshot_provenance_and_corruption_rejection(tmp_path):
    bank, embedder, tokenizer = new_bank()
    run(bank.delete_turns(['a1']))
    path = tmp_path / 'branch.pkl'
    bank.snapshot(path)
    restored = SessionBank(bank.settings, turns_fixture(), embedder, tokenizer=tokenizer)
    restored.restore(path)
    assert restored.state_identity() == bank.state_identity()
    assert restored.meta['deleted_turns'] == ['a1']
    assert not restored.reverse_trace('a1')
    assert restored.trace(next(iter(restored.reverse_trace('a2')))) == {'a2'}
    restored.lineage.validate(observed=restored.observation())
    assert restored.system_render(run(restored.search('query', top_k=20))) == bank.system_render(run(bank.search('query', top_k=20)))
    with pytest.raises(FileExistsError):
        bank.snapshot(path)
    path.write_bytes(path.read_bytes() + b'corrupt')
    with pytest.raises(ValueError, match='Changed snapshot'):
        restored.restore(path)


def test_full_and_s0_residual_match_raw_turn_bytes_and_do_not_use_source_metadata():
    bank, _, tokenizer = new_bank()
    raw = RawTurnBank(bank.settings, bank.turns, FixtureEmbedding())
    raw.form()
    claim = {'stem': 'query', 'evidence_turns': ['b2', 'a1'], 'disclosure_turns': ['a2']}
    for deleted in ([], ['a2']):
        if deleted:
            run(bank.delete_turns(deleted))
            run(raw.delete_turns(deleted))
        session_ctx = run(bank.contexts(claim))
        turn_ctx = run(raw_contexts(raw, claim))
        assert session_ctx['residual']['context'] == turn_ctx['residual']['context']
        assert session_ctx['residual']['context_sha256'] == turn_ctx['residual']['context_sha256']
        assert 'beta first' not in session_ctx['residual']['context']
        assert 'gamma' not in session_ctx['residual']['context']
    bank.turns['a2']['text'] = 'AUDITOR GOLD MUST NOT BE RESTORED'
    ctx = run(bank.contexts(claim))
    assert 'AUDITOR GOLD' not in ctx['residual']['context']
    assert 'AUDITOR GOLD' not in ctx['native']['context']
    run(bank.delete_turns(['a1', 'b2']))
    ctx = run(bank.contexts(claim))
    assert ctx['residual']['empty'] and ctx['residual']['context'] == ''


def test_pool_uses_turn_scores_and_precise_text_exposures_without_gold():
    bank, _, _ = new_bank()
    # Direct is removed before querying S0. A(.9) is retrieved while B(.8)
    # is not. Turn a2 must score .1 independently of its session's best .9.
    run(bank.delete_turns(['c1']))
    pool = run(bank.build_pool('query', ['c1'], n_obj=1, n_turn=20))
    verify_pool(pool)
    assert [r['turn_id'] for r in pool['turns']] == ['a1', 'a2']
    assert pool['turns'][0]['score'] == pytest.approx(.9)
    assert pool['turns'][1]['score'] == pytest.approx(.1)
    payload, groups = selector_input(pool)
    assert len(groups) == 2
    assert {o['text'] for o in payload['objects']} == {bank.turns[t]['text'] for t in ('a1', 'a2')}
    assert len({tuple(row['exposed_obj_ids']) for row in pool['turns']}) == 2
    # Research-only evidence annotations never enter search/turn scoring.
    for row in bank.turns.values():
        row['evidence_for'] = ['SECRET_GOLD_CLAIM']
        row['target_choice_id'] = 'SECRET_GOLD_ANSWER'
    again = run(bank.build_pool('query', ['c1'], n_obj=1, n_turn=20))
    assert again['turns'] == pool['turns']
    assert 'SECRET_GOLD' not in json.dumps(selector_input(again)[0])



def test_native_coverage_and_evidence_diagnostics_do_not_expand_retrieval():
    from memcut.session_rag.text import coverage_tokens
    bank, _, _ = new_bank(native_eval_k=1)
    claim = {'stem': 'query', 'evidence_turns': ['a1', 'b1'], 'disclosure_turns': ['a2']}
    context = run(bank.contexts(claim))['native']
    expected = '\n'.join(bank.turns[t]['text'] for t in ('a1', 'a2'))
    episode = '\n'.join(t['text'] for t in bank.turns.values())
    assert context['context'] == expected
    assert context['sessions_retrieved'] == ['A']
    assert context['evidence_turns_in_ctx'] == 1
    assert context['context_tokens'] == coverage_tokens(expected)
    assert context['episode_tokens'] == coverage_tokens(episode)
    assert context['episode_coverage'] == pytest.approx(coverage_tokens(expected) / coverage_tokens(episode))
    # Adding a gold label to another session cannot cause it to be retrieved.
    changed = run(bank.contexts({**claim, 'evidence_turns': ['b1', 'b2', 'b3']}))['native']
    assert changed['context'] == expected
    assert changed['evidence_turns_in_ctx'] == 0
    assert changed['context_sha256'] == context['context_sha256']


def test_raw_compatible_long_query_is_passed_whole_and_flagged():
    bank, embedder, _ = new_bank()
    embedder.max_tokens = None
    query = ' '.join(f'query{i}' for i in range(25))
    offset = len(embedder.inputs)
    result = run(bank.search(query, top_k=1))
    assert embedder.inputs[offset:] == [query]
    assert result.costs['query_embedding_truncated'] is True


def test_pool_requires_s0_direct_deletion():
    bank, _, _ = new_bank()
    before = bank.state_identity()
    with pytest.raises(ValueError):
        run(bank.build_pool('query', ['a2'], n_obj=20, n_turn=20))
    assert bank.state_identity() == before

def fixture_response(**request):
    system = request['messages'][0]['content']
    body = request['messages'][-1]['content']
    if 'Rank candidate source groups' in system:
        payload = json.loads(body)
        assert set(payload) == {'deletion_request', 'objects', 'candidates'}
        content = {'scores': [
            {'id': candidate['id'], 'score': 10 * i / max(1, len(payload['candidates']) - 1),
             'reason': 'Reverse relevance for fixture.'}
            for i, candidate in enumerate(payload['candidates'])]}
    else:
        assert request['max_tokens'] == 1024
        label = re.search(r'^([a-e])\. Known option$', body, re.M)[1]
        content = {'reason': 'Fixture known option.', 'answer': label}
    return {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(content)}}],
            'usage': {'prompt_tokens': 1, 'completion_tokens': 1}}


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    from memcut.session_rag import runner
    # This fixture uses synthetic embeddings/tokenization, not installed neural packages.
    monkeypatch.setattr(runner, "runtime_identity", lambda: {"fixture-runtime": "1"})
    turns = [{'id': f't{i}', 'role': 'user' if i % 2 else 'assistant', 'speaker': 'Mira',
              'text': f'Rank {i}: unchanged source text {i}.'} for i in range(1, 41)]
    source = tmp_path / 'source.json'
    write_json(source, {'id': 'session-fixture', 'sessions': [
        {'id': f's{j}', 'timestamp': f'2026-01-0{j + 1}T12:00:00', 'turns': turns[j * 10:(j + 1) * 10]}
        for j in range(4)]})
    claim = {'claim_id': 'c1', 'category': 'factual_composition', 'stem': 'Which option?', 'deletion_request': 'Forget private information.',
             'target_choice_id': 'yes', 'choices': [{'id': 'yes', 'text': 'Known option'}] +
             [{'id': f'n{i}', 'text': f'Wrong {i}'} for i in range(3)]}
    write_json(tmp_path / 'contracts.json', {'source_sha256': runner.sha(source),
               'claims': [claim, {**claim, 'claim_id': 'c2', 'category': 'relational_linking'}]})
    def annotation(cid, i):
        turn = turns[i - 1]
        return {'claim_id': cid, 'record_id': turn['id'], 'role': turn['role'],
                'text_sha256': hashlib.sha256(turn['text'].encode()).hexdigest()}
    write_json(tmp_path / 'annotations.json', {'source_sha256': runner.sha(source),
        'direct': [annotation('c1', 30), annotation('c2', 31)],
        'carriers': [annotation(c, i) for c in ('c1', 'c2') for i in (1, 2)],
        'distractors': [annotation(c, 29) for c in ('c1', 'c2')]})
    folder = tmp_path / 'run'
    kwargs = dict(source=source, contracts=tmp_path / 'contracts.json', annotations=tmp_path / 'annotations.json',
                  answer_model='offline-fixture', endpoint='offline-fixture', k_list=[10])
    plan = runner.prepare(folder, **kwargs)
    tokenizer = WordTokenizer()
    factory = runner.Factory(plan, embedder=FixtureEmbedding(dimensions=384), tokenizer=tokenizer)
    return runner, folder, plan, factory, kwargs


def test_full_pipeline_fixed_budget_independent_claims_and_cached_resume(prepared):
    runner, folder, plan, factory, _ = prepared
    assert runner.load_plan(folder) == plan
    assert 'write' not in plan['models']
    full = runner.build(folder, plan, factory=factory)
    original_hash = runner.sha(full)
    original_calls = len(factory.embedder.inputs)
    runner.build(folder, plan, factory=factory)
    assert len(factory.embedder.inputs) == original_calls
    run(runner.delete(folder, plan, factory=factory, claim_ids=['c1'], strategies=['S1']))
    pool = runner.read(folder / 'claims/c1/POOL.json')
    assert pool['actual_candidate_turns'] == 20
    assert 't30' not in {r['turn_id'] for r in pool['turns']}
    assert len({tuple(r['exposed_obj_ids']) for r in pool['turns']}) == 20
    receipt = runner.read(folder / 'claims/c1/S1@10/RECEIPT.json')
    assert receipt['actual_k'] == 10
    assert len(receipt['all_deleted_turns']) == 11
    assert receipt['deletion_llm_calls'] == 0
    from memcut.session_rag.transport import Gateway
    cache = Gateway(folder, plan, create_factory=lambda role, request: fixture_response(**request))
    runner.answer(folder, plan, cache, claim_ids=['c1'], conditions=['Full', 'S0', 'S1@10'])
    partial = runner.report(folder, plan)
    assert partial['metrics']['native-S1@10']['denominator'] == 2
    assert partial['metrics']['native-S1@10']['completed_claims'] == 1
    assert partial['metrics']['native-S1@10']['rate'] is None
    assert partial['metrics']['native-S2@10']['completed_claims'] == 0
    run(runner.delete(folder, plan, cache, factory=factory))
    ranking = runner.read(folder / 'claims/c1/S2_ORDER.json')
    assert ranking['order'] == [r['turn_id'] for r in pool['turns']][::-1]
    runner.answer(folder, plan, cache)
    report = runner.report(folder, plan)
    assert report['main_complete']
    assert report['scoring']['rotation_count'] == 2
    assert len(report['metrics']) == 8
    assert report['metrics']['residual-S1@10']['rate'] == 0
    assert report['metrics']['residual-S2@10']['rate'] == 1
    categories = runner.read(folder / 'CATEGORY_RESULTS.json')['categories']
    assert set(categories) == {'factual_composition', 'relational_linking'}
    assert all(c['metrics']['native-Full']['denominator'] == 1 for c in categories.values())
    for path in folder.glob('claims/*/*/native.json'):
        assert len(runner.read(path)['result']['answers']) == 2
    sends = sum(not c['cached'] for c in cache.calls)
    run(runner.delete(folder, plan, cache, factory=factory))
    runner.answer(folder, plan, cache)
    assert sum(not c['cached'] for c in cache.calls) == sends
    assert runner.sha(full) == original_hash
    assert factory.create(full).meta['deleted_turns'] == []
    assert factory.create(folder / 'claims/c2/S0/state.pkl').reverse_trace('t30')
    before_audit = sum(not c['cached'] for c in cache.calls)
    audit = run(runner.audit(folder, plan, factory=factory))
    assert audit['status'] == 'PASS' and audit['provider_receipts_verified']
    assert audit['states_checked'] == 8
    assert runner.report(folder, plan)['audited_complete']
    diagnostic = run(runner.coverage(folder, plan, factory=factory, k_values=(1, 3, 16)))
    assert diagnostic['api_calls'] == 0 and diagnostic['not_model_performance']
    assert len(diagnostic['records']) == 2 * 4 * 3
    assert sum(not c['cached'] for c in cache.calls) == before_audit


def test_malformed_answer_is_not_scored_as_zero_or_silently_retried(prepared):
    runner, folder, plan, factory, _ = prepared
    runner.build(folder, plan, factory=factory)
    run(runner.delete(folder, plan, factory=factory, claim_ids=['c1'], strategies=['S0']))
    calls = []
    def malformed(**request):
        calls.append(request)
        return {'choices': [{'finish_reason': 'length', 'message': {'content': '{"answer":"a"}'}}],
                'usage': {'prompt_tokens': 1, 'completion_tokens': 1}}
    cache = CachedCompletions(folder / 'requests', malformed, endpoint='offline-fixture')
    with pytest.raises(RuntimeError, match='Incomplete/empty model response'):
        runner.answer(folder, plan, cache, claim_ids=['c1'], conditions=['Full'])
    assert not (folder / 'claims/c1/Full/native.json').exists()
    result = runner.report(folder, plan)
    assert result['metrics']['native-Full']['rate'] is None
    assert result['metrics']['native-Full']['completed_claims'] == 0
    with pytest.raises(RuntimeError, match='Incomplete/empty model response'):
        runner.answer(folder, plan, cache, claim_ids=['c1'], conditions=['Full'])
    assert len(calls) == 1


def test_malformed_selector_does_not_create_s2_and_preserves_full(prepared):
    runner, folder, plan, factory, _ = prepared
    full = runner.build(folder, plan, factory=factory)
    before = runner.sha(full)
    def invalid_selector(**request):
        return {'choices': [{'finish_reason': 'stop', 'message': {'content': '{"scores":[]}'}}],
                'usage': {'prompt_tokens': 1, 'completion_tokens': 1}}
    cache = CachedCompletions(folder / 'requests', invalid_selector, endpoint='offline-fixture')
    with pytest.raises(ValueError, match='Selector must score'):
        run(runner.delete(folder, plan, cache, factory=factory, claim_ids=['c1'], strategies=['S2']))
    assert not (folder / 'claims/c1/S2_ORDER.json').exists()
    assert not (folder / 'claims/c1/S2@10/STATE.json').exists()
    assert runner.sha(full) == before
    assert factory.create(full).meta['deleted_turns'] == []


def test_completed_full_reuse_across_answer_models_does_not_reembed(prepared):
    runner, folder, plan, factory, kwargs = prepared
    full = runner.build(folder, plan, factory=factory)
    original_hash = runner.sha(full)
    other_folder = folder.parent / 'second-model'
    other_plan = runner.prepare(other_folder, **{**kwargs, 'answer_model': 'another-fixture', 'reuse_full': full})
    class ForbiddenEmbedding:
        def encode(self, _):
            raise AssertionError('Reusable Session Full must not be re-embedded')
    reused = runner.build(other_folder, other_plan, factory=runner.Factory(
        other_plan, embedder=ForbiddenEmbedding(), tokenizer=WordTokenizer()))
    assert runner.formation_identity(plan) == runner.formation_identity(other_plan)
    first = factory.create(full)
    second = runner.Factory(other_plan, embedder=ForbiddenEmbedding(), tokenizer=WordTokenizer()).create(reused)
    assert second.state_identity() == first.state_identity()
    assert second.dump_native() == first.dump_native()
    assert second.lineage.data == first.lineage.data
    assert runner.sha(full) == original_hash


def test_always_two_trials_pipeline_audits_original_responses(prepared):
    from memcut.session_rag.transport import Gateway
    runner, original_folder, _, _, kwargs = prepared
    folder = original_folder.parent / 'full-two-trials'
    plan = runner.prepare(folder, **kwargs, early_stop=False)
    factory = runner.Factory(plan, embedder=FixtureEmbedding(dimensions=384), tokenizer=WordTokenizer())
    cache = Gateway(folder, plan, create_factory=lambda role, request: fixture_response(**request))
    runner.build(folder, plan, factory=factory)
    run(runner.delete(folder, plan, cache, factory=factory))
    runner.answer(folder, plan, cache)
    assert run(runner.audit(folder, plan, factory=factory))['status'] == 'PASS'
    assert runner.report(folder, plan)['audited_complete']
    for path in folder.glob('claims/*/*/native.json'):
        assert len(runner.read(path)['result']['answers']) == 2


def test_run_cli_stops_before_any_api_when_coverage_exceeds_threshold(prepared, monkeypatch):
    import importlib.util
    import sys
    runner, folder, _, factory, _ = prepared
    spec = importlib.util.spec_from_file_location('session_rag_cli_gate', runner.ROOT/'tools/session_rag.py')
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(runner, 'Factory', lambda *args, **kwargs: factory)
    def forbidden_gateway(*args, **kwargs):
        raise AssertionError('Coverage gate must stop before constructing a model transport')
    monkeypatch.setattr(runner, 'gateway', forbidden_gateway)
    monkeypatch.setattr(sys, 'argv', ['session_rag.py', 'run', '--run', str(folder)])
    cli.main()
    stage = runner.read(folder/'LAST_STAGE.json')['result']
    assert stage['stage'] == 'coverage_review_required' and stage['api_calls'] == 0
    assert any(g['above_25_percent'] for g in stage['coverage'])
    assert not (folder/'requests').exists()
    assert not list(folder.glob('claims/*/S2_ORDER.json'))


def test_frozen_input_change_fails_plan_loading(prepared):
    runner, folder, plan, _, _ = prepared
    source = Path(plan['inputs']['source']['path'])
    source.write_text(source.read_text() + '\n')
    with pytest.raises(ValueError, match='Frozen source or annotations changed'):
        runner.load_plan(folder)


def test_insufficient_turn_pool_records_actual_budget_without_padding(prepared):
    runner, folder, _, _, kwargs = prepared
    small = folder.parent / 'small-pool'
    plan = runner.prepare(small, **kwargs, candidate_objects=1, candidate_turns=5)
    factory = runner.Factory(plan, embedder=FixtureEmbedding(dimensions=384), tokenizer=WordTokenizer())
    runner.build(small, plan, factory=factory)
    run(runner.delete(small, plan, factory=factory, claim_ids=['c1'], strategies=['S1']))
    pool = runner.read(small / 'claims/c1/POOL.json')
    receipt = runner.read(small / 'claims/c1/S1@10/RECEIPT.json')
    assert pool['actual_objects'] == 1 and pool['actual_candidate_turns'] == 5
    assert receipt['requested_k'] == 10 and receipt['actual_k'] == 5
    assert not receipt['budget_complete']
    assert len(receipt['all_deleted_turns']) == 6
    assert len(set(receipt['extra_deleted_turns'])) == 5


def test_audit_manifest_is_invalidated_by_changed_raw_provider_receipt(prepared):
    from memcut.session_rag.transport import Gateway
    runner, folder, plan, factory, _ = prepared
    runner.build(folder, plan, factory=factory)
    cache = Gateway(folder, plan, create_factory=lambda role, request: fixture_response(**request))
    run(runner.delete(folder, plan, cache, factory=factory))
    runner.answer(folder, plan, cache)
    audit = run(runner.audit(folder, plan, factory=factory))
    assert audit['status'] == 'PASS'
    before = runner.report(folder, plan)
    assert before['audited_complete']
    response_path = next((folder / 'requests').glob('*/response.json'))
    response = runner.read(response_path)
    response['usage']['prompt_tokens'] += 1
    write_json(response_path, response)
    after = runner.report(folder, plan)
    assert after['main_complete'] and after['metrics'] == before['metrics']
    assert not after['audited_complete']
    with pytest.raises((ValueError, RuntimeError)):
        run(runner.audit(folder, plan, factory=factory))


def test_audit_rejects_symlinked_receipt_directory(prepared):
    import shutil
    from memcut.session_rag.transport import Gateway
    runner, folder, plan, factory, _ = prepared
    runner.build(folder, plan, factory=factory)
    cache = Gateway(folder, plan, create_factory=lambda role, request: fixture_response(**request))
    run(runner.delete(folder, plan, cache, factory=factory))
    runner.answer(folder, plan, cache)
    assert run(runner.audit(folder, plan, factory=factory))['status'] == 'PASS'
    receipt = next((folder/'requests').glob('*/response.json')).parent
    external = folder.parent/'external_receipt'
    shutil.move(str(receipt), str(external))
    receipt.symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'):
        runner.report(folder, plan)
    with pytest.raises(ValueError, match='symlink'):
        run(runner.audit(folder, plan, factory=factory))


@pytest.mark.parametrize('condition', ['Full', 'S0', 'S1@10'])
def test_interrupted_branch_seal_recovers_only_identical_state(prepared, condition):
    runner, folder, plan, factory, _ = prepared
    runner.build(folder, plan, factory=factory)
    run(runner.delete(folder, plan, factory=factory, claim_ids=['c1'], strategies=['S1']))
    path = folder / 'claims' / 'c1' / condition / 'state.pkl'
    blob = path.read_bytes()
    seal_path = path.with_suffix('.frozen.json')
    seal_path.unlink()
    run(runner.delete(folder, plan, factory=factory, claim_ids=['c1'], strategies=['S1']))
    assert path.read_bytes() == blob
    assert seal_path.exists()
    assert factory.create(path).state_identity() == runner.read(path.parent / 'STATE.json')['state_identity']
    # A missing seal is not permission to replace or bless different bytes.
    seal_path.unlink()
    changed = blob + b'unexplained bytes'
    path.write_bytes(changed)
    with pytest.raises((ValueError, RuntimeError)):
        run(runner.delete(folder, plan, factory=factory, claim_ids=['c1'], strategies=['S1']))
    assert path.read_bytes() == changed
    assert not seal_path.exists()


def test_interrupted_full_seal_recovers_only_deterministic_original_bytes(prepared):
    runner, folder, plan, factory, _ = prepared
    path = Path(runner.build(folder, plan, factory=factory))
    blob = path.read_bytes()
    seal_path = path.with_suffix('.frozen.json')
    seal_path.unlink()
    runner.build(folder, plan, factory=factory)
    assert path.read_bytes() == blob
    assert factory.create(path).frozen
    seal_path.unlink()
    path.write_bytes(blob + b'unexplained bytes')
    with pytest.raises((ValueError, RuntimeError)):
        runner.build(folder, plan, factory=factory)
    assert path.read_bytes() == blob + b'unexplained bytes'
    assert not seal_path.exists()


def test_orphan_snapshot_seal_is_not_overwritten_by_a_new_bank(prepared):
    runner, folder, plan, factory, _ = prepared
    path = Path(runner.build(folder, plan, factory=factory))
    seal = path.with_suffix('.frozen.json')
    old_seal = seal.read_bytes()
    path.unlink()
    with pytest.raises(ValueError):
        runner.build(folder, plan, factory=factory)
    assert not path.exists()
    assert seal.read_bytes() == old_seal


@pytest.mark.parametrize('extended', [False, True])
def test_cli_reference_inherits_cohort_budget_and_execution(prepared, monkeypatch, capsys, extended):
    import importlib.util
    import sys
    from memcut.answer_execution.early_stop import POLICY
    from memcut.raw_turn_rag import runner as raw_runner
    runner, folder, _, _, kwargs = prepared
    reference = folder.parent / 'turn-reference'
    ref = raw_runner.prepare(reference, **kwargs, claim_ids=['c2'])
    if extended:
        ref['answer_policy'] = {'max_tokens': 768}
        ref['answer_execution'] = dict(POLICY)
        write_json(reference / 'PLAN.json', ref)
    script = runner.ROOT / 'tools/session_rag.py'
    spec = importlib.util.spec_from_file_location('session_rag_cli_under_test', script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    output = folder.parent / 'session-from-reference'
    monkeypatch.setattr(sys, 'argv', [str(script), 'prepare', '--output', str(output),
                                     '--from-turn-run', str(reference)])
    cli.main()
    result = json.loads(capsys.readouterr().out)
    plan = runner.load_plan(output)
    assert result['api_calls'] == 0
    assert [c['claim_id'] for c in plan['claims']] == ['c2']
    assert plan['qa_max_tokens'] == (768 if extended else 400)
    assert plan['early_stop'] is extended
    assert plan['models'] == ref['models']
    provenance = runner.read(output / 'TURN_RAG_REFERENCE.json')
    assert provenance['reference_claim_ids'] == ['c2']
    assert provenance['selected_qa_max_tokens'] == plan['qa_max_tokens']
    # Requesting a claim outside the paired reference cannot silently widen it.
    forbidden = folder.parent / 'outside-reference'
    monkeypatch.setattr(sys, 'argv', [str(script), 'prepare', '--output', str(forbidden),
                                     '--from-turn-run', str(reference), '--claims', 'c1'])
    with pytest.raises(ValueError, match='Claims exceed'):
        cli.main()
    assert not forbidden.exists()
