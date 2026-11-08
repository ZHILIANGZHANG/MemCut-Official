"""Small offline regressions for the new entry; no remote clients or neural models."""
import asyncio
import copy
import hashlib
import json
import os
from types import SimpleNamespace

import pytest

from memcut.aligned_s012.runtime import native_profile, runtime
from memcut.source_anchored_v2 import runner as core
from memcut.source_anchored_v2.completions import CachedCompletions
from tools import memcut_s012 as entry


@pytest.fixture(autouse=True)
def private_paid_lock(tmp_path, monkeypatch):
    # Never take the real shared lock: a paid run in this checkout would block on the test suite.
    monkeypatch.setenv('MEMCUT_PAID_LOCK', str(tmp_path / 'paid.lock'))


def profile_fixture(monkeypatch, system):
    # Configuration roundtrip tests use the packaged pins, not a downloaded checkout.
    # Real upstream imports/lifecycle are covered separately by opt-in native tests.
    if system == 'memorybank':
        monkeypatch.setattr(core, 'verify_profile', lambda *_: {'ok': True})
    else:
        from memcut.source_anchored_systems import runner
        config = core.read(entry.ROOT / 'configs/source_anchored_systems_v1.json')
        monkeypatch.setattr(runner, 'load_profile', lambda name: copy.deepcopy(config[name]))


def native_environment(dependency):
    if os.environ.get('MEMCUT_ALIGNED_NATIVE_TESTS') != '1':
        pytest.skip('Opt in with MEMCUT_ALIGNED_NATIVE_TESTS=1 in a prepared native runtime')
    pytest.importorskip(dependency)


def inputs(tmp_path, system):
    source = tmp_path / 'source.json'
    turns = [dict(id=f't{i}', role='user' if i % 2 else 'assistant', speaker='Mira',
                  text=f'Rank {i}: Source item{i}.') for i in range(1, 23)]
    core.write_json(source, dict(id='fixture', sessions=[dict(id='s1',
        timestamp='2026-01-01T12:00:00', turns=turns)]))
    claim = dict(claim_id='c1', stem='Which option?', deletion_request='Forget the private inventory.',
        target_choice_id='yes', choices=[dict(id='yes', text='Known option')] +
        [dict(id=f'n{i}', text=f'Wrong {i}') for i in range(3)])
    contracts, annotations = tmp_path / 'contracts.json', tmp_path / 'annotations.json'
    core.write_json(contracts, dict(source_sha256=core.sha(source), claims=[claim]))
    def annotation(i):
        t = turns[i - 1]
        return dict(claim_id='c1', record_id=t['id'], role=t['role'],
                    text_sha256=hashlib.sha256(t['text'].encode()).hexdigest())
    core.write_json(annotations, dict(source_sha256=core.sha(source), direct=[annotation(20)],
                                      carriers=[annotation(1), annotation(2)], distractors=[annotation(21)]))
    return SimpleNamespace(system=system, output=tmp_path / 'run', source=source, contracts=contracts,
        annotations=annotations, model='offline-fixture', endpoint='https://offline.invalid/v1', claims=None)


@pytest.mark.parametrize('system', ['memorybank', 'nemori', 'lightmem'])
def test_prepare_roundtrip_without_loading_models(tmp_path, system, monkeypatch):
    profile_fixture(monkeypatch, system)
    args = inputs(tmp_path, system)
    plan = entry.prepare(args)
    with runtime(system) as base:
        assert base.load_plan(args.output) == plan
    assert not (args.output / 'requests').exists()
    assert plan['k_list'] == [10]
    assert len(plan['claims'][0]['rotations']) == 2
    if system == 'memorybank':
        assert plan['native_qa']['top_k'] == 10
    else:
        assert plan['settings']['memcut_alignment']


def test_runtime_restores_originals_and_real_lightmem_rate(monkeypatch):
    profile_fixture(monkeypatch, "lightmem")
    from memcut.source_anchored_systems import runner, lightmem
    before = runner.load_profile('lightmem')
    original = lightmem.native_settings
    with runtime('lightmem', native=True):
        assert runner.load_profile('lightmem')['settings']['compression_rate'] == .6
        cfg = lightmem.native_settings('/unused', dict(compressor='fixture', embedder='fixture'),
                                      model='fixture', base_url='offline-fixture')
        assert cfg['pre_compressor']['configs']['compress_config']['rate'] == .6
    assert lightmem.native_settings is original
    assert runner.load_profile('lightmem') == before




def test_unresolved_dispatch_blocks_other_stages(tmp_path, monkeypatch):
    profile_fixture(monkeypatch, "memorybank")
    args = inputs(tmp_path, 'memorybank')
    entry.prepare(args)
    core.write_json(args.output / 'requests/unknown/request.json', {'request': 'fixture'})
    with pytest.raises(ValueError, match='Unresolved dispatch'):
        entry.execute(args.output)
    assert not (args.output / 'full').exists()




def test_nemori_native_merge_prediction_sources_and_resume(tmp_path):
    native_environment('chromadb')
    from memcut.aligned_s012.nemori import AlignedNemoriBank
    from memcut.source_anchored_systems.profiles import profile
    from memcut.source_anchored_v2.whole_object_deletion import delete_source_turns
    from aligned_fixtures import source_turns
    class Embedding:
        model = 'fixture'
        embedding_dim = 384
        def embed_text(self, text):
            return [1.0] + [0.0] * 383
        def embed_texts(self, texts):
            return SimpleNamespace(embeddings=[self.embed_text(t) for t in texts], usage={})
        def cosine_similarity(self, a, b):
            return 1.0
    settings = {**native_profile(profile, 'nemori')['settings'], 'model': 'fixture', 'batch_threshold': 2}
    turns = source_turns(4)
    calls = []
    def create(**q):
        prompt = '\n'.join(m['content'] for m in q['messages'])
        calls.append(prompt)
        if 'response_format' not in q:
            text = 'Predicted original receipt using earlier statement.'
        else:
            if '"indices"' in prompt:
                value = {'episodes': [{'indices': [1, 2]}]}
            elif 'merge_target_id' in prompt:
                # The native merger supplies the actual stable candidate ID.
                ident = next(iter(bank.episodes.list_by_user(bank.user))).episode_id
                value = dict(decision='merge', merge_target_id=ident, reason='Same receipt')
            elif 'statements' in prompt:
                value = {'statements': ['Persistent receipt knowledge ' + str(len(calls))]}
            else:
                value = dict(title='Receipts', content='CODE01 CODE02 CODE03 CODE04',
                             timestamp='2026-01-01T12:00:01')
            text = json.dumps(value)
        return dict(choices=[dict(message=dict(content=text), finish_reason='stop')])
    cache = CachedCompletions(tmp_path / 'requests', create, endpoint='offline-fixture')
    bank = AlignedNemoriBank(settings, turns, embedder=Embedding())
    try:
        # Dedup threshold is exercised by existing tests; keep both fixture facts
        # so this test isolates ancestry through prediction and a native merge.
        bank._is_duplicate = lambda *_: False
        for tid in ('t1', 't2'):
            bank.ingest(tid, cache)
        before = copy.deepcopy(bank.export())
        for tid in ('t3', 't4'):
            bank.ingest(tid, cache)
        assert len(bank.episodes.list_by_user(bank.user)) == 1
        assert any('Predicted original receipt' in p for p in calls)
        newest = bank.semantics.list_by_user(bank.user)[-1]
        assert bank.trace(bank.oid('semantic', newest.memory_id)) == set(turns)
        n = len(calls)
        bank.restore_value(before)
        for tid in ('t3', 't4'):
            bank.ingest(tid, cache)
        assert len(calls) == n, 'Replayed merge and prediction must reuse exact receipts'
        bank.finish(cache)
        full = tmp_path / 'full.pkl'
        bank.snapshot(full)
        asyncio.run(delete_source_turns(bank, ['t1']))
        assert not bank.reverse_trace('t1')
        assert len(calls) == n, 'Whole-object deletion must not regenerate'
        bank.restore(full)
        assert bank.reverse_trace('t1')
    finally:
        bank.close()
