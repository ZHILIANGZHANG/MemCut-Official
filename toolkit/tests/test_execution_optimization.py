"""No API: exact identity, concurrency, failure, accounting and 2/2 regression."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import tempfile
import pickle
import threading
import time
import unittest

from memcut.execution_optimization import receipts, scheduling
from memcut.model_adapters.glm import GLMCompletions, MODEL
from memcut.native_memory.contract import digest
from memcut.source_anchored_v2 import runner
from memcut.source_anchored_v2.completions import CachedCompletions


def request():
    return {'model': MODEL, 'temperature': 0, 'max_tokens': 400,
            'messages': [{'role': 'user', 'content': 'same input'}]}


def response(text='{"answer":"a","reason":"fixture"}', finish='stop'):
    return {'choices': [{'finish_reason': finish, 'message': {'content': text}}],
            'usage': {'prompt_tokens': 100, 'completion_tokens': 10,
                      'prompt_tokens_details': {'cached_tokens': 20}}}


def forbidden(**kwargs):
    raise AssertionError('Unexpected model dispatch')


class Gateway:
    def __init__(self, cache):
        self.cache = cache

    def for_role(self, role):
        return self.cache

    @property
    def calls(self):
        return self.cache.calls


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.old = GLMCompletions(self.root/'old/requests', lambda **k: response())
        self.saved = self.old.complete(request())
        self.catalog = receipts.catalog([self.root/'old'])

    def reuse(self, catalog=None):
        return receipts.ReuseCache(GLMCompletions(self.root/'new/requests', forbidden), catalog or self.catalog)

    def test_normalized_import_is_one_receipt_under_concurrency(self):
        cache = self.reuse()
        with ThreadPoolExecutor(max_workers=4) as pool:
            answers = list(pool.map(lambda _: cache.complete(request()), range(8)))
        self.assertTrue(all(a['cached'] and a['request_id']==self.saved['request_id'] for a in answers))
        report = receipts.usage(self.root/'new')
        self.assertEqual(report['imported']['requests'], 1)
        self.assertEqual(report['new']['requests'], 0)
        self.assertEqual(report['imported']['completion_tokens'], 10)
        self.assertEqual(len(list((self.root/'new/requests').glob('*/response.json'))), 1)

    def test_different_model_parameters_do_not_import(self):
        for field, value in [('max_tokens', 2048), ('temperature', .2), ('messages', [{'role':'user','content':'changed'}])]:
            with self.subTest(field=field):
                changed = {**request(), field: value}
                with self.assertRaises(AssertionError):
                    self.reuse().complete(changed)
        self.assertFalse(list((self.root/'new/requests').glob('*/REUSED_RECEIPT.json')))

    def test_different_provider_does_not_import(self):
        cache = CachedCompletions(self.root/'new/requests', forbidden, endpoint='https://other.invalid')
        with self.assertRaises(AssertionError):
            receipts.ReuseCache(cache, self.catalog).complete(request())

    def test_deepseek_normalization_preserved(self):
        old = CachedCompletions(self.root/'ds/requests', lambda **k: response(),
                                endpoint='https://ds.invalid', compatibility='deepseek')
        req = {**request(), 'model': 'deepseek-fixture'}
        result = old.complete(req)
        new = CachedCompletions(self.root/'new/requests', forbidden, endpoint='https://ds.invalid', compatibility='deepseek')
        value = receipts.ReuseCache(new, receipts.catalog([self.root/'ds'])).complete(req)
        self.assertEqual(result['request_id'], value['request_id'])

    def test_unresolved_local_dispatch_is_not_hidden(self):
        cache = self.reuse()
        identity = receipts.request_identity(cache.delegate, request())
        target = self.root/'new/requests'/digest(identity)
        runner.write_json(target/'request.json', identity)
        with self.assertRaisesRegex(RuntimeError, 'Unresolved'):
            cache.complete(request())
        self.assertFalse((target/'response.json').exists())

    def test_changed_source_fails_before_import(self):
        path = self.root/'old/requests'/self.saved['request_id']/'response.json'
        runner.write_json(path, response('different'))
        with self.assertRaisesRegex(ValueError, 'changed'):
            self.reuse().complete(request())

    def test_conflicting_samples_rejected(self):
        other = GLMCompletions(self.root/'other/requests', lambda **k: response('another sample'))
        other.complete(request())
        with self.assertRaisesRegex(ValueError, 'Different received samples'):
            receipts.catalog([self.root/'old', self.root/'other'])

    def test_incomplete_answer_receipt_remains_invalid(self):
        old = GLMCompletions(self.root/'truncated/requests', lambda **k: response('nonempty', 'length'))
        old.complete(request(), allow_length=True)
        cache = self.reuse(receipts.catalog([self.root/'truncated']))
        with self.assertRaisesRegex(RuntimeError, 'Incomplete'):
            cache.complete(request())
        self.assertEqual(receipts.usage(self.root/'new')['new']['requests'], 0)

    def test_missing_prefetch_never_dispatches(self):
        cache = self.reuse()
        prefetch = scheduling.PrefetchedFormation(cache, [request()])
        with self.assertRaisesRegex(RuntimeError, 'missing'):
            prefetch.complete(request(), allow_length=True)
        self.assertFalse(list((self.root/'new').glob('requests/*/request.json')))


def fixture_plan():
    claim = runner.normalize_claim({'claim_id': 'c1', 'stem': 'Which choice?',
        'target_choice_id': 'right', 'choices': [{'id': 'right', 'text': 'Correct'}] +
        [{'id': f'w{i}', 'text': f'Wrong{i}'} for i in range(3)]})
    return {'claims': [claim], 'conditions': ['Full', 'S0', 'S1@10', 'S2@10'],
            'models': {'answer': MODEL}, 'controls': False}


def states(folder, plan):
    for condition in plan['conditions']:
        target = folder/'claims/c1'/condition
        target.mkdir(parents=True)
        (target/'state.pkl').write_bytes(condition.encode())
        contexts = {}
        for family in ('native', 'residual'):
            # Exact duplicate S1/S2 native contexts exercise simultaneous reuse.
            text = 'same' if condition.startswith('S') and family=='native' else condition+family
            empty = condition=='S2@10' and family=='residual'
            if empty:
                text = ''
            contexts[family] = {'context': text, 'context_sha256': digest(text),
                                'state_identity': condition, 'empty': empty}
        runner.write_json(target/'STATE.json', {'plan_sha256': digest(plan),
            'snapshot_sha256': runner.sha(target/'state.pkl'), 'state_identity': condition, 'contexts': contexts})


class AnswerTests(unittest.TestCase):
    def test_parallel_matches_serial_and_preserves_both_permutations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = fixture_plan()
            slow, fast = root/'serial', root/'parallel'
            for folder in (slow, fast):
                states(folder, plan)
            active, peak, count, lock = 0, 0, 0, threading.Lock()
            def transport(**kwargs):
                nonlocal active, peak, count
                with lock:
                    active += 1
                    count += 1
                    peak = max(peak, active)
                time.sleep(.02)
                with lock:
                    active -= 1
                return response()
            serial = GLMCompletions(slow/'requests', lambda **k: response())
            parallel = GLMCompletions(fast/'requests', transport)
            runner.answer(slow, plan, Gateway(serial))
            scheduling.answer(fast, plan, Gateway(parallel), workers=4)
            self.assertGreater(peak, 1)
            self.assertLessEqual(peak, 4)
            old_ids = {p.parent.name for p in (slow/'requests').glob('*/response.json')}
            self.assertEqual(count, len(old_ids))
            for f in (slow/'claims').glob('*/*/*.json'):
                self.assertEqual(runner.read(f), runner.read(fast/f.relative_to(slow)))
            for f in (fast/'claims').glob('*/*/native.json'):
                self.assertEqual(len(runner.read(f)['result']['answers']), 2)
            self.assertEqual(runner.read(fast/'claims/c1/S2@10/residual.json')['result']['answers'], [])

    def test_invalid_answer_stops_without_zero_score(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            plan = fixture_plan()
            states(folder, plan)
            cache = GLMCompletions(folder/'requests', lambda **k: response('{"answer":"unknown"}'))
            with self.assertRaises(ValueError):
                scheduling.answer(folder, plan, Gateway(cache), workers=1)
            self.assertEqual(len(cache.calls), 1)
            self.assertFalse(list((folder/'claims').glob('*/*/native.json')))


@unittest.skipUnless(os.environ.get('MEMCUT_MEMORYBANK_NATIVE_TESTS') == '1', 'MemoryBank native environment required')
class MemoryBankFormationTests(unittest.TestCase):
    def test_uncached_parallel_formation_preserves_native_state_and_call_count(self):
        from test_memorybank_s012 import episode, FakeModel
        from memcut.source_anchored_v2.memorybank import open_memorybank
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = episode()
            runner.write_json(root/'source.json', source)
            seed = open_memorybank(Path(__file__).resolve().parents[2]/'memorybank',
                                   'Mira', root/'unused.pkl', 'fixture')
            plan = {'inputs': {'source': {'path': str(root/'source.json')}}, 'reuse_full': None}
            slow_model, fast_model = FakeModel(), FakeModel()
            active, peak, lock = 0, 0, threading.Lock()
            def transport(**kwargs):
                nonlocal active, peak
                with lock:
                    active += 1
                    peak = max(peak, active)
                time.sleep(.03)
                try:
                    return fast_model(**kwargs)
                finally:
                    with lock:
                        active -= 1
            serial = CachedCompletions(root/'serial/requests', slow_model, endpoint='https://fixture.invalid')
            fast = CachedCompletions(root/'fast/requests', transport, endpoint='https://fixture.invalid')
            a = asyncio.run(runner.build(root/'serial', plan, Gateway(serial), seed=seed))
            b = asyncio.run(scheduling.build_memorybank(root/'fast', plan, Gateway(fast), seed=seed, workers=4))
            expected, actual = pickle.loads(Path(a).read_bytes()), pickle.loads(Path(b).read_bytes())
            for field in ('memory', 'pair_ids', 'source_anchored_v2', 'retrieval_config'):
                self.assertEqual(expected[field], actual[field], field)
            self.assertGreater(peak, 1)
            self.assertLessEqual(peak, 4)
            self.assertEqual(len(fast_model.requests), 2*len(source['sessions'])+2)
            self.assertCountEqual(fast_model.requests, slow_model.requests)
            self.assertEqual(fast_model.requests[-2:], slow_model.requests[-2:])


if __name__ == '__main__':
    unittest.main()
