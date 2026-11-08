"""No native models/API: immutable audited branches, answer-only compatibility."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from memcut.execution_optimization import branches
from memcut.native_memory.contract import digest
from memcut.source_anchored_v2 import runner as core
from memcut.source_anchored_v2.candidates import rank_s2


class BranchReuseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source, self.target = self.root / 'old', self.root / 'new'
        self.claim = {'claim_id': 'c1', 'stem': 'What is the fact?', 'deletion_request': 'Remove the fact',
                      'disclosure_turns': ['t1'], 'evidence_turns': ['t3']}
        self.old = {'system': 'nemori', 'claims': [self.claim], 'conditions': ['Full', 'S0', 'S1@10', 'S2@10'],
                    'protocol': {'n_obj': 20, 'n_turn': 20, 'deletion_policy': 'whole-object-source-closure-v1'},
                    'k_list': [10], 'models': {'write': 'fixture-write', 'selector': 'fixture-selector', 'answer': 'old-answer'},
                    'providers': {r: {'endpoint': 'fixture', 'compatibility': 'openai'} for r in ('write', 'selector', 'answer')},
                    'settings': {'model': 'fixture-write', 'native_eval_k': 10}, 'code': {'frozen_core.py': 'same'},
                    'profile': {'native': 'same'}, 'inputs': {'source': {'path': '/fixture', 'sha256': 'same'}},
                    'controls': False, 'retention': [], 'residual_plus': False, 'reuse_full': None}
        self.source.mkdir()
        core.write_json(self.source / 'PLAN.json', self.old)
        (self.source / 'PROTOCOL.md').write_text('frozen fixture protocol')
        full = self.source / 'full/full.pkl'
        full.parent.mkdir()
        full.write_bytes(b'fixture Full snapshot')
        seal = {'sha256': core.sha(full), 'state_identity': 'Full-state', 'system': 'nemori', 'format': 'fixture'}
        core.write_json(full.with_suffix('.frozen.json'), seal)
        base = self.source / 'claims/c1'
        pool = {'request': self.claim['deletion_request'], 'state_identity': 'S0-state',
                'objects': {'obj1': {'text': 'native exposed content', 'score': .9, 'rank': 1, 'version': 'v1'}},
                'turns': [{'turn_id': 't2', 'score': .9, 'best_rank': 1, 'exposed_obj_ids': ['obj1']}],
                'requested_objects': 20, 'requested_candidate_turns': 20}
        pool['pool_sha256'] = digest(pool)
        core.write_json(base / 'POOL.json', pool)
        source = self.source
        class FrozenSelector:
            def complete(self, request):
                identity = {'endpoint': 'fixture', 'compatibility': 'openai', 'request': request}
                rid = digest(identity)
                text = '{"scores":[{"id":"t1","score":7,"reason":"related"}]}'
                core.write_json(source / 'requests' / rid / 'request.json', identity)
                core.write_json(source / 'requests' / rid / 'response.json', {
                    'choices': [{'finish_reason': 'stop', 'message': {'content': text}}]})
                return {'text': text, 'request_id': rid, 'finish_reason': 'stop'}
        ranking = {**rank_s2(pool, FrozenSelector(), self.old['models']['selector']),
                   'pool_sha256': pool['pool_sha256'], 'plan_sha256': digest(self.old)}
        core.write_json(base / 'S2_ORDER.json', ranking)
        for condition in self.old['conditions']:
            path = base / condition
            path.mkdir(parents=True)
            state_path = path / 'state.pkl'
            state_path.write_bytes(full.read_bytes() if condition == 'Full' else condition.encode())
            state_id = condition + '-state'
            state_seal = {**seal, 'sha256': core.sha(state_path), 'state_identity': state_id}
            core.write_json(state_path.with_suffix('.frozen.json'), state_seal)
            receipt = {}
            if condition != 'Full':
                receipt = {'deletion_policy': self.old['protocol']['deletion_policy'], 'deletion_llm_calls': 0,
                           'all_deleted_turns': ['t1'] if condition == 'S0' else ['t1', 't2'],
                           'rewritten_objs': [], 'created_objs': [], 'selected_turns_have_empty_der': True}
                if condition != 'S0':
                    receipt.update(extra_deleted_turns=['t2'], requested_k=10, actual_k=1, budget_complete=False)
                core.write_json(path / 'RECEIPT.json', receipt)
            contexts = {}
            for family in ('native', 'residual'):
                text = f'{condition} {family} exact frozen content'
                contexts[family] = {'context': text, 'context_sha256': digest(text), 'state_identity': state_id,
                                    'empty': False, 'obj_ids': ['survivor']}
                if family == 'native':
                    contexts[family]['retrieval'] = {'query': self.claim['stem']}
            core.write_json(path / 'STATE.json', {'claim_id': 'c1', 'condition': condition,
                'plan_sha256': digest(self.old), 'snapshot_sha256': core.sha(state_path), 'state_identity': state_id,
                'pool_sha256': pool['pool_sha256'] if condition.startswith(('S1', 'S2')) else None,
                'deletion': receipt, 'contexts': contexts})
            core.write_json(path / 'native.json', {'old_answer_must_not_be_copied': True})
        core.write_json(base / 'STATE_AUDIT.json', {'status': 'PASS', 'claim_id': 'c1',
            'states': [{'condition': c} for c in self.old['conditions']]})
        core.write_json(self.source / 'COMPLETE.json', {'plan_sha256': digest(self.old), 'full_unchanged': True, 'claims': ['c1']})
        core.write_json(self.source / 'RESULT_AUDIT.json', {'status': 'PASS', 'full_unchanged': True, 'claims': 1, 'state_audits': 4})
        self.new = copy.deepcopy(self.old)
        self.new['models']['answer'] = 'new-answer'
        self.new['providers']['answer']['endpoint'] = 'different-answer-only-endpoint'
        self.new['answer_policy'] = {'max_tokens': 2048}
        self.new['execution_optimization'] = {'new_local_execution': True}
        self.new['reuse_full'] = {'path': str(full), 'sha256': core.sha(full)}
        target_full = self.target / 'full/full.pkl'
        target_full.parent.mkdir(parents=True)
        target_full.write_bytes(full.read_bytes())
        core.write_json(target_full.with_suffix('.frozen.json'), seal)
        core.write_json(self.target / 'full/IDENTITY.json', {'plan_sha256': digest(self.new), 'state_identity': seal['state_identity']})
        self.addCleanup(patch.stopall)
        patch.object(branches, '_source_plan', side_effect=lambda folder: core.read(folder / 'PLAN.json')).start()

    def test_answer_only_change_imports_exact_states_without_answers_or_source_writes(self):
        original = {str(p): core.sha(p) for p in self.source.rglob('*') if p.is_file()}
        catalog = branches.prepare_catalog(self.source, self.new)
        result = branches.import_claims(self.target, self.new, catalog)
        self.assertEqual(result['claims'], ['c1'])
        self.assertEqual(result['api_calls'], 0)
        self.assertEqual(result['answers_copied'], 0)
        for condition in self.old['conditions']:
            old = core.read(self.source / 'claims/c1' / condition / 'STATE.json')
            new = core.read(self.target / 'claims/c1' / condition / 'STATE.json')
            self.assertEqual(new, {**old, 'plan_sha256': digest(self.new)})
            self.assertFalse((self.target / 'claims/c1' / condition / 'native.json').exists())
            self.assertEqual(core.sha(self.source / 'claims/c1' / condition / 'state.pkl'),
                             core.sha(self.target / 'claims/c1' / condition / 'state.pkl'))
        self.assertEqual(branches.import_claims(self.target, self.new, catalog), result)
        self.assertEqual(original, {str(p): core.sha(p) for p in self.source.rglob('*') if p.is_file()})

    def test_different_write_selector_qa_budget_source_claim_or_core_rejected(self):
        mutations = [lambda p: p['models'].update(write='other'), lambda p: p['models'].update(selector='other'),
                     lambda p: p['providers']['selector'].update(endpoint='other'),
                     lambda p: p['settings'].update(native_eval_k=2), lambda p: p['protocol'].update(n_obj=40),
                     lambda p: p['claims'][0].update(stem='Different question'),
                     lambda p: p['inputs']['source'].update(sha256='other'),
                     lambda p: p['code'].update({'frozen_core.py': 'changed'})]
        for mutate in mutations:
            new = copy.deepcopy(self.new)
            mutate(new)
            with self.assertRaises(ValueError):
                branches.prepare_catalog(self.source, new)

    def test_source_tampering_after_catalog_is_rejected(self):
        catalog = branches.prepare_catalog(self.source, self.new)
        path = self.source / 'claims/c1/S1@10/STATE.json'
        value = core.read(path)
        value['contexts']['native']['context'] += ' changed'
        core.write_json(path, value)
        with self.assertRaisesRegex(ValueError, 'changed after catalog'):
            branches.import_claims(self.target, self.new, catalog)
        self.assertFalse((self.target / 'claims').exists())

    def test_missing_or_failed_state_audit_and_changed_selector_refused(self):
        path = self.source / 'claims/c1/STATE_AUDIT.json'
        value = core.read(path)
        core.write_json(path, {**value, 'status': 'FAIL'})
        with self.assertRaisesRegex(ValueError, 'state audit'):
            branches.prepare_catalog(self.source, self.new)
        core.write_json(path, value)
        ranking_path = self.source / 'claims/c1/S2_ORDER.json'
        ranking = core.read(ranking_path)
        ranking['scores'][0]['llm_score'] = 1
        core.write_json(ranking_path, ranking)
        with self.assertRaisesRegex(ValueError, 'original response'):
            branches.prepare_catalog(self.source, self.new)

    def test_wrong_full_and_existing_partial_claim_refused(self):
        catalog = branches.prepare_catalog(self.source, self.new)
        path = self.target / 'full/full.frozen.json'
        seal = core.read(path)
        core.write_json(path, {**seal, 'state_identity': 'different-Full'})
        with self.assertRaisesRegex(ValueError, 'Destination Full'):
            branches.import_claims(self.target, self.new, catalog)
        core.write_json(path, seal)
        (self.target / 'claims/c1').mkdir(parents=True)
        with self.assertRaises(FileNotFoundError):
            branches.import_claims(self.target, self.new, catalog)

    def test_invalid_context_never_catalogued(self):
        path = self.source / 'claims/c1/S0/STATE.json'
        value = core.read(path)
        value['contexts']['residual']['context_sha256'] = 'wrong'
        core.write_json(path, value)
        with self.assertRaisesRegex(ValueError, 'Readout context'):
            branches.prepare_catalog(self.source, self.new)

    def test_missing_snapshot_never_catalogued(self):
        (self.source / 'claims/c1/S2@10/state.pkl').unlink()
        with self.assertRaises(FileNotFoundError):
            branches.prepare_catalog(self.source, self.new)

    def test_imported_snapshot_is_independent_and_later_corruption_is_detected(self):
        catalog = branches.prepare_catalog(self.source, self.new)
        branches.import_claims(self.target, self.new, catalog)
        original = self.source / 'claims/c1/S1@10/state.pkl'
        before = original.read_bytes()
        (self.target / 'claims/c1/S1@10/state.pkl').write_bytes(b'changed destination')
        self.assertEqual(original.read_bytes(), before)
        with self.assertRaisesRegex(ValueError, 'imported branch changed'):
            branches.import_claims(self.target, self.new, catalog)

    def test_failed_copy_leaves_no_partial_claim(self):
        catalog = branches.prepare_catalog(self.source, self.new)
        # Import the exact selector receipt first so the injected failure is
        # specifically inside the subsequent atomic claim-directory copy.
        branches._import_selector_receipts(self.target, self.source, catalog, 'c1')
        with patch.object(branches.shutil, 'copyfile', side_effect=OSError('injected copy failure')):
            with self.assertRaisesRegex(OSError, 'injected copy failure'):
                branches.import_claims(self.target, self.new, catalog)
        self.assertFalse((self.target / 'claims/c1').exists())
        self.assertEqual(list((self.target / 'claims').glob('.branch-import-*')), [])

    def test_selector_receipt_usage_and_normalization_import_without_new_cost(self):
        from memcut.execution_optimization.receipts import usage
        rid = core.read(self.source / 'claims/c1/S2_ORDER.json')['request_id']
        path = self.source / 'requests' / rid
        response = core.read(path / 'response.json')
        parsed = json.loads(response['choices'][0]['message']['content'])
        parsed['scores'][0]['score'] = '7'
        response['choices'][0]['message']['content'] = json.dumps(parsed)
        response['usage'] = {'prompt_tokens': 123, 'completion_tokens': 45}
        core.write_json(path / 'response.json', response)
        core.write_json(path / 'usage.json', {'usage': response['usage'], 'seconds': 1})
        core.write_json(path / 'NUMERIC_SCORE_NORMALIZATION.json', {
            'policy': 'selector-exact-numeric-json-string-v1', 'request_id': rid,
            'response_sha256': core.sha(path / 'response.json'),
            'changes': [{'id': 't1', 'before': '7', 'after': 7}]})
        catalog = branches.prepare_catalog(self.source, self.new)
        result = branches.import_claims(self.target, self.new, catalog)
        self.assertEqual(result['reused_selector_request_ids'], [rid])
        for name in ('request.json', 'response.json', 'usage.json', 'NUMERIC_SCORE_NORMALIZATION.json'):
            self.assertEqual(core.sha(path / name), core.sha(self.target / 'requests' / rid / name))
        self.assertTrue((self.target / 'requests' / rid / 'REUSED_RECEIPT.json').exists())
        self.assertEqual(usage(self.target)['new']['requests'], 0)
        self.assertEqual(usage(self.target)['imported']['requests'], 1)

    def test_unresolved_local_selector_request_never_repaired(self):
        catalog = branches.prepare_catalog(self.source, self.new)
        rid = core.read(self.source / 'claims/c1/S2_ORDER.json')['request_id']
        original = self.source / 'requests' / rid / 'request.json'
        target = self.target / 'requests' / rid / 'request.json'
        target.parent.mkdir(parents=True)
        target.write_bytes(original.read_bytes())
        with self.assertRaisesRegex(RuntimeError, 'Unresolved local selector'):
            branches.import_claims(self.target, self.new, catalog)
        self.assertFalse(target.with_name('response.json').exists())
        self.assertFalse((self.target / 'claims/c1').exists())


if __name__ == '__main__':
    unittest.main()
