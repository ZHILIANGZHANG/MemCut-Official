"""Offline: unchanged two-trial conjunction, strict skips, isolation and errors."""
import copy
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from memcut.answer_execution import early_stop
from memcut.native_memory.contract import digest
from memcut.source_anchored_v2 import evaluation, runner


def claim(cid='c1'):
    return runner.normalize_claim({'claim_id': cid, 'stem': 'Which?', 'target_choice_id': 'right',
        'choices': [{'id': 'right', 'text': 'Correct'}]+[{'id': f'w{i}', 'text': f'Wrong{i}'} for i in range(3)]})


def context(text='complete original stored memory', empty=False):
    return {'context': text, 'context_sha256': digest(text), 'state_identity': 'same_state', 'empty': empty}


class Recorder:
    def __init__(self, c, hits):
        self.claim, self.hits, self.requests = c, hits, []

    def complete(self, request):
        trial = len(self.requests)
        self.requests.append(request)
        rotation = self.claim['rotations'][trial]
        label = next(k for k,v in rotation['display_to_choice'].items()
                     if (v==self.claim['target_choice_id']) == self.hits[trial])
        return {'text': json.dumps({'answer':label,'reason':'fixture'}), 'finish_reason':'stop',
                'request_id': digest(request)}


class EarlyStopTests(unittest.TestCase):
    def test_all_truth_table_cases_keep_outcome_requests_and_real_answers(self):
        for hits in itertools.product([False,True],repeat=2):
            with self.subTest(hits=hits):
                c,ctx=claim(),context();old,new=Recorder(c,hits),Recorder(c,hits)
                expected=evaluation.evaluate(c,ctx,old,'fixture')
                actual=early_stop.evaluate(c,ctx,new,'fixture')
                n=2 if hits[0] else 1
                self.assertEqual(actual['recovered'],expected['recovered'])
                self.assertEqual(new.requests,old.requests[:n])
                self.assertEqual(actual['answers'],expected['answers'][:n])
                self.assertEqual(actual['answer_execution']['skipped_trials'],[] if hits[0] else [2])
                early_stop.validate_result(actual,c,'native')
                if not hits[0]:
                    with self.assertRaisesRegex(ValueError,'Missing or invalid'):
                        evaluation.summarize([{'claim_id':'c1','condition':'Full','family':'native','result':actual}],
                                             [c],['Full'])

    def test_invalid_or_truncated_answers_never_become_valid_misses(self):
        for invalid in [('{"answer":"z"}','stop'),('not json','stop'),('{"answer":"b"}','length')]:
            with self.subTest(invalid=invalid):
                class Invalid:
                    def complete(self,req):
                        return {'text':invalid[0],'finish_reason':invalid[1],'request_id':'bad'}
                with self.assertRaises((ValueError,json.JSONDecodeError)):
                    early_stop.evaluate(claim(),context(),Invalid(),'fixture')

    def test_transport_error_after_correct_first_answer_is_not_zero(self):
        c=claim();rec=Recorder(c,[True,True]);original=rec.complete
        def fail(req):
            if rec.requests:raise TimeoutError('fixture')
            return original(req)
        rec.complete=fail
        with self.assertRaises(TimeoutError):early_stop.evaluate(c,context(),rec,'fixture')

    def test_structural_empty_has_no_calls_and_is_residual_only(self):
        c=claim();rec=Recorder(c,[]);r=early_stop.evaluate(c,context('',True),rec,'fixture')
        self.assertEqual(rec.requests,[]);self.assertEqual(r['answers'],[])
        early_stop.validate_result(r,c,'residual')
        with self.assertRaises(ValueError):early_stop.validate_result(r,c,'native')
        with self.assertRaises(ValueError):early_stop.evaluate(c,context('not empty',True),rec,'fixture')

    def test_missing_correct_trial2_forged_hit_and_wrong_skip_are_rejected(self):
        c=claim();r=early_stop.evaluate(c,context(),Recorder(c,[False,True]),'fixture')
        def missing_second(x):
            rr=early_stop.evaluate(c,context(),Recorder(c,[True,True]),'fixture')
            rr['answers']=rr['answers'][:1];rr['recovered']=False
            rr['answer_execution']=early_stop.execution(rr['answers']);return rr
        mutations=[missing_second,
            lambda x:{**x,'answer_execution':{**x['answer_execution'],'skipped_trials':[]}},
            lambda x:{**x,'recovered':True},
            lambda x:{**x,'answers':[{**x['answers'][0],'valid':False}]},
            lambda x:{**x,'answers':[{**x['answers'][0],'trial':2}]},
            lambda x:{**x,'answers':[{**x['answers'][0],'target_hit':True}]}]
        for mutate in mutations:
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):
                early_stop.validate_result(mutate(copy.deepcopy(r)),c,'residual')

    def test_metrics_cohort_and_paired_differences_match_full_trials(self):
        cs=[claim('c'+str(i+1)) for i in range(4)];conditions=['Full','S0','S1@10','S2@10']
        rows,new=[],[]
        truth=list(itertools.product([False,True],repeat=2))
        for i,c in enumerate(cs):
            for j,cond in enumerate(conditions):
                for k,fam in enumerate(['native','residual']):
                    hits=truth[(i+j+k)%4];envelope={'claim_id':c['claim_id'],'condition':cond,'family':fam}
                    rows.append({**envelope,'result':evaluation.evaluate(c,context(),Recorder(c,hits),'fixture')})
                    new.append({**envelope,'result':early_stop.evaluate(c,context(),Recorder(c,hits),'fixture')})
        for a,b in [(rows,new),(rows[:-1],new[:-1])]:
            old=evaluation.summarize(a,cs,conditions);actual=early_stop.summarize(b,cs,conditions)
            for field in ['metrics','paired_differences','planned_claims']:self.assertEqual(actual[field],old[field])
        with self.assertRaises(ValueError):early_stop.summarize(new+[new[0]],cs,conditions)

    def test_changed_context_and_reordered_rotations_fail_before_call(self):
        c=claim();rec=Recorder(c,[True,True]);ctx=context();ctx['context']='changed'
        with self.assertRaises(ValueError):early_stop.evaluate(c,ctx,rec,'fixture')
        c['rotations'].reverse()
        with self.assertRaises(ValueError):early_stop.evaluate(c,context(),rec,'fixture')
        self.assertEqual(rec.requests,[])



if __name__=='__main__':unittest.main()
