"""Offline regression of retries, exact reuse, immutable identities and short circuit."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from memcut.efficient_execution import recovery, branches, replay
from memcut.source_anchored_v2.completions import CachedCompletions
from memcut.source_anchored_v2 import evaluation, runner
from memcut.native_memory.contract import digest
from memcut.answer_execution import early_stop


def fixture():
    c=runner.normalize_claim({'claim_id':'c1','stem':'Which?', 'target_choice_id':'right',
        'choices':[{'id':'right','text':'Correct'},{'id':'wrong','text':'Wrong'},{'id':'other','text':'Other'},{'id':'last','text':'Last'}]})
    ctx={'context':'memory','context_sha256':digest('memory'),'state_identity':'state','empty':False}
    req={**evaluation.answer_request(c,ctx,c['rotations'][0],'fixture'),'max_tokens':1024}
    return c,ctx,req


def raw(text,finish='stop'):
    return {'choices':[{'message':{'content':text},'finish_reason':finish}],'usage':{'prompt_tokens':10,'completion_tokens':5}}


class EfficientTests(unittest.TestCase):
    def test_invalid_then_valid_cached_resume_and_unchanged_body(self):
        c,ctx,req=fixture();calls=[]
        right=next(k for k,v in c['rotations'][0]['display_to_choice'].items() if v=='right')
        with tempfile.TemporaryDirectory() as td:
            responses=[raw('bad json'),raw(json.dumps({'answer':right}))]
            def create(**kw):calls.append(kw);return responses[len(calls)-1]
            cache=recovery.RecoveryCache(CachedCompletions(td,create,endpoint='fixture'))
            response=cache.complete(req);self.assertEqual(len(calls),2)
            second=copy.deepcopy(calls[1]);second.pop('extra_headers');self.assertEqual(second,calls[0]);self.assertEqual(second['max_tokens'],1024)
            self.assertEqual(cache.complete(req)['request_id'],response['request_id']);self.assertEqual(len(calls),2)
            expected=replay.request_identity(cache,req);actual=runner.read(Path(td)/response['request_id']/'request.json')

    def test_valid_wrong_is_never_retried_and_trial2_is_not_sent(self):
        c,ctx,req=fixture();calls=[]
        wrong=next(k for k,v in c['rotations'][0]['display_to_choice'].items() if v!='right')
        with tempfile.TemporaryDirectory() as td:
            def create(**kw):calls.append(kw);return raw(json.dumps({'answer':wrong}))
            cache=recovery.RecoveryCache(CachedCompletions(td,create,endpoint='fixture'))
            result=early_stop.evaluate(c,ctx,cache,'fixture')
            self.assertFalse(result['recovered']);self.assertEqual(len(calls),1);self.assertEqual(result['answer_execution']['skipped_trials'],[2])

    def test_invalid_exhaustion_is_not_a_zero_score(self):
        c,ctx,req=fixture();calls=[]
        with tempfile.TemporaryDirectory() as td:
            def create(**kw):calls.append(kw);return raw('{"answer":"z"}')
            cache=recovery.RecoveryCache(CachedCompletions(td,create,endpoint='fixture'))
            with self.assertRaisesRegex(RuntimeError,'exhausted'):early_stop.evaluate(c,ctx,cache,'fixture')
            self.assertEqual(len(calls),3)
            with self.assertRaises(RuntimeError):cache.complete(evaluation.answer_request(c,ctx,c['rotations'][0],'fixture'))
            self.assertEqual(len(calls),3)

    def test_timeout_unknown_dispatch_is_not_retried(self):
        _,_,req=fixture();calls=[]
        with tempfile.TemporaryDirectory() as td:
            def create(**kw):calls.append(kw);raise TimeoutError('unknown')
            cache=recovery.RecoveryCache(CachedCompletions(td,create,endpoint='fixture'))
            with self.assertRaises(TimeoutError):cache.complete(req)
            with self.assertRaises(RuntimeError):cache.complete(req)
            self.assertEqual(len(calls),1)

    def test_connection_retry_is_bounded_and_charge_unknown(self):
        _,_,req=fixture();calls=[]
        APIConnectionError=type('APIConnectionError',(Exception,),{})
        with tempfile.TemporaryDirectory() as td:
            def create(**kw):calls.append(kw);raise APIConnectionError('fixture')
            cache=recovery.RecoveryCache(CachedCompletions(td,create,endpoint='fixture'))
            with self.assertRaises(APIConnectionError):cache.complete(req)
            self.assertEqual(len(calls),2)

    def test_proof_rejects_retry_of_valid_answer(self):
        _,_,req=fixture()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);calls=[]
            def create(**kw):calls.append(kw);return raw('bad' if len(calls)==1 else '{"answer":"a"}')
            cache=recovery.RecoveryCache(CachedCompletions(root/'requests',create,endpoint='fixture'))
            result=cache.complete(req);expected=replay.request_identity(cache,req)
            actual=runner.read(root/'requests'/result['request_id']/'request.json')
            recovery.verify_received_attempt(root,expected,actual)
            p=root/'requests'/digest(expected)/'response.json';runner.write_json(p,raw('{"answer":"a"}'))
            with self.assertRaisesRegex(ValueError,'valid answer'):recovery.verify_received_attempt(root,expected,actual)

    def test_binding_allows_answer_changes_but_not_deletion_or_native_k(self):
        base={'system':'raw_turn_rag','claims':[],'models':{'answer':'a','selector':'s'},'providers':{'answer':{},'selector':{}},'settings':{'native_eval_k':16},'protocol':{'k':10}}
        for system in ('memorybank','raw_turn_rag','nemori','lightmem'):
            p=copy.deepcopy(base);p['system']=system;q=copy.deepcopy(p)
            q['models']['answer']='b';q['answer_execution']={'new':True};q['efficient_execution']={'policy':'new'}
            self.assertEqual(branches._binding(p),branches._binding(q))
            q['settings']['native_eval_k']=2;self.assertNotEqual(branches._binding(p),branches._binding(q))

    def test_selector_missing_ids_retried_without_candidate_change(self):
        request={'model':'fixture','messages':[{'role':'system','content':'Rank candidate source groups'}, {'role':'user','content':json.dumps({'candidates':[{'id':'g1'},{'id':'g2'}]})}]}
        calls=[]
        with tempfile.TemporaryDirectory() as td:
            def create(**kw):
                calls.append(kw);return raw(json.dumps({'scores':[{'id':'g1','score':'7','reason':'x'}]+([] if len(calls)==1 else [{'id':'g2','score':1,'reason':'y'}])}))
            cache=recovery.RecoveryCache(CachedCompletions(td,create,endpoint='fixture'))
            result=cache.complete(request);self.assertEqual(len(calls),2);self.assertEqual(json.loads(result['text'])['scores'][0]['score'],7)
            self.assertEqual(calls[0]['messages'],calls[1]['messages'])


if __name__=='__main__':unittest.main()
