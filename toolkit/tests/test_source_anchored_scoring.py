"""Two fixed trials: spending, strict scoring, shared cohort and old cache reuse."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from memcut.native_memory.contract import digest
from memcut.source_anchored_v2 import runner
from memcut.source_anchored_v2.completions import CachedCompletions
from memcut.source_anchored_v2.evaluation import answer_request, evaluate, summarize


def authored_claim():
    ids = ['yes','no','maybe','unknown']
    return {'claim_id':'c1','stem':'Which?', 'choices':[{'id':i,'text':i} for i in ids],
            'target_choice_id':'yes', 'rotations':[
                {'trial':j+1,'display_to_choice':dict(zip('abcd',ids[j:]+ids[:j]))} for j in range(3)]}


class ScoringTests(unittest.TestCase):
    def test_first_two_fixed_without_mutating_authored_three(self):
        source = authored_claim(); before = copy.deepcopy(source)
        claim = runner.normalize_claim(source)
        self.assertEqual(claim['rotations'],source['rotations'][:2])
        self.assertEqual(source,before)
        with self.assertRaises(ValueError):
            runner.normalize_claim({**source,'rotations':source['rotations'][1:]})

    def test_only_two_dispatches_and_one_wrong_fails(self):
        claim = runner.normalize_claim(authored_claim())
        ctx={'context':'stored fact','context_sha256':digest('stored fact'),'state_identity':'same','empty':False}
        for correct in [(True,True),(True,False),(False,True)]:
            calls=[]
            def create(**q):
                i=len(calls); calls.append(q)
                mapping=claim['rotations'][i]['display_to_choice']
                label=next(k for k,v in mapping.items() if (v=='yes')==correct[i])
                return {'choices':[{'message':{'content':json.dumps({'answer':label})},'finish_reason':'stop'}]}
            with tempfile.TemporaryDirectory() as td:
                result=evaluate(claim,ctx,CachedCompletions(td,create,endpoint='offline-fixture'),'fixture')
            self.assertEqual(len(calls),2)
            self.assertEqual(result['recovered'],all(correct))

    def test_old_first_two_response_bytes_are_reused_without_dispatch(self):
        source=authored_claim(); claim=runner.normalize_claim(source)
        ctx={'context':'same stored context','context_sha256':digest('same stored context'),'state_identity':'same','empty':False}
        with tempfile.TemporaryDirectory() as td:
            for rot in source['rotations']:
                req=answer_request(source,ctx,rot,'fixture')
                identity={'endpoint':'offline-fixture','compatibility':'openai','request':req}
                p=Path(td)/digest(identity);p.mkdir()
                (p/'request.json').write_text(json.dumps(identity))
                label=next(k for k,v in rot['display_to_choice'].items() if v=='yes')
                (p/'response.json').write_text(json.dumps({'choices':[{'message':{'content':json.dumps({'answer':label})},'finish_reason':'stop'}]}))
            def forbidden(**q): raise AssertionError('No new model request allowed')
            cache=CachedCompletions(td,forbidden,endpoint='offline-fixture')
            self.assertTrue(evaluate(claim,ctx,cache,'fixture')['recovered'])
            self.assertEqual(len(cache.calls),2)
            self.assertTrue(all(c['cached'] for c in cache.calls))
            with self.assertRaises(ValueError): evaluate(source,ctx,cache,'fixture')
            self.assertEqual(len(cache.calls),2)

    def test_shared_full_residual_cohort_and_incomplete_or_three_trials_rejected(self):
        claims=[runner.normalize_claim(authored_claim()),runner.normalize_claim({**authored_claim(),'claim_id':'c2'})]
        rows=[]
        for c in claims:
            for condition in ('Full','S0'):
                for family in ('native','residual'):
                    flags=[True,c['claim_id']=='c1'] if family=='residual' else [False,False]
                    rows.append({'claim_id':c['claim_id'],'condition':condition,'family':family,
                                 'result':{'recovered':all(flags),'state_identity':condition,
                                           'answers':[{'trial':i+1,'valid':True,'target_hit':hit} for i,hit in enumerate(flags)]}})
        report=summarize(rows,claims,['Full','S0'])
        self.assertEqual(report['scoring']['rotation_count'],2)
        self.assertEqual(report['metrics']['residual-S0']['rate'],.5)
        self.assertEqual(report['metrics']['native-S0']['conditional']['claim_ids'],['c1'])
        self.assertIsNone(summarize(rows[:-1],claims,['Full','S0'])['metrics']['residual-S0']['rate'])
        for answers in [[{'trial':1,'valid':True,'target_hit':False}],
                        [{'trial':i,'valid':True,'target_hit':False} for i in (1,2,3)]]:
            bad=copy.deepcopy(rows);bad[0]['result']['answers']=answers
            with self.assertRaises(ValueError): summarize(bad,claims,['Full','S0'])


if __name__ == '__main__': unittest.main()
