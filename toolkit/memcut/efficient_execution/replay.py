"""Validate inherited readouts against actual provider receipts, without dispatch."""
import copy
import json
import re
from pathlib import Path
from memcut.native_memory.contract import digest
from memcut.source_anchored_v2 import runner as core, evaluation
from memcut.source_anchored_v2.candidates import rank_leakcut
from memcut.source_anchored_v2.completions import parse_json
from memcut.model_adapters.glm import MODEL, GLMCompletions
from memcut.source_anchored_v2.completions import CachedCompletions
from memcut.execution_optimization.receipts import request_identity as base_request_identity

def request_identity(cache,request):
    while hasattr(cache,"delegate"):
        cache=cache.delegate
    return base_request_identity(cache,request)


def identity(plan, role, request):
    def forbidden(**kwargs):
        raise AssertionError('Offline receipt validation cannot call API')
    p=plan['providers'][role]
    cache=(GLMCompletions(Path('/unused'),forbidden) if plan['models'][role]==MODEL else
           CachedCompletions(Path('/unused'),forbidden,endpoint=p['endpoint'],compatibility=p['compatibility']))
    return request_identity(cache,request)


def receipt(folder, rid):
    if not isinstance(rid,str) or not re.fullmatch('[0-9a-f]{64}',rid):
        raise ValueError('Invalid request ID')
    p=Path(folder)/'requests'/rid
    actual=core.read(p/'request.json')
    if digest(actual)!=rid:
        raise ValueError('Request hash mismatch')
    raw=core.read(p/'response.json')['choices'][0]
    if raw['finish_reason']!='stop' or not isinstance(raw['message'].get('content'),str):
        raise ValueError('Require a complete received answer')
    files=[p/'request.json',p/'response.json']
    if (p/'usage.json').exists():files.append(p/'usage.json')
    return actual,{'text':raw['message']['content'],'request_id':rid,'finish_reason':'stop'},files


def semantic_identity(value):
    result=copy.deepcopy(value)
    headers=result['request'].pop('extra_headers',{})
    if any(not k.startswith('X-MemCut-') for k in headers):
        raise ValueError('Unknown request header cannot be ignored')
    return result


def selector_receipts(folder, plan, pool, ranking):
    files=[]
    class Replay:
        def complete(self, request):
            actual,response,paths=receipt(folder,ranking['request_id']); files.extend(paths)
            expected=identity(plan,'selector',request)
            stripped=semantic_identity(actual)
            if stripped!=expected:
                # Historical, explicitly audited grouped presentation. Never use
                # this fallback to generate new selector requests.
                payload=json.loads(request['messages'][-1]['content'])
                ids=[g['id'] for g in payload['candidates']]
                clarified=copy.deepcopy(request)
                clarified['messages'][0]['content']+='\nOutput schema clarification: There are exactly '+str(len(ids))+' candidate GROUPS. Return exactly these group IDs once each: '+', '.join(ids)+'. Score candidate groups from the candidates array, not individual memory objects. Do not create any other group ID.'
                alternatives=[identity(plan,'selector',clarified)]
                objects={o['id']:o for o in payload['objects']}
                used={oid for g in payload['candidates'] for oid in g['object_ids']}
                grouped={'deletion_request':payload['deletion_request'],'candidates':[
                    {'id':g['id'],'memory_objects':[objects[oid] for oid in g['object_ids']]} for g in payload['candidates']],
                    'other_exposed_objects':[o for o in payload['objects'] if o['id'] not in used]}
                clarified['messages'][-1]['content']=json.dumps(grouped,ensure_ascii=False,separators=(',',':'))
                alternatives.append(identity(plan,'selector',clarified))
                if stripped not in alternatives:
                    raise ValueError('Inherited selector differs from original candidate content or scoring prompt')
            parsed=parse_json(response['text'])
            for row in parsed.get('scores',[]):
                val=row.get('score')
                if isinstance(val,str) and re.fullmatch(r'(?:0|[1-9][0-9]*)(?:\.[0-9]+)?',val) and 0<=float(val)<=10:
                    row['score']=float(val)
            return {**response,'text':json.dumps(parsed,ensure_ascii=False)}
    expected=rank_leakcut(pool,Replay(),plan['models']['selector'])
    if expected!={k:v for k,v in ranking.items() if k not in {'plan_sha256','pool_sha256'}}:
        raise ValueError('Inherited selector scores or ordering disagree with its receipt')
    return files


def validate_source_answers(folder, plan, claim):
    files=[]
    for condition in plan['conditions']:
        state=core.validate_state(folder,plan,claim,condition)
        for family,ctx in state['contexts'].items():
            path=Path(folder)/'claims'/claim['claim_id']/condition/(family+'.json')
            envelope=core.read(path); result=envelope['result']; files.append(path)
            if envelope['plan_sha256']!=digest(plan) or result['context_sha256']!=ctx['context_sha256'] or result['state_identity']!=ctx['state_identity']:
                raise ValueError('Historical answer is not bound to this readout')
            if ctx['empty']:
                if family!='residual' or result['answers'] or result['recovered'] or not result.get('evidence_derived_empty'):
                    raise ValueError('Invalid historical structural empty')
                continue
            answers=result['answers']
            if len(answers) not in (1,2) or (len(answers)==1 and (not plan.get('answer_execution') or answers[0]['target_hit'])):
                raise ValueError('Missing historical trial')
            for saved,rotation in zip(answers,claim['rotations']):
                actual,response,paths=receipt(folder,saved['request_id']); files.extend(paths)
                expected=evaluation.answer_request(claim,ctx,rotation,plan['models']['answer'])
                # Historical budgets remain explicitly bound to their receipts.
                budget=actual['request'].get('max_tokens',actual['request'].get('max_completion_tokens'))
                if type(budget)is not int or budget<=0:raise ValueError('Invalid historical QA budget')
                expected['max_tokens']=budget
                if semantic_identity(actual)!=identity(plan,'answer',expected):
                    raise ValueError('Historical context/options differ from actual QA request')
                if evaluation.score_response(response,rotation,claim['target_choice_id'],claim['choices'])!=saved:
                    raise ValueError('Historical score differs from raw response')
            if result['recovered']!=(len(answers)==2 and all(a['target_hit'] for a in answers)):
                raise ValueError('Historical recovery differs from 2/2')
    return files


def answer_aliases(source,plan):
    """Only audited answer receipts with an identical normalized model input.

    Transport retry headers may differ; budget, model, options, context and all
    sampling parameters must match. Never turn an old400 answer into a1024 one.
    """
    folder=Path(source);old=core.read(folder/'PLAN.json');aliases={}
    claims={c['claim_id']:c for c in old['claims']}
    for claim in plan['claims']:
        for cond in plan['conditions']:
            state=core.validate_state(folder,old,claims[claim['claim_id']],cond)
            for fam,ctx in state['contexts'].items():
                result=core.read(folder/'claims'/claim['claim_id']/cond/(fam+'.json'))['result']
                for a,rot in zip(result['answers'],claim['rotations']):
                    actual,_,_=receipt(folder,a['request_id'])
                    req={**evaluation.answer_request(claim,ctx,rot,plan['models']['answer']),'max_tokens':1024}
                    expected=identity(plan,'answer',req)
                    if semantic_identity(actual)!=expected:continue
                    key=digest(expected)
                    if key in aliases and aliases[key]['request_id']!=a['request_id']:
                        raise ValueError('Conflicting historical answers; cannot choose a sample')
                    aliases[key]={'request_id':a['request_id'],'identity':actual,'source':str(folder),
                                  'request_sha256':core.sha(folder/'requests'/a['request_id']/'request.json'),
                                  'response_sha256':core.sha(folder/'requests'/a['request_id']/'response.json')}
    return aliases


class AliasCache:
    def __init__(self,delegate,aliases):self.delegate=delegate;self.aliases=aliases
    def __getattr__(self,name):return getattr(self.delegate,name)
    @property
    def before_dispatch(self):return self.delegate.before_dispatch
    @before_dispatch.setter
    def before_dispatch(self,value):self.delegate.before_dispatch=value
    def complete(self,request,**kwargs):
        expected=request_identity(self.delegate,request);entry=self.aliases.get(digest(expected))
        if not entry:return self.delegate.complete(request,**kwargs)
        # Do not mask any new unresolved attempt with an inherited response.
        local=Path(self.folder)/digest(expected)
        if (local/'request.json').exists() and not (local/'response.json').exists():
            raise RuntimeError('Unresolved local answer dispatch')
        actual=entry['identity']
        if semantic_identity(actual)!=expected:raise ValueError('Answer alias changed input')
        source=Path(entry['source'])/'requests'/entry['request_id']
        if core.sha(source/'request.json')!=entry['request_sha256'] or core.sha(source/'response.json')!=entry['response_sha256']:raise ValueError('Historical answer receipt changed')
        result=self.delegate.complete(actual['request'],**kwargs)
        if not result['cached'] or result['request_id']!=entry['request_id']:raise RuntimeError('Historical answer unexpectedly dispatched')
        core.sealed_write(Path(self.folder)/entry['request_id']/'REUSED_ANSWER_RESOLUTION.json',entry)
        return result


class AliasGateway:
    def __init__(self,delegate,aliases):self.delegate=delegate;self.aliases=aliases;self.caches={}
    def __getattr__(self,name):return getattr(self.delegate,name)
    def for_role(self,role):
        if role not in self.caches:
            cache=self.delegate.for_role(role)
            self.caches[role]=AliasCache(cache,self.aliases) if role=='answer' else cache
        return self.caches[role]
