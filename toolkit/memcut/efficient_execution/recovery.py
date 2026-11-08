"""Bounded, receipted retries of unusable responses, never of a valid wrong answer."""
import copy
import json
import re
from pathlib import Path
from memcut.native_memory.contract import digest
from memcut.source_anchored_v2 import runner as core
from memcut.source_anchored_v2.completions import parse_json
from .replay import request_identity

POLICY={'version':'received-invalid-same-body-v1','maximum_extra_attempts':2,
        'connection_extra_attempts':1,'budget_increases':False,'prompt_changes':False,
        'valid_wrong_answer_retry':False,'unknown_dispatch_retry':False}


def validate(request,response):
    prompt=request['messages'][0]['content']
    parsed=parse_json(response['text'])
    if prompt.startswith('Rank candidate source groups'):
        payload=json.loads(request['messages'][-1]['content']);ids={g['id'] for g in payload['candidates']}
        rows=parsed.get('scores'); seen=set()
        if not isinstance(rows,list):raise ValueError('Missing scores')
        for row in rows:
            if not isinstance(row,dict):raise ValueError('Malformed score')
            score=row.get('score')
            if isinstance(score,str) and re.fullmatch(r'(?:0|[1-9][0-9]*)(?:\.[0-9]+)?',score):score=float(score)
            if type(score) not in (int,float) or not 0<=score<=10 or not isinstance(row.get('reason'),str) or row.get('id') not in ids or row['id'] in seen:
                raise ValueError('Invalid score, reason or candidate coverage')
            row['score']=score;seen.add(row['id'])
        if seen!=ids:raise ValueError('Incomplete candidate coverage')
        return {**response,'text':json.dumps(parsed,ensure_ascii=False)}
    if prompt.startswith('Answer the multiple-choice question'):
        # Match exactly the same accepted display-label forms as score_response;
        # no target answer or correctness is available to the retry layer.
        text=request['messages'][-1]['content'];options=text.rsplit('\nOptions:\n',1)[1].split('\n\n',1)[0]
        chunks=re.split(r'(?m)^([a-z])\. ',options)
        pairs={chunks[i]:chunks[i+1].removesuffix('\n') for i in range(1,len(chunks),2)}
        label=parsed.get('answer')
        if not isinstance(label,str):raise ValueError('Missing answer label')
        label=label.strip()
        if label.lower() not in pairs and not any(label==f'{form}{sep} {t}' for k,t in pairs.items() for form in (k,k.upper()) for sep in ('.',')')):
            raise ValueError('Invalid answer label')
        return response
    return response


class RecoveryCache:
    def __init__(self,delegate):self.delegate=delegate
    def __getattr__(self,name):return getattr(self.delegate,name)
    @property
    def before_dispatch(self):return self.delegate.before_dispatch
    @before_dispatch.setter
    def before_dispatch(self,value):self.delegate.before_dispatch=value
    def complete(self,request,**kwargs):
        prompt=request.get('messages',[{}])[0].get('content','')
        if not prompt.startswith(('Answer the multiple-choice question','Rank candidate source groups')):
            return self.delegate.complete(request,**kwargs)
        origin=digest(request_identity(self.delegate,request)); failed=[];limit=2
        for attempt in range(3):
            current=copy.deepcopy(request)
            if attempt:current.setdefault('extra_headers',{})['X-MemCut-Efficient-Retry']=str(attempt)
            rid=digest(request_identity(self.delegate,current));p=Path(self.folder)/rid
            try:
                response=self.delegate.complete(current,**kwargs)
                result=validate(request,response)
            except Exception as exc:
                # Recovery is authorized only by a durable, received unusable
                # response. Unknown dispatches and programming errors propagate.
                if not (p/'response.json').exists():
                    error=core.read(p/'error.json') if (p/'error.json').exists() else None
                    if error not in ({'type':'APIConnectionError'},{'type':'InternalServerError'}) or attempt>=1:raise
                    failed.append({'request_id':rid,'error_sha256':core.sha(p/'error.json'),'charge':'unknown'})
                    limit=1
                    continue
                raw=core.read(p/'response.json')['choices'][0]
                if raw.get('finish_reason') not in {'stop','length'}:raise
                if raw.get('finish_reason')=='stop' and raw['message'].get('content'):
                    try:validate(request,{'text':raw['message']['content']})
                    except (ValueError,KeyError,IndexError,TypeError):pass
                    else:raise
                failed.append({'request_id':rid,'response_sha256':core.sha(p/'response.json')})
                core.sealed_write(p/'INVALID_EXECUTION_RESPONSE.json',{'policy':POLICY,'origin':origin,'attempt':attempt,'response_sha256':core.sha(p/'response.json')})
                if attempt>=limit:raise RuntimeError('Bounded invalid-response retries exhausted') from exc
                continue
            if failed:
                core.sealed_write(Path(self.folder)/origin/'RETRY_PROOF.json',{'policy':POLICY,'origin':origin,'selected_request_id':rid,'failed':failed})
            return result


class RecoveryGateway:
    def __init__(self,delegate):self.delegate=delegate;self.caches={}
    def __getattr__(self,name):return getattr(self.delegate,name)
    def for_role(self,role):
        if role not in self.caches:self.caches[role]=RecoveryCache(self.delegate.for_role(role)) if role in {'answer','selector'} else self.delegate.for_role(role)
        return self.caches[role]


def verify_received_attempt(folder,expected,actual):
    """Recheck the first-usable-attempt proof without trusting saved scores."""
    headers=actual['request'].get('extra_headers',{})
    if not headers:return
    if set(headers)!={'X-MemCut-Efficient-Retry'}:
        note=Path(folder)/'requests'/digest(actual)/'REUSED_ANSWER_RESOLUTION.json'
        if not note.exists():raise ValueError('Unknown or unproven QA retry')
        entry=core.read(note)
        if entry['identity']!=actual or entry['request_id']!=digest(actual):raise ValueError('Inherited retry alias differs')
        return
    attempt=int(headers['X-MemCut-Efficient-Retry'])
    if not 1<=attempt<=2:raise ValueError('Retry count exceeds policy')
    origin=digest(expected);proof=core.read(Path(folder)/'requests'/origin/'RETRY_PROOF.json')
    failed=[]
    for n in range(attempt):
        previous=copy.deepcopy(expected)
        if n:previous['request'].setdefault('extra_headers',{})['X-MemCut-Efficient-Retry']=str(n)
        rid=digest(previous);p=Path(folder)/'requests'/rid
        if core.read(p/'request.json')!=previous:raise ValueError('Retry changed input')
        if not (p/'response.json').exists():
            if n!=0 or attempt!=1 or core.read(p/'error.json') not in ({'type':'APIConnectionError'},{'type':'InternalServerError'}):raise ValueError('Unproven network retry')
            failed.append({'request_id':rid,'error_sha256':core.sha(p/'error.json'),'charge':'unknown'})
        else:
            raw=core.read(p/'response.json')['choices'][0]
            if raw['finish_reason']=='stop' and raw['message'].get('content'):
                try:validate(expected['request'],{'text':raw['message']['content']})
                except (ValueError,KeyError,TypeError,IndexError):pass
                else:raise ValueError('A valid answer was retried')
            elif raw['finish_reason'] not in {'stop','length'}:raise ValueError('Unsupported retry reason')
            failed.append({'request_id':rid,'response_sha256':core.sha(p/'response.json')})
    if proof!={'policy':POLICY,'origin':origin,'selected_request_id':digest(actual),'failed':failed}:raise ValueError('Retry proof changed')
