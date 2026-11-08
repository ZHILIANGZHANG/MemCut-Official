"""Bailian Qwen3.8-Flash text profile; explicitly disabled thinking."""
import copy
MODEL='qwen3.8-flash'
ENDPOINT='https://dashscope.aliyuncs.com/compatible-mode/v1'
PROFILE={'version':'qwen38-flash-bailian-text-v1','model':MODEL,'endpoint':ENDPOINT,'enable_thinking':False,'preserve_thinking':False,'automatic_retries':0}
def normalize(q):
 q=copy.deepcopy(q)
 if q.get('model')!=MODEL:return q
 extra=q.setdefault('extra_body',{})
 for k in ('enable_thinking','preserve_thinking'):
  if q.pop(k,False) is not False or extra.get(k,False) is not False:raise ValueError('Conflicting frozen Qwen thinking setting')
  extra[k]=False
 if 'thinking_budget' in extra or 'reasoning_effort' in q or 'reasoning_effort' in extra:raise ValueError('Thinking parameters conflict with disabled mode')
 return q

def install():
 from memcut.source_anchored_v2.completions import CachedCompletions
 from memcut.execution_optimization import receipts
 from memcut.efficient_execution import replay
 old=CachedCompletions.complete;old_identity=receipts.request_identity
 def checked(cache,q):
  if q.get('model')==MODEL and (cache.endpoint.rstrip('/')!=ENDPOINT or cache.compatibility!='openai'):raise ValueError('Qwen profile requires Beijing Bailian endpoint')
  return normalize(q)
 def complete(self,q,**kw):return old(self,checked(self,q),**kw)
 def identity(cache,q):return old_identity(cache,checked(cache,q))
 CachedCompletions.complete=complete
 receipts.request_identity=identity
 replay.base_request_identity=identity
