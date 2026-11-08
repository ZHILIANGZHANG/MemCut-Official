import json,tempfile,unittest
from pathlib import Path
from memcut.model_adapters.qwen import MODEL,ENDPOINT,normalize,install
from memcut.source_anchored_v2.completions import CachedCompletions
from memcut.efficient_execution import replay
class QwenTest(unittest.TestCase):
 @classmethod
 def setUpClass(cls):install()
 def test_profile(self):
  q={'model':MODEL,'max_tokens':1024};n=normalize(q)
  self.assertEqual(n['extra_body'],{'enable_thinking':False,'preserve_thinking':False});self.assertNotIn('extra_body',q)
  with self.assertRaises(ValueError):normalize({'model':MODEL,'extra_body':{'enable_thinking':True}})
  other={'model':'deepseek-flash'};self.assertEqual(normalize(other),other)
 def test_receipt_identity_cache(self):
  with tempfile.TemporaryDirectory() as d:
   calls=[]
   def create(**kw):
    calls.append(kw)
    return {'model':MODEL,'choices':[{'finish_reason':'stop','message':{'content':'{"ok":true}'}}],'usage':{'prompt_tokens':10,'completion_tokens':5}}
   c=CachedCompletions(Path(d),create,endpoint=ENDPOINT)
   q={'model':MODEL,'max_tokens':64,'messages':[{'role':'user','content':'JSON'}]}
   r=c.complete(q);self.assertTrue(c.complete(q)['cached']);self.assertEqual(len(calls),1)
   self.assertEqual(json.loads((Path(d)/r['request_id']/'request.json').read_text()),replay.request_identity(c,q))
   with self.assertRaises(ValueError):CachedCompletions(Path(d),create,endpoint='https://example.com').complete(q)
if __name__=='__main__':unittest.main()
