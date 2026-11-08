import json,tempfile,unittest
from pathlib import Path
from memcut.model_adapters.qwen import MODEL,ENDPOINT,normalize,install
from memcut.source_anchored_v2.completions import CachedCompletions
class QwenTest(unittest.TestCase):
 @classmethod
 def setUpClass(cls):install()
 def test_profile(self):
  q={'model':MODEL,'max_tokens':1024};n=normalize(q)
  self.assertEqual(n['extra_body'],{'enable_thinking':False,'preserve_thinking':False});self.assertNotIn('extra_body',q)
  with self.assertRaises(ValueError):normalize({'model':MODEL,'extra_body':{'enable_thinking':True}})
  other={'model':'deepseek-flash'};self.assertEqual(normalize(other),other)
if __name__=='__main__':unittest.main()
