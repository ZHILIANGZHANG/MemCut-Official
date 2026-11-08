#!/usr/bin/env python3
"""Qwen3.8-Flash on Bailian; same four-system efficient S012 execution."""
import os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from memcut.model_adapters.qwen import MODEL,ENDPOINT,PROFILE,install
from memcut.source_anchored_v2 import runner as core
from tools import efficient_s012 as entry

def main():
 install()
 if os.environ.get('DASHSCOPE_API_KEY'):os.environ['MEMCUT_MODEL_API_KEY']=os.environ['DASHSCOPE_API_KEY']
 old_prepare,old_load=entry.prepare,entry.load_plan
 def profile():return {'profile':PROFILE,'code_sha256':core.sha(Path(__file__).resolve().parents[1]/'memcut/model_adapters/qwen.py')}
 def prepare(args):
  if args.model!=MODEL or args.endpoint!=ENDPOINT or args.compatibility!='openai':raise ValueError('Use frozen Qwen Bailian profile')
  for value in (args.answer_model,args.selector_model):
   if value and value!=MODEL:raise ValueError('This entry requires all roles Qwen')
  plan=old_prepare(args);plan['qwen_adapter']=profile();core.write_json(Path(args.output)/'PLAN.json',plan);return plan
 def load(folder):
  plan=old_load(folder)
  if plan.get('qwen_adapter')!=profile():raise ValueError('Frozen Qwen profile differs')
  if set(plan['models'].values())!={MODEL}:raise ValueError('Qwen role mismatch')
  return plan
 entry.prepare=prepare;entry.load_plan=load
 if len(sys.argv)>1 and sys.argv[1]=='prepare':
  for flag,value in [('--model',MODEL),('--endpoint',ENDPOINT),('--compatibility','openai')]:
   if flag not in sys.argv:sys.argv.extend([flag,value])
 entry.main()
if __name__=='__main__':main()
