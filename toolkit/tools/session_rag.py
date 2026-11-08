#!/usr/bin/env python3
"""Session RAG: bounded local encoding, whole-session retrieval and audited S012."""
import argparse
import asyncio
from contextlib import redirect_stdout
import fcntl
import json
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memcut.session_rag import runner
from memcut.native_memory.persistence import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    doctor=sub.add_parser('doctor',help='Verify local pinned encoder/tokenizer and coverage assets; no API')
    doctor.add_argument('--embedding',choices=['e5-small-v2', 'all-MiniLM-L6-v2'],default='e5-small-v2')
    p=sub.add_parser('prepare',help='Freeze a new independent experiment; no API')
    p.add_argument('--output',required=True)
    p.add_argument('--from-turn-run',help='Read matching completed Turn RAG PLAN inputs and model/provider roles')
    for name in ('source','contracts','annotations','answer-model','endpoint','selector-model','selector-endpoint','selector-compatibility','compatibility'):
        p.add_argument('--'+name)
    p.add_argument('--claims',nargs='+')
    p.add_argument('--k',nargs='+',type=int,default=[10])
    p.add_argument('--candidate-objects',type=int,default=20)
    p.add_argument('--candidate-turns',type=int,default=20)
    p.add_argument('--native-eval-k',type=int,default=16)
    p.add_argument('--embedding',choices=['e5-small-v2', 'all-MiniLM-L6-v2'],help='retrieval encoder (configs/embeddings.json); default e5-small-v2, or the reference Turn RAG run\'s')
    p.add_argument('--qa-max-tokens',type=int,help='Default1024; from-turn-run inherits frozen reference budget')
    p.add_argument('--always-two-trials',action='store_true')
    p.add_argument('--reuse-full')
    for name in ('build','delete','answer','coverage','audit','report','run'):
        q=sub.add_parser(name)
        q.add_argument('--run',required=True)
        if name in {'build','delete','coverage','audit','run'}:q.add_argument('--embedding-cache')
        if name in {'delete','answer'}:q.add_argument('--claims',nargs='+')
        if name=='delete':q.add_argument('--strategies',nargs='+',choices=['S0','S1','S2'],default=['S0','S1','S2'])
        if name=='answer':q.add_argument('--conditions',nargs='+')
        if name=='coverage':q.add_argument('--k-values',nargs='+',type=int,default=[1,3,16])
        if name=='audit':
            q.add_argument('--allow-partial',action='store_true',help='Audit existing local branches without claiming completion')
            q.add_argument('--local-only',action='store_true',help='No receipt verification; never produces a full PASS')
        if name=='run':q.add_argument('--accept-coverage',action='store_true',help='Explicitly accept measured >25%% context coverage for this frozen k')
    args=parser.parse_args()
    if args.command=='doctor':
        from memcut.session_rag.text import load_tokenizer,coverage_identity,coverage_tokens
        from memcut.raw_turn_rag.embedding import LocalEmbeddings, embedding_settings
        settings={'embedding':embedding_settings(args.embedding)}
        with redirect_stdout(sys.stderr):
            tok=load_tokenizer(settings['embedding']);encoder=LocalEmbeddings(settings['embedding'],runner.ROOT/'runtime/raw-turn-rag/embeddings')
            vectors=encoder.encode(['Session retrieval local verification.'])
            counted=coverage_tokens('Session retrieval local verification.')
        print(json.dumps({'api_calls':0,'vector_shape':list(vectors.shape),'runtime':runner.runtime_identity(),
            'coverage_counter':coverage_identity(),'coverage_probe_tokens':counted,'embedding':encoder.identity()},ensure_ascii=False,indent=2));return
    if args.command=='prepare':
        kw=vars(args).copy();kw.pop('command');folder=kw.pop('output');reference=kw.pop('from_turn_run')
        kw['claim_ids']=kw.pop('claims');kw['k_list']=kw.pop('k');always_two=kw.pop('always_two_trials');kw['early_stop']=not always_two
        if reference:
            ref=runner.read(Path(reference)/'PLAN.json')
            if ref['system']!='raw_turn_rag':raise ValueError('Reference must be a Turn RAG plan')
            reference_claims=[c['claim_id'] for c in ref['claims']]
            if kw['claim_ids'] is None:kw['claim_ids']=reference_claims
            elif not set(kw['claim_ids'])<=set(reference_claims):raise ValueError('Claims exceed the frozen Turn RAG reference cohort')
            if kw['qa_max_tokens'] is None:kw['qa_max_tokens']=ref.get('answer_policy',{}).get('max_tokens',400)
            kw['early_stop']=bool(ref.get('answer_execution')) and not always_two
            for key in ('source','contracts','annotations'):
                original=ref['inputs'][key]
                if runner.sha(original['path'])!=original['sha256']:raise ValueError('Reference input changed')
                if kw[key] and str(Path(kw[key]).resolve())!=str(Path(original['path']).resolve()):raise ValueError('Explicit source conflicts with reference')
                kw[key]=original['path']
            inherited={'answer_model':ref['models']['answer'],'selector_model':ref['models']['selector'],
                'endpoint':ref['providers']['answer']['endpoint'],'compatibility':ref['providers']['answer']['compatibility'],
                'selector_endpoint':ref['providers']['selector']['endpoint'],'selector_compatibility':ref['providers']['selector']['compatibility']}
            ref_embedding=ref['settings']['embedding']['name'] if isinstance(ref['settings'].get('embedding'),dict) else None
            if kw['embedding'] is not None and ref_embedding and kw['embedding']!=ref_embedding:raise ValueError('Explicit embedding conflicts with Turn RAG reference')
            kw['embedding']=kw['embedding'] or ref_embedding
            for key,value in inherited.items():
                if kw[key] is not None and kw[key]!=value:raise ValueError('Explicit model/provider conflicts with Turn RAG reference')
                kw[key]=value
        kw['embedding']=kw['embedding'] or 'e5-small-v2'
        for key in ('source','contracts','annotations','answer_model','endpoint'):
            if not kw[key]:parser.error('--'+key.replace('_','-')+' or --from-turn-run is required')
        kw['compatibility']=kw['compatibility'] or 'openai'
        if kw['qa_max_tokens'] is None:kw['qa_max_tokens']=1024
        plan=runner.prepare(folder,**kw)
        if reference:runner.sealed_write(Path(folder)/'TURN_RAG_REFERENCE.json',{'run':str(Path(reference).resolve()),'plan_sha256':runner.sha(Path(reference)/'PLAN.json'),
            'reference_claim_ids':reference_claims,'reference_answer_policy':ref.get('answer_policy',{'max_tokens':400}),
            'reference_answer_execution':ref.get('answer_execution'),'selected_qa_max_tokens':plan['qa_max_tokens'],'selected_early_stop':plan['early_stop']})
        print(json.dumps({'plan':str(Path(folder)/'PLAN.json'),'claims':len(plan['claims']),'api_calls':0}));return
    folder=Path(args.run);plan=runner.load_plan(folder);client=None
    with (folder/'RUN.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            with redirect_stdout(sys.stderr):
                factory=runner.Factory(plan,cache_dir=getattr(args,'embedding_cache',None)) if args.command in {'build','delete','coverage','audit','run'} else None
                if args.command=='build':result={'full':runner.build(folder,plan,factory=factory),'api_calls':0}
                elif args.command=='coverage':result=asyncio.run(runner.coverage(folder,plan,factory=factory,k_values=args.k_values))
                elif args.command=='audit':result=asyncio.run(runner.audit(folder,plan,factory=factory,require_complete=not args.allow_partial,verify_api=not args.local_only))
                elif args.command=='report':result=runner.report(folder,plan)
                elif args.command=='delete':
                    completions,client=runner.gateway(folder,plan) if 'S2' in args.strategies else (None,None)
                    asyncio.run(runner.delete(folder,plan,completions,factory=factory,claim_ids=args.claims,strategies=args.strategies))
                    result={'stage':'delete','completed':True}
                elif args.command=='answer':
                    completions,client=runner.gateway(folder,plan)
                    runner.answer(folder,plan,completions,claim_ids=args.claims,conditions=args.conditions)
                    result=runner.report(folder,plan)
                else:
                    runner.build(folder,plan,factory=factory)
                    asyncio.run(runner.delete(folder,plan,factory=factory,strategies=['S0','S1']))
                    measured=asyncio.run(runner.coverage(folder,plan,factory=factory,k_values=[plan['settings']['native_eval_k']]))
                    if any(g['above_25_percent'] for g in measured['summary']) and not args.accept_coverage:
                        result={'stage':'coverage_review_required','api_calls':0,'coverage':measured['summary'],
                            'message':'Actual context coverage exceeds25%; choose a new frozen k or explicitly accept this measured k before model requests.'}
                    else:
                        completions,client=runner.gateway(folder,plan)
                        asyncio.run(runner.delete(folder,plan,completions,factory=factory))
                        runner.answer(folder,plan,completions)
                        result=asyncio.run(runner.audit(folder,plan,factory=factory))
            if client:
                result['new_requests_this_process']=sum(not r['cached'] for r in client.calls)
                result['cache_hits_this_process']=sum(r['cached'] for r in client.calls)
            write_json(folder/'LAST_STAGE.json',{'command':args.command,'result':result})
            print(json.dumps(result,ensure_ascii=False,indent=2))
        except BaseException as exc:
            write_json(folder/'LAST_ERROR.json',{'command':args.command,'type':type(exc).__name__,'message':str(exc),
                'traceback':traceback.format_exc(),'not_a_score':True,'automatic_restart':False})
            raise
        finally:
            if client:client.close()


if __name__=='__main__':
    main()
