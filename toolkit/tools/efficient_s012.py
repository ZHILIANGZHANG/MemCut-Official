#!/usr/bin/env python3
"""Four-system native experiments: exact reuse, concurrent QA, valid-miss early stop."""
import argparse
import asyncio
import copy
from contextlib import ExitStack, contextmanager
import fcntl
import inspect
import json
from pathlib import Path
import shutil
import sys
import time
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from memcut.execution_optimization import cli_core as legacy
from memcut.answer_execution import early_stop
from memcut.efficient_execution import POLICY, branches, replay, recovery
from memcut.execution_optimization import receipts, scheduling
from memcut.model_adapters import experiment
from memcut.model_adapters.glm import MODEL, ENDPOINT, Gateway
from memcut.source_anchored_v2 import runner as core, evaluation
from memcut.native_memory.contract import digest
SYSTEMS=('raw_turn_rag','memorybank','nemori','lightmem')


def code_identity():
    files=list((ROOT/'memcut/efficient_execution').glob('*.py'))+[Path(__file__).resolve()]
    files += [ROOT/p for p in legacy.code_identity()]
    files += list((ROOT/'memcut/answer_execution').glob('*.py'))
    return {str(p.relative_to(ROOT)):core.sha(p) for p in sorted(set(files))}


def prepare(args):
    folder=Path(args.output)
    if folder.exists():raise FileExistsError('Use a new run directory')
    if args.reuse_deletions_from:
        source=Path(args.reuse_deletions_from).resolve();old=branches._source_plan(source)
        if branches.system_of(old)!=args.system:raise ValueError('Source system differs')
        if 'write' in old['models'] and old['models']['write']!=args.model:raise ValueError('Writing model differs')
        if args.selector_model and args.selector_model!=old['models']['selector']:raise ValueError('Selector model differs')
        if args.reuse_full and Path(args.reuse_full).resolve()!=source/'full/full.pkl':raise ValueError('Different Full requested')
        for name in ('source','contracts','annotations'):
            given=getattr(args,name)
            if given and (str(Path(given).resolve())!=old['inputs'][name]['path'] or core.sha(given)!=old['inputs'][name]['sha256']):
                raise ValueError('Source inputs differ')
        plan=copy.deepcopy(old)
        if args.claims:
            if len(args.claims)!=len(set(args.claims)) or not set(args.claims)<={c['claim_id'] for c in old['claims']}:raise ValueError('Unknown/duplicate claims')
            plan['claims']=[c for c in old['claims'] if c['claim_id'] in args.claims]
        plan['reuse_full']={'path':str(source/'full/full.pkl'),'sha256':core.sha(source/'full/full.pkl')}
        if args.answer_model:plan['models']['answer']=args.answer_model
        if args.endpoint:
            # In reuse mode endpoint changes apply to answering only.
            plan['providers']['answer']={**plan['providers']['answer'],'endpoint':args.endpoint,'compatibility':args.compatibility}
        if MODEL in plan['models'].values():plan['model_adapter']=experiment.identity(args.system,plan)
        else:plan.pop('model_adapter',None)
        folder.mkdir(parents=True)
        protocol=source/'PROTOCOL.md' if (source/'PROTOCOL.md').exists() else ROOT/'PROTOCOL.md'
        if core.sha(protocol)!=old['protocol_document_sha256']:raise ValueError('Source protocol document changed')
        shutil.copyfile(protocol,folder/'PROTOCOL.md')
        catalog=receipts.catalog([str(source),*(args.reuse_requests_from or [])])
    else:
        if not all(getattr(args,k) for k in ('source','contracts','annotations')):raise ValueError('Fresh runs require all source input paths')
        if args.system=='raw_turn_rag':
            kwargs=dict(source=args.source,contracts=args.contracts,annotations=args.annotations,
                        answer_model=args.answer_model or args.model,selector_model=args.selector_model or args.model,
                        endpoint=args.endpoint or ENDPOINT,compatibility=args.compatibility,claim_ids=args.claims,reuse_full=args.reuse_full)
            plan=(experiment.prepare(args.output,system=args.system,**kwargs) if MODEL in [kwargs['answer_model'],kwargs['selector_model']] else experiment.runner(args.system).prepare(args.output,**kwargs))
            sources=list(args.reuse_requests_from or [])
            if args.reuse_full:sources.append(str(Path(args.reuse_full).resolve().parent.parent))
            catalog=receipts.catalog(sources)
        else:
            plan=legacy.prepare(args)
            catalog=core.read(folder/'RECEIPT_CATALOG.json')
    plan.pop('execution_optimization',None)
    plan['answer_execution']=copy.deepcopy(early_stop.POLICY)
    plan['answer_policy']={'version':'four-system-qa1024-v1','max_tokens':1024,'all_conditions':plan['conditions'],'all_readouts':['native','residual']}
    core.sealed_write(folder/'RECEIPT_CATALOG.json',catalog)
    plan['efficient_execution']={'policy':POLICY,'system':args.system,'code':code_identity(),
        'answer_workers':args.answer_workers,'build_workers':args.build_workers,'retry_policy':recovery.POLICY,
        'receipt_catalog_sha256':core.sha(folder/'RECEIPT_CATALOG.json')}
    if args.reuse_deletions_from:
        catalog=branches.prepare_catalog(args.reuse_deletions_from,plan)
        core.sealed_write(folder/'BRANCH_CATALOG.json',catalog)
        plan['efficient_execution']['branch_catalog_sha256']=core.sha(folder/'BRANCH_CATALOG.json')
        core.sealed_write(folder/'ANSWER_ALIAS_CATALOG.json',replay.answer_aliases(args.reuse_deletions_from,plan))
        plan['efficient_execution']['answer_alias_catalog_sha256']=core.sha(folder/'ANSWER_ALIAS_CATALOG.json')
    core.write_json(folder/'PLAN.json',plan)
    return plan


def load_plan(folder):
    folder=Path(folder);raw=core.read(folder/'PLAN.json');p=raw.get('efficient_execution',{})
    if p.get('policy')!=POLICY or p.get('system') not in SYSTEMS:raise ValueError('Use the four-system prepare entry')
    plan=experiment.load_plan(folder) if 'model_adapter' in raw else experiment.runner(p['system']).load_plan(folder)
    if p['code']!=code_identity() or p['receipt_catalog_sha256']!=core.sha(folder/'RECEIPT_CATALOG.json'):raise ValueError('Frozen execution implementation or receipt catalog changed')
    if plan['answer_execution']!=early_stop.POLICY or plan['answer_policy']!={'version':'four-system-qa1024-v1','max_tokens':1024,'all_conditions':plan['conditions'],'all_readouts':['native','residual']} or p['retry_policy']!=recovery.POLICY:raise ValueError('Frozen answer policy changed')
    if plan.get('controls') or plan.get('retention') or plan.get('residual_plus'):raise ValueError('Main two-readout experiment only')
    for k in ('answer_workers','build_workers'):
        if type(p[k])is not int or not 1<=p[k]<=16:raise ValueError('Invalid concurrency')
    if p.get('answer_alias_catalog_sha256') and p['answer_alias_catalog_sha256']!=core.sha(folder/'ANSWER_ALIAS_CATALOG.json'):raise ValueError('Answer alias catalog changed')
    if p.get('branch_catalog_sha256') and p['branch_catalog_sha256']!=core.sha(folder/'BRANCH_CATALOG.json'):raise ValueError('Branch catalog changed')
    if p['system']=='memorybank' and plan['native_qa']['top_k']!=10:raise ValueError('Require MemoryBank k10')
    return plan


@contextmanager
def runtime(plan,native=False):
    system=plan['efficient_execution']['system'];original=evaluation.answer_request
    with ExitStack() as stack:
        if system=='memorybank':stack.enter_context(patch.object(legacy.SourceMemoryBank,'native_answer_top_k',10))
        elif native and system in {'nemori','lightmem'}:stack.enter_context(legacy.storage.runtime(system))
        stack.enter_context(patch.object(evaluation,'answer_request',lambda *a,**k:{**original(*a,**k),'max_tokens':1024}))
        stack.enter_context(patch.object(scheduling,'evaluate',early_stop.evaluate))
        yield


def report(folder,plan):
    rows=[];folder=Path(folder)
    for claim in plan['claims']:
        for condition in plan['conditions']:
            state=core.validate_state(folder,plan,claim,condition)
            for family,ctx in state['contexts'].items():
                path=folder/'claims'/claim['claim_id']/condition/(family+'.json')
                if not path.exists():continue
                row=core.read(path);r=row['result']
                if row['plan_sha256']!=digest(plan) or (row['claim_id'],row['condition'],row['family'])!=(claim['claim_id'],condition,family) or r['context_sha256']!=ctx['context_sha256'] or r['state_identity']!=ctx['state_identity']:raise ValueError('Answer/state identity differs')
                if bool(r.get('evidence_derived_empty'))!=ctx['empty']:raise ValueError('Invalid structural-empty shortcut')
                early_stop.validate_result(r,claim,family)
                for a,rotation in zip(r['answers'],claim['rotations']):
                    actual,response,_=replay.receipt(folder,a['request_id'])
                    expected=replay.identity(plan,'answer',evaluation.answer_request(claim,ctx,rotation,plan['models']['answer']))
                    if replay.semantic_identity(actual)!=expected or evaluation.score_response(response,rotation,claim['target_choice_id'],claim['choices'])!=a:raise ValueError('Answer request or raw score differs')
                    recovery.verify_received_attempt(folder,expected,actual)
                rows.append(row)
    with patch.object(core,'summarize',early_stop.summarize):
        summary=experiment.runner(plan['efficient_execution']['system']).report(folder,plan)
    summary['main_complete']=all(v['completed_claims']==len(plan['claims']) for v in summary['metrics'].values())
    core.write_json(folder/'SUMMARY.json',summary)
    return summary


def import_full(folder,plan,catalog):
    source=Path(catalog['source']);folder=Path(folder)
    for rel,h in catalog['files'].items():
        if core.sha(source/rel)!=h:raise ValueError('Source assets changed after prepare')
    for name in ('full.pkl','full.frozen.json'):
        target=folder/'full'/name;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            if core.sha(target)!=catalog['files']['full/'+name]:raise ValueError('Destination Full differs')
        else:shutil.copyfile(source/'full'/name,target)
    core.sealed_write(folder/'full/IDENTITY.json',{'plan_sha256':digest(plan),'state_identity':catalog['source_full']['state_identity']})
    # Preserve source extension proofs verbatim, separately from new execution.
    for rel in catalog['files']:
        if len(Path(rel).parts)==1 or (rel.startswith('full/') and rel.endswith('.json')):
            target=folder/'inherited_provenance'/rel;target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                if core.sha(target)!=catalog['files'][rel]:raise ValueError('Inherited proof changed')
            else:shutil.copyfile(source/rel,target)


def main():
    parser=argparse.ArgumentParser(description=__doc__);subs=parser.add_subparsers(dest='command',required=True)
    p=subs.add_parser('prepare');p.add_argument('--system',choices=SYSTEMS,required=True);p.add_argument('--output',required=True)
    for name in ('source','contracts','annotations','user-id','reuse-full','reuse-deletions-from','answer-model','selector-model','endpoint'):p.add_argument('--'+name)
    p.add_argument('--model',default=MODEL);p.add_argument('--compatibility',choices=['openai','deepseek','mimo'],default='openai')
    p.add_argument('--claims',nargs='+');p.add_argument('--reuse-requests-from',nargs='+')
    for name in ('build-workers','answer-workers'):p.add_argument('--'+name,type=legacy.worker_count,default=4)
    for name in ('build','delete','answer','report'):
        p=subs.add_parser(name);p.add_argument('--run',required=True)
        if name in ('delete','answer'):p.add_argument('--claims',nargs='+')
        if name!='report':p.add_argument('--cached-only',action='store_true')
    args=parser.parse_args()
    if args.command=='prepare':
        plan=prepare(args);print(json.dumps({'prepared':args.output,'claims':len(plan['claims']),'api_calls':0}));return
    folder=Path(args.run);plan=load_plan(folder);policy=plan['efficient_execution'];base=experiment.runner(policy['system'])
    reused=bool(policy.get('branch_catalog_sha256'));started=time.perf_counter();complete=False
    with (folder/'RUN.lock').open('a+b') as lock,runtime(plan,native=args.command in ('build','delete') and not reused):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.command=='report':print(json.dumps(report(folder,plan),ensure_ascii=False));return
        delegate=Gateway(folder,plan) if 'model_adapter' in plan else core.gateway(folder,plan)[0]
        gateway=receipts.ReuseGateway(delegate,core.read(folder/'RECEIPT_CATALOG.json'),cached_only=args.cached_only)
        if reused:gateway=replay.AliasGateway(gateway,core.read(folder/'ANSWER_ALIAS_CATALOG.json'))
        gateway=recovery.RecoveryGateway(gateway)
        try:
            if reused and args.command=='build':import_full(folder,plan,core.read(folder/'BRANCH_CATALOG.json'))
            elif reused and args.command=='delete':branches.import_claims(folder,plan,core.read(folder/'BRANCH_CATALOG.json'),claim_ids=args.claims)
            elif args.command=='answer':scheduling.answer(folder,plan,gateway,workers=policy['answer_workers'],claim_ids=args.claims)
            elif args.command=='build' and policy['system']=='raw_turn_rag':base.build(folder,plan)
            elif args.command=='build' and policy['system']=='memorybank':asyncio.run(scheduling.build_memorybank(folder,plan,gateway,workers=policy['build_workers']))
            else:
                result=getattr(base,args.command)(folder,plan,gateway,**({'claim_ids':args.claims} if args.command=='delete' else {}))
                if inspect.isawaitable(result):asyncio.run(result)
            complete=True
        finally:
            gateway.close();core.write_json(folder/'EXECUTION_USAGE.json',receipts.usage(folder))
            with (folder/'EXECUTION_STAGES.jsonl').open('a') as f:f.write(json.dumps({'stage':args.command,'complete':complete,'seconds':time.perf_counter()-started,'cached_only':args.cached_only,'new_requests':sum(not c['cached'] for c in gateway.calls)})+'\n')
        print(json.dumps({'stage':args.command,'complete':complete,'new_requests':sum(not c['cached'] for c in gateway.calls)}))


if __name__=='__main__':main()
