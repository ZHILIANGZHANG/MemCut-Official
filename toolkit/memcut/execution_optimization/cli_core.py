#!/usr/bin/env python3
"""Three-system execution reuse; native experiment conditions remain frozen."""
import argparse
import asyncio
import copy
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import fcntl
import inspect
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from memcut.execution_optimization import POLICY
from memcut.execution_optimization import receipts, scheduling, storage, branches
from memcut.model_adapters import experiment
from memcut.model_adapters.glm import MODEL, ENDPOINT, Gateway
from memcut.source_anchored_v2 import runner as core
from memcut.source_anchored_v2.memorybank import SourceMemoryBank


def code_identity():
    files = list((ROOT / 'memcut/execution_optimization').glob('*.py')) + [Path(__file__).resolve()]
    # Scheduling also reuses the already accepted bounded worker implementation.
    files.append(ROOT / 'memcut/source_anchored_systems/formation_runtime/concurrency.py')
    return {str(p.relative_to(ROOT)): core.sha(p) for p in sorted(files)}


def worker_count(value):
    value = int(value)
    if not 1 <= value <= 16:
        raise argparse.ArgumentTypeError('Use 1–16 workers')
    return value


def prepare(args):
    reuse_branches = getattr(args, 'reuse_deletions_from', None)
    if reuse_branches:
        if args.system not in {'nemori', 'lightmem'}:
            raise ValueError('Audited branch import currently supports Nemori/LightMem')
        if not args.reuse_full:
            args.reuse_full = str(Path(reuse_branches).resolve()/'full/full.pkl')
    endpoint = args.endpoint or (ENDPOINT if args.model == MODEL else None)
    if not endpoint:
        raise ValueError('--endpoint is required for non-GLM models')
    sources = list(args.reuse_requests_from or [])
    if reuse_branches:
        sources.append(reuse_branches)
    if args.reuse_full:
        sources.append(str(Path(args.reuse_full).resolve().parent.parent))
    receipt_catalog = receipts.catalog(sources)
    kwargs = dict(source=args.source, contracts=args.contracts, annotations=args.annotations,
                  write_model=args.model, answer_model=args.answer_model or args.model,
                  selector_model=args.selector_model or args.model, endpoint=endpoint,
                  compatibility=args.compatibility, claim_ids=args.claims,
                  user_id=args.user_id, reuse_full=args.reuse_full)
    uses_glm = MODEL in [kwargs[k] for k in ('write_model', 'answer_model', 'selector_model')]
    if args.system in {'nemori', 'lightmem'}:
        kwargs.update(build_workers=args.build_workers, checkpoint_turns=80)
    if uses_glm:
        if args.system == 'memorybank':
            kwargs['native_eval_k'] = 10
        plan = experiment.prepare(args.output, system=args.system, **kwargs)
    else:
        base = experiment.runner(args.system)
        if args.system in {'nemori', 'lightmem'}:
            kwargs['system'] = args.system
        plan = base.prepare(args.output, **kwargs)
        if args.system == 'memorybank':
            plan['native_qa'] = {'top_k': 10, 'method': 'native FAISS search, adjacency expansion and renderer'}
    if args.system == 'nemori' and args.reuse_full:
        # Preserve the already built Full's response-format identity. Loading
        # a frozen bank does not re-run its generation parser or issue writes.
        original = core.read(Path(args.reuse_full).resolve().parent.parent/'PLAN.json')
        original_settings = copy.deepcopy(original['settings'])
        adapter = original_settings.pop('response_schema_adapter', None)
        if original_settings != plan['settings']:
            raise ValueError('Reusable Nemori Full has different native formation settings')
        if adapter:
            if core.sha(adapter['path']) != adapter['sha256']:
                raise ValueError('Reusable Full response adapter changed')
            plan['settings']['response_schema_adapter'] = adapter
        if experiment.runner('nemori').formation_identity(original) != experiment.runner('nemori').formation_identity(plan):
            raise ValueError('Reusable Nemori Full has a different source or formation identity')
    folder = Path(args.output)
    if args.system == 'lightmem':
        # This is the already accepted formal LightMem QA budget, separately
        # frozen from execution scheduling; never silently inherit generic 400.
        plan['answer_policy'] = {'version': 'lightmem-uniform-2048-v1', 'max_tokens': 2048,
                                'all_conditions': list(plan['conditions']),
                                'all_readouts': ['native', 'residual']}
    core.sealed_write(folder / 'RECEIPT_CATALOG.json', receipt_catalog)
    if reuse_branches:
        core.sealed_write(folder / 'BRANCH_CATALOG.json', branches.prepare_catalog(reuse_branches, plan))
    plan['execution_optimization'] = {
        'policy': POLICY, 'system': args.system, 'build_workers': args.build_workers,
        'answer_workers': args.answer_workers, 'code': code_identity(),
        'receipt_catalog_sha256': core.sha(folder / 'RECEIPT_CATALOG.json'),
        'local_storage_policy': storage.POLICY,
        'request_changes': False, 'paid_retries': 0, 'storage_commit_order': 'original'}
    if reuse_branches:
        plan['execution_optimization']['branch_catalog_sha256'] = core.sha(folder/'BRANCH_CATALOG.json')
    # New plan only, before any formation or dispatch; historical plans unchanged.
    core.write_json(folder / 'PLAN.json', plan)
    return plan


def load_plan(folder):
    raw = core.read(Path(folder) / 'PLAN.json')
    policy = raw.get('execution_optimization', {})
    if policy.get('policy') != POLICY or policy.get('system') not in {'memorybank', 'nemori', 'lightmem'}:
        raise ValueError('Use prepare to freeze a new optimized run; historical runs are not rewritten')
    base = experiment.runner(policy['system'])
    plan = experiment.load_plan(folder) if 'model_adapter' in raw else base.load_plan(folder)
    if policy['code'] != code_identity() or policy['receipt_catalog_sha256'] != core.sha(Path(folder) / 'RECEIPT_CATALOG.json'):
        raise ValueError('Frozen execution code or receipt catalog changed')
    for key in ('build_workers', 'answer_workers'):
        if type(policy[key]) is not int or not 1 <= policy[key] <= 16:
            raise ValueError('Invalid frozen concurrency')
    if policy['request_changes'] or policy['paid_retries'] != 0 or policy['storage_commit_order'] != 'original':
        raise ValueError('Unsupported execution semantics')
    if policy.get('local_storage_policy') != storage.POLICY:
        raise ValueError('Unknown frozen local storage execution policy')
    if policy.get('branch_catalog_sha256'):
        if policy['system'] not in {'nemori', 'lightmem'} or policy['branch_catalog_sha256'] != core.sha(Path(folder)/'BRANCH_CATALOG.json'):
            raise ValueError('Frozen branch catalog changed or has an unsupported system')
    if policy['system'] == 'memorybank' and plan.get('native_qa') != {
            'top_k': 10, 'method': 'native FAISS search, adjacency expansion and renderer'}:
        raise ValueError('MemoryBank optimized plans require the accepted native QA k10')
    adapter = plan.get('settings', {}).get('response_schema_adapter')
    if adapter and core.sha(adapter['path']) != adapter['sha256']:
        raise ValueError('Frozen Full response adapter changed')
    if policy['system'] == 'lightmem' and plan.get('answer_policy') != {
            'version': 'lightmem-uniform-2048-v1', 'max_tokens': 2048,
            'all_conditions': plan['conditions'], 'all_readouts': ['native', 'residual']}:
        raise ValueError('LightMem requires its accepted uniform 2048-token QA budget')
    return plan


@contextmanager
def runtime(plan, *, native=True):
    system = plan['execution_optimization']['system']
    with ExitStack() as stack:
        if system == 'memorybank':
            stack.enter_context(patch.object(SourceMemoryBank, 'native_answer_top_k', plan['native_qa']['top_k']))
        elif native:
            stack.enter_context(storage.runtime(system))
        if system == 'lightmem':
            from memcut.source_anchored_v2 import evaluation
            original = evaluation.answer_request
            def answer_request(*args, **kwargs):
                return {**original(*args, **kwargs), 'max_tokens': plan['answer_policy']['max_tokens']}
            if plan.get('retention') or plan.get('residual_plus'):
                raise ValueError('Optimized LightMem entry freezes the accepted two-readout experiment')
            stack.enter_context(patch.object(evaluation, 'answer_request', answer_request))
        yield


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare', help='Freeze a new plan and explicit receipt sources; no API')
    prep.add_argument('--system', choices=['memorybank', 'nemori', 'lightmem'], required=True)
    for name in ('output', 'source', 'contracts', 'annotations'):
        prep.add_argument('--' + name, required=True)
    prep.add_argument('--model', default=MODEL, help='Writing model; default GLM-5.3-Flash')
    prep.add_argument('--answer-model')
    prep.add_argument('--selector-model')
    prep.add_argument('--endpoint')
    prep.add_argument('--compatibility', choices=['openai', 'deepseek', 'mimo'], default='openai')
    prep.add_argument('--user-id')
    prep.add_argument('--claims', nargs='+')
    prep.add_argument('--reuse-full')
    prep.add_argument('--reuse-requests-from', nargs='+', help='Explicit old run directories; complete exact requests only')
    prep.add_argument('--reuse-deletions-from', help='Nemori/LightMem: reuse audited states and contexts from an identical deletion experiment')
    prep.add_argument('--build-workers', type=worker_count, default=4)
    prep.add_argument('--answer-workers', type=worker_count, default=4)
    for name in ('build', 'delete', 'answer', 'report'):
        cmd = commands.add_parser(name)
        cmd.add_argument('--run', required=True)
        if name in {'delete', 'answer'}:
            cmd.add_argument('--claims', nargs='+')
        if name != 'report':
            cmd.add_argument('--cached-only', action='store_true', help='Fail on a cache miss; never dispatch API')
    args = parser.parse_args()
    if args.command == 'prepare':
        plan = prepare(args)
        print(json.dumps({'prepared': args.output, 'claims': len(plan['claims']), 'api_calls': 0}))
        return
    plan = load_plan(args.run)
    policy = plan['execution_optimization']
    base = experiment.runner(policy['system'])
    folder = Path(args.run)
    reuse_branches = args.command == 'delete' and bool(policy.get('branch_catalog_sha256'))
    with (folder / 'RUN.lock').open('a+b') as lock, runtime(plan, native=(args.command in {'build', 'delete'} and not reuse_branches)):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.command == 'report':
            print(json.dumps(base.report(folder, plan), ensure_ascii=False))
            core.write_json(folder / 'EXECUTION_USAGE.json', receipts.usage(folder))
            return
        delegate = Gateway(folder, plan) if 'model_adapter' in plan else core.gateway(folder, plan)[0]
        gateway = receipts.ReuseGateway(delegate, core.read(folder / 'RECEIPT_CATALOG.json'),
                                        cached_only=args.cached_only)
        started = time.perf_counter()
        completed = False
        try:
            if args.command == 'build' and policy['system'] == 'memorybank':
                asyncio.run(scheduling.build_memorybank(folder, plan, gateway, workers=policy['build_workers']))
            elif reuse_branches:
                imported = branches.import_claims(folder, plan, core.read(folder/'BRANCH_CATALOG.json'), claim_ids=args.claims)
                core.write_json(folder/'PROGRESS.json', {'stage': 'branches_imported', **imported})
            elif args.command == 'answer':
                scheduling.answer(folder, plan, gateway, workers=policy['answer_workers'],
                                  claim_ids=args.claims, source_import=base.import_source)
            else:
                kwargs = {'claim_ids': args.claims} if args.command == 'delete' else {}
                result = getattr(base, args.command)(folder, plan, gateway, **kwargs)
                if inspect.isawaitable(result):
                    asyncio.run(result)
            completed = True
        finally:
            gateway.close()
            core.write_json(folder / 'EXECUTION_USAGE.json', receipts.usage(folder))
            # Durable per-invocation time includes resumptions separately and
            # never confuses a cached replay with a newly paid full run.
            with (folder / 'EXECUTION_STAGES.jsonl').open('a') as stream:
                stream.write(json.dumps({'stage': args.command, 'complete': completed,
                    'ended_utc': datetime.now(timezone.utc).isoformat(),
                    'elapsed_seconds': time.perf_counter()-started, 'cached_only': args.cached_only,
                    'new_requests_this_process': sum(not c['cached'] for c in gateway.calls),
                    'local_reuses_this_process': sum(c['cached'] for c in gateway.calls)})+'\n')
        print(json.dumps({'stage': args.command, 'completed': True,
                          'new_requests_this_process': sum(not c['cached'] for c in gateway.calls),
                          'usage': receipts.usage(folder)}))


if __name__ == '__main__':
    main()
