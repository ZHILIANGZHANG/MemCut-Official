#!/usr/bin/env python3
"""GLM-5.3-Flash low-reasoning entrypoint for the existing S012 native runners."""
import argparse
import asyncio
import fcntl
import getpass
import inspect
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memcut.model_adapters import experiment
from memcut.model_adapters.glm import Gateway


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare', help='Freeze a new all-GLM plan locally; no API')
    prep.add_argument('--system', choices=list(experiment.MODULES), required=True)
    for name in ('output', 'source', 'contracts', 'annotations'):
        prep.add_argument('--'+name, required=True)
    prep.add_argument('--claims', nargs='+')
    prep.add_argument('--reuse-full', help='Same-policy GLM Full, or model-independent raw RAG Full')
    prep.add_argument('--user-id')
    prep.add_argument('--native-eval-k', type=int, help='MemoryBank QA seed k; default 10, independent of deletion top20')
    prep.add_argument('--build-workers', type=int, default=4)
    prep.add_argument('--checkpoint-turns', type=int, default=80)
    for name in ('build', 'delete', 'answer', 'report'):
        cmd = sub.add_parser(name)
        cmd.add_argument('--run', required=True)
        if name in {'delete', 'answer'}:
            cmd.add_argument('--claims', nargs='+')
    args = parser.parse_args()
    if args.command == 'prepare':
        kwargs = {name: getattr(args, name) for name in ('source', 'contracts', 'annotations', 'reuse_full')}
        kwargs['claim_ids'] = args.claims
        if args.system != 'raw_turn_rag':
            kwargs['user_id'] = args.user_id
        if args.native_eval_k is not None:
            if args.system != 'memorybank':
                parser.error('--native-eval-k here is for MemoryBank only')
            kwargs['native_eval_k'] = args.native_eval_k
        if args.system in {'nemori', 'lightmem'}:
            kwargs.update(build_workers=args.build_workers, checkpoint_turns=args.checkpoint_turns)
        plan = experiment.prepare(args.output, system=args.system, **kwargs)
        print(json.dumps({'prepared': args.output, 'claims': len(plan['claims']), 'models': plan['models'],
                          'api_calls': 0}, ensure_ascii=False))
        return
    plan = experiment.load_plan(args.run)
    system = plan['model_adapter']['system']
    runner = experiment.runner(system)
    if args.command == 'report':
        print(json.dumps(runner.report(args.run, plan), ensure_ascii=False, indent=2))
        experiment.usage_report(args.run)
        return
    with (Path(args.run)/'RUN.lock').open('a+b') as lock, experiment.runtime(plan):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        gateway = Gateway(args.run, plan)
        # Lazy secret prompt: offline builds and exact cache replays need no key.
        for role in plan['model_adapter']['roles']:
            cache = gateway.for_role(role)
            guard = cache.before_dispatch
            def ensure_secret(guard=guard):
                with gateway.lock:
                    if not os.environ.get('MEMCUT_GLM_API_KEY'):
                        if not sys.stdin.isatty():
                            raise ValueError('Set MEMCUT_GLM_API_KEY for noninteractive execution')
                        os.environ['MEMCUT_GLM_API_KEY'] = getpass.getpass('GLM API key (hidden, not saved): ').strip()
                guard()
            cache.before_dispatch = ensure_secret
        try:
            kwargs = {'claim_ids': args.claims} if args.command in {'delete', 'answer'} else {}
            if system == 'raw_turn_rag' and args.command == 'build':
                runner.build(args.run, plan)
            else:
                result = getattr(runner, args.command)(args.run, plan, gateway, **kwargs)
                if inspect.isawaitable(result):
                    asyncio.run(result)
        finally:
            gateway.close()
            experiment.usage_report(args.run)
        print(json.dumps({'stage': args.command, 'completed': True,
            'new_requests': sum(not c['cached'] for c in gateway.calls),
            'local_reuses': sum(c['cached'] for c in gateway.calls)}))


if __name__ == '__main__':
    main()
