#!/usr/bin/env python3
"""MemoryBank turn-budget protocol: prepare/build/delete/answer/retention/report."""
import argparse
import asyncio
import fcntl
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memcut.source_anchored_v2 import runner


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare', help='Local only: freeze sources, models and protocol')
    for name in ('output', 'source', 'contracts', 'annotations', 'write-model', 'answer-model', 'endpoint'):
        prep.add_argument('--'+name, required=True)
    prep.add_argument('--selector-model')
    prep.add_argument('--answer-endpoint')
    prep.add_argument('--answer-compatibility', choices=['openai', 'deepseek', 'mimo'])
    prep.add_argument('--selector-endpoint')
    prep.add_argument('--selector-compatibility', choices=['openai', 'deepseek', 'mimo'])
    prep.add_argument('--user-id')
    prep.add_argument('--compatibility', choices=['openai', 'deepseek', 'mimo'], default='openai')
    prep.add_argument('--claims', nargs='+')
    prep.add_argument('--k', type=int, nargs='+', default=[10])
    prep.add_argument('--residual-plus', action='store_true')
    prep.add_argument('--controls', action='store_true')
    prep.add_argument('--retention')
    prep.add_argument('--reuse-full', help='A compatible frozen S012 Full .pkl; never an old object-budget bank')
    for name in ('build', 'delete', 'answer', 'retention', 'report'):
        cmd = sub.add_parser(name)
        cmd.add_argument('--run', required=True)
        if name in {'delete', 'answer'}:
            cmd.add_argument('--claims', nargs='+', help='Execute part of the frozen cohort without changing its denominator')
    args = p.parse_args()
    if args.command == 'prepare':
        plan = runner.prepare(args.output, source=args.source, contracts=args.contracts, annotations=args.annotations,
            write_model=args.write_model, answer_model=args.answer_model,
            selector_model=args.selector_model or args.write_model, endpoint=args.endpoint,
            compatibility=args.compatibility, claim_ids=args.claims, k_list=args.k, user_id=args.user_id,
            residual_plus=args.residual_plus, controls=args.controls, retention=args.retention, reuse_full=args.reuse_full,
            answer_endpoint=args.answer_endpoint, answer_compatibility=args.answer_compatibility,
            selector_endpoint=args.selector_endpoint, selector_compatibility=args.selector_compatibility)
        print(json.dumps({'prepared': args.output, 'claims': len(plan['claims']), 'conditions': plan['conditions']}, ensure_ascii=False))
        return
    plan = runner.load_plan(args.run)
    if args.command == 'report':
        result = runner.report(args.run, plan)
        print(json.dumps(result['metrics'], ensure_ascii=False, indent=2))
        return
    with (Path(args.run) / 'RUN.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.command == 'build' and plan.get('reuse_full'):
            from memcut.source_anchored_v2.completions import CachedCompletions
            def no_request(**kwargs):
                raise RuntimeError('Full reuse must not call a model')
            completions = CachedCompletions(Path(args.run) / 'requests', no_request,
                                            endpoint=plan['endpoint'], compatibility=plan['compatibility'])
            client = None
        else:
            completions, client = runner.gateway(args.run, plan)
        try:
            fn = getattr(runner, args.command)
            kwargs = {'claim_ids': args.claims} if args.command in {'delete', 'answer'} else {}
            if args.command in {'build', 'delete', 'retention'}:
                asyncio.run(fn(args.run, plan, completions, **kwargs))
            else:
                fn(args.run, plan, completions, **kwargs)
        finally:
            if client is not None:
                client.close()
    print(json.dumps({'stage': args.command, 'completed': True, 'cache_hits': sum(r['cached'] for r in completions.calls),
                      'new_requests': sum(not r['cached'] for r in completions.calls)}))


if __name__ == '__main__':
    main()
