#!/usr/bin/env python3
"""Pinned SimpleMem text: prepare/build/delete/answer/report/audit, with reuse."""
import argparse
import asyncio
from datetime import datetime, timezone
import fcntl
import inspect
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from memcut.simplemem_s012 import runner
    from memcut.simplemem_s012.upstream import doctor
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('doctor', help='Verify pinned local code, packages and weights; no API')
    prep = commands.add_parser('prepare', help='Freeze a new plan locally; no API')
    for name in ('output', 'source', 'contracts', 'annotations'):
        prep.add_argument('--'+name, required=True)
    prep.add_argument('--model', default=runner.MODEL)
    for name in ('answer-model', 'selector-model', 'retrieval-model'):
        prep.add_argument('--'+name)
    prep.add_argument('--endpoint')
    prep.add_argument('--compatibility', choices=('openai','deepseek','mimo'), default='openai')
    for role in ('answer','selector','retrieval'):
        prep.add_argument('--'+role+'-endpoint')
        prep.add_argument('--'+role+'-compatibility', choices=('openai','deepseek','mimo'))
    prep.add_argument('--claims', nargs='+')
    prep.add_argument('--reuse-full')
    prep.add_argument('--reuse-requests-from', nargs='+', default=[])
    prep.add_argument('--reuse-deletions-from')
    prep.add_argument('--build-workers', type=int, default=4)
    prep.add_argument('--answer-workers', type=int, default=4)
    prep.add_argument('--candidate-objects', type=int, default=20)
    prep.add_argument('--candidate-turns', type=int, default=20)
    prep.add_argument('--k-list', type=int, nargs='+', default=[10])
    for name in ('build','delete','answer','report','audit'):
        cmd = commands.add_parser(name)
        cmd.add_argument('--run', required=True)
        if name in ('delete','answer'):
            cmd.add_argument('--claims', nargs='+')
        if name == 'build':
            cmd.add_argument('--stop-after-windows', type=int, help='Save a prefix checkpoint; never label it Full')
        if name in ('build','delete','answer'):
            cmd.add_argument('--cached-only', action='store_true')
    args = parser.parse_args()
    if args.command == 'doctor':
        print(json.dumps(doctor(), ensure_ascii=False))
        return
    if args.command == 'prepare':
        values = vars(args).copy()
        values.pop('command')
        folder = values.pop('output')
        values['write_model'] = values.pop('model')
        values['claim_ids'] = values.pop('claims')
        values['endpoint'] = values['endpoint'] or (runner.ENDPOINT if values['write_model'] == runner.MODEL else None)
        if not values['endpoint']:
            parser.error('--endpoint is required for a non-GLM writing model')
        plan = runner.prepare(folder, **values)
        print(json.dumps({'prepared': folder, 'claims': len(plan['claims']), 'api_calls': 0}))
        return
    plan = runner.load_plan(args.run)
    folder = Path(args.run)
    with (folder/'RUN.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.command == 'report':
            print(json.dumps(runner.report(folder, plan), ensure_ascii=False))
            return
        gateway = runner.gateway(folder, plan, cached_only=args.command=='audit' or args.cached_only)
        started, completed = time.perf_counter(), False
        try:
            if args.command == 'audit':
                from memcut.simplemem_s012.audit import run
                result = asyncio.run(run(folder, plan, completions=gateway))
            else:
                kwargs = {'claim_ids': args.claims} if args.command in ('delete','answer') else {
                    'stop_after_windows': args.stop_after_windows}
                result = getattr(runner, args.command)(folder, plan, gateway, **kwargs)
                if inspect.isawaitable(result):
                    result = asyncio.run(result)
            completed = True
            print(json.dumps({'stage': args.command, 'result': result}, ensure_ascii=False))
        finally:
            gateway.close()
            runner.core.write_json(folder/'EXECUTION_USAGE.json', runner.receipts.usage(folder))
            with (folder/'EXECUTION_STAGES.jsonl').open('a') as stream:
                stream.write(json.dumps({'stage': args.command, 'complete': completed,
                    'ended_utc': datetime.now(timezone.utc).isoformat(), 'elapsed_seconds': time.perf_counter()-started,
                    'new_requests_this_process': sum(not c['cached'] for c in gateway.calls),
                    'local_reuses_this_process': sum(c['cached'] for c in gateway.calls)})+'\n')


if __name__ == '__main__':
    main()
