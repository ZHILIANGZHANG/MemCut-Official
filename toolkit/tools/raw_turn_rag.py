#!/usr/bin/env python3
"""Raw-turn RAG: local formation, S012 deletion and shared two-readout scoring."""
import argparse
import asyncio
from contextlib import redirect_stdout
import fcntl
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memcut.raw_turn_rag import runner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    doctor = sub.add_parser('doctor', help='Load the cached local encoder and check its output; no API')
    doctor.add_argument('--embedding', choices=['e5-small-v2', 'all-MiniLM-L6-v2'], default='e5-small-v2')
    prep = sub.add_parser('prepare', help='Freeze inputs and evaluation settings; no API')
    for name in ('output', 'source', 'contracts', 'annotations', 'answer-model', 'endpoint'):
        prep.add_argument('--'+name, required=True)
    prep.add_argument('--selector-model')
    prep.add_argument('--selector-endpoint')
    prep.add_argument('--selector-compatibility', choices=['openai', 'deepseek', 'mimo'])
    prep.add_argument('--compatibility', choices=['openai', 'deepseek', 'mimo'], default='openai')
    prep.add_argument('--claims', nargs='+')
    prep.add_argument('--k', nargs='+', type=int, default=[10])
    prep.add_argument('--candidate-objects', type=int, default=20)
    prep.add_argument('--candidate-turns', type=int, default=20)
    prep.add_argument('--native-eval-k', type=int, default=16)
    prep.add_argument('--embedding', choices=['e5-small-v2', 'all-MiniLM-L6-v2'], default='e5-small-v2',
                      help='retrieval encoder (configs/embeddings.json); default e5-small-v2')
    prep.add_argument('--controls', action='store_true')
    prep.add_argument('--residual-plus', action='store_true')
    prep.add_argument('--reuse-full')
    for name in ('build', 'delete', 'answer', 'report', 'trace'):
        command = sub.add_parser(name)
        command.add_argument('--run', required=True)
        if name in {'build', 'delete', 'trace'}:
            command.add_argument('--embedding-cache')
        if name in {'delete', 'answer'}:
            command.add_argument('--claims', nargs='+')
        if name == 'delete':
            command.add_argument('--strategies', nargs='+', choices=['S0', 'S1', 'S2'], default=['S0', 'S1', 'S2'],
                                 help='S0/S1 need no LLM; S2 uses the selector')
        if name == 'answer':
            command.add_argument('--conditions', nargs='+', help='Stage a subset without changing the planned denominator')
        if name == 'trace':
            command.add_argument('--snapshot')
            command.add_argument('--top-k', type=int, default=20)
            choice = command.add_mutually_exclusive_group(required=True)
            choice.add_argument('--turn-id')
            choice.add_argument('--object-id')
            choice.add_argument('--query')
    args = parser.parse_args()
    if args.command == 'doctor':
        from memcut.raw_turn_rag.embedding import LocalEmbeddings, embedding_settings
        settings = {'embedding': embedding_settings(args.embedding)}
        embedder = LocalEmbeddings(settings['embedding'], runner.ROOT/'runtime/raw-turn-rag/embeddings')
        with redirect_stdout(sys.stderr):
            vectors = embedder.encode(['Raw-turn RAG local encoder check.'])
        print(json.dumps({'system': 'raw_turn_rag', 'embedding': embedder.identity(),
                          'vector_shape': list(vectors.shape), 'api_calls': 0}))
        return
    if args.command == 'prepare':
        values = vars(args).copy()
        values.pop('command')
        folder = values.pop('output')
        values['claim_ids'] = values.pop('claims')
        values['k_list'] = values.pop('k')
        plan = runner.prepare(folder, **values)
        print(json.dumps({'plan': str(Path(folder)/'PLAN.json'), 'claims': len(plan['claims']), 'api_calls': 0}))
        return
    plan = runner.load_plan(args.run)
    if args.command == 'trace':
        from memcut.source_anchored_systems.lightmem_sources import source_links
        factory = runner.Factory(plan, cache_dir=args.embedding_cache)
        bank = factory.create(args.snapshot or Path(args.run)/'full/full.pkl')
        with redirect_stdout(sys.stderr):
            result = asyncio.run(source_links(bank, query=args.query, turn_id=args.turn_id,
                                             object_id=args.object_id, top_k=args.top_k))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    client = completions = None
    with (Path(args.run)/'RUN.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if args.command == 'answer' or (args.command == 'delete' and 'S2' in args.strategies):
                completions, client = runner.gateway(args.run, plan)
            with redirect_stdout(sys.stderr):
                if args.command == 'build':
                    runner.build(args.run, plan, factory=runner.Factory(plan, cache_dir=args.embedding_cache))
                elif args.command == 'delete':
                    asyncio.run(runner.delete(args.run, plan, completions,
                        factory=runner.Factory(plan, cache_dir=args.embedding_cache),
                        claim_ids=args.claims, strategies=args.strategies))
                elif args.command == 'answer':
                    runner.answer(args.run, plan, completions, claim_ids=args.claims, conditions=args.conditions)
                else:
                    result = runner.report(args.run, plan)
            if args.command == 'report':
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                calls = completions.calls if completions else []
                print(json.dumps({'stage': args.command, 'completed': True,
                    'cache_hits': sum(r['cached'] for r in calls),
                    'new_requests': sum(not r['cached'] for r in calls)}))
        finally:
            if client:
                client.close()


if __name__ == '__main__':
    main()
