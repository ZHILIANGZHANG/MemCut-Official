#!/usr/bin/env python3
"""MemCut S0/S1/S2 for four memory systems (native mechanisms aligned with LoCoMo) and the two retrieval
controls, Raw-turn and Session RAG: resumable build → delete → answer → report, one failure policy."""
import argparse
import asyncio
from contextlib import ExitStack
import fcntl
import inspect
import json
import os
from pathlib import Path
import pickle
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from memcut.aligned_s012 import RAG, SYSTEMS  # noqa: E402
from memcut.native_memory.contract import digest  # noqa: E402
from memcut.aligned_s012.runtime import ANSWER_POLICY, identity, runtime, trial_names  # noqa: E402
from memcut.aligned_s012 import transport  # noqa: E402
from memcut.model_adapters import local, qwen  # noqa: E402
from memcut.source_anchored_v2 import runner as core  # noqa: E402


def paid_lock_path():
    # MEMCUT_PAID_LOCK: the same file as the local LoCoMo runner's --paid-lock, so the two never pay at once.
    return Path(os.environ.get('MEMCUT_PAID_LOCK') or ROOT / 'outputs/MEMCUT_PAID.lock')


def prepare(args):
    folder = Path(args.output)
    if folder.exists():
        raise FileExistsError('Use a new run directory; historical plans are immutable')
    native_eval_k = getattr(args, 'native_eval_k', None)
    accept_coverage = bool(getattr(args, 'accept_coverage', False))
    embedding = getattr(args, 'embedding', None)
    if (native_eval_k is not None or embedding is not None) and args.system not in RAG:
        raise ValueError('--native-eval-k and --embedding are retrieval control settings (turnrag, sessionrag)')
    if accept_coverage and args.system != 'sessionrag':
        raise ValueError('--accept-coverage applies to sessionrag only')
    # Prepare next to the target and rename at the end: an interrupted prepare leaves no half plan behind.
    staging = folder.with_name(f'.{folder.name}.preparing-{os.getpid()}')
    shutil.rmtree(staging, ignore_errors=True)
    normalizations = []
    try:
        with runtime(args.system) as base, trial_names(args.system, normalizations):
            options = dict(source=args.source, contracts=args.contracts, annotations=args.annotations,
                answer_model=args.model, selector_model=args.model, endpoint=args.endpoint, claim_ids=args.claims,
                compatibility=getattr(args, 'compatibility', 'openai'))
            if args.system in RAG:              # formation is local: no write model
                options.update(native_eval_k=16 if native_eval_k is None else native_eval_k,
                               embedding=embedding or 'e5-small-v2')
                if args.system == 'sessionrag':  # the uniform cap and two fixed trials, as for the other rows
                    options.update(qa_max_tokens=ANSWER_POLICY['max_tokens'], early_stop=False)
            else:
                options.update(write_model=args.model)
            if args.system in ('nemori', 'lightmem'):
                options.update(system=args.system, build_workers=1, checkpoint_turns=20)
            plan = base.prepare(staging, **options)
        plan['alignment'] = identity(args.system)
        plan['aligned_answer_policy'] = ANSWER_POLICY
        plan['input_normalization'] = {'policy': 'exact-trial-name-to-integer-v1', 'changes': normalizations,
                                       'source_files_modified': False, 'choice_mapping_changes': False}
        if args.system == 'memorybank':
            from memcut.answer_execution.early_stop import POLICY
            plan['native_qa'] = {'top_k': 10}
            plan['answer_execution'] = POLICY
        if args.system == 'sessionrag':
            plan['aligned_options'] = {'accept_coverage': accept_coverage}
        core.write_json(staging / 'PLAN.json', plan)
        if (staging / 'PLAN_IDENTITY.json').exists():     # Session RAG binds its plan's digest
            core.write_json(staging / 'PLAN_IDENTITY.json', {'plan_sha256': digest(plan)})
        os.rename(staging, folder)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return plan


def unresolved(folder):
    # Cheap receipt presence check; no response re-scoring or prefix hash audit.
    return [p.parent.name for p in (folder / 'requests').glob('*/request.json')
            if not p.with_name('response.json').exists()]


def write_diagnostics(folder, plan):
    """What Full encoded: source turns no live memory derives from (a turn upstream dropped or chose not to
    store), the claims whose evidence or disclosure is among them, and the bank's own counters."""
    full = folder / 'full/full.pkl'
    if plan['alignment']['system'] in ('memorybank',) + RAG or not full.exists():
        return None                       # MemoryBank's bank has its own format; the RAG controls store every turn
    try:
        value = pickle.loads(full.read_bytes())
        der, order = value['lineage']['der'], list(value['turns'])
    except Exception as exc:  # noqa: BLE001 - diagnostics never stop a run
        result = {'error': f'{type(exc).__name__}: {exc}'[:300]}
    else:
        empty = [t for t in order if not der.get(t)]
        missing = set(empty)
        result = {'turns': len(order), 'turns_without_memory': len(empty), 'first_turns_without_memory': empty[:50],
                  'claims_with_unencoded_turns': {
                      c['claim_id']: [t for t in c['evidence_turns'] + c['disclosure_turns'] if t in missing]
                      for c in plan['claims'] if missing & set(c['evidence_turns'] + c['disclosure_turns'])},
                  'system': value.get('progress', {}).get('diagnostics', {})}
    core.write_json(folder / 'full/DIAGNOSTICS.json', result)
    return result


def execute(folder, command='run'):
    folder = Path(folder)
    raw = core.read(folder / 'PLAN.json')
    extension = raw.get('alignment', {})
    system = extension.get('system')
    if system not in SYSTEMS or extension != identity(system):
        raise ValueError('Use a matching versioned memcut_s012 plan')
    if raw.get('aligned_answer_policy') != ANSWER_POLICY:
        raise ValueError('Aligned answer policy changed; prepare a new run')
    stages = ('build', 'delete', 'answer', 'report') if command == 'run' else (command,)
    with ExitStack() as stack:
        lock = stack.enter_context((folder / 'RUN.lock').open('a+b'))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = folder / 'STATUS.json'
        if (command == 'run' and status.exists()
                and core.read(status) == {'stage': 'run', 'state': 'complete'}
                and (folder / 'SUMMARY.json').exists()):
            return core.read(folder / 'SUMMARY.json')
        if command != 'report':
            shared = paid_lock_path()
            shared.parent.mkdir(parents=True, exist_ok=True)
            paid_lock = stack.enter_context(shared.open('a+b'))
            fcntl.flock(paid_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            transport.archive_not_billed(folder)      # a provider error billed nothing: send it again
            if unresolved(folder):
                raise ValueError('Unresolved dispatch retained (sent, outcome unknown); list them with '
                                 f'`python run.py aligned resolve --run {folder}` and add --retry to send them '
                                 'again; no automatic resend')
        base = stack.enter_context(runtime(system, native=any(s in ('build', 'delete') for s in stages)))
        plan = base.load_plan(folder)
        if system == 'memorybank':
            from memcut.answer_execution.early_stop import POLICY
            if plan['native_qa'] != {'top_k': 10} or plan['answer_execution'] != POLICY:
                raise ValueError('MemoryBank answer policy changed')
        gateway = None
        try:
            for stage in stages:
                core.write_json(folder / 'STATUS.json', {'stage': stage, 'state': 'running', 'pid': os.getpid()})
                started = time.time()
                if stage == 'report':
                    result = base.report(folder, plan)
                    extra = {key: path for key, path in (('formation_diagnostics', folder / 'full/DIAGNOSTICS.json'),
                                                         ('coverage', folder / 'COVERAGE.json')) if path.exists()}
                    for key, path in extra.items():
                        value = core.read(path)
                        result[key] = value['summary'] if key == 'coverage' else value
                    if extra:
                        core.write_json(folder / 'SUMMARY.json', result)
                else:
                    if gateway is None:
                        inner = core.gateway(folder, plan)[0]
                        gateway = transport.TransportGateway(inner)
                    result = getattr(base, stage)(folder, plan, gateway)
                    if inspect.isawaitable(result):
                        result = asyncio.run(result)
                    if stage == 'build':
                        write_diagnostics(folder, plan)
                with (folder / 'STAGES.jsonl').open('a') as out:
                    out.write(json.dumps({'stage': stage, 'complete': True,
                                           'seconds': time.time() - started}) + '\n')
        except BaseException as exc:
            core.write_json(folder / 'STATUS.json', dict(stage=stage, state='stopped', error_type=type(exc).__name__))
            raise
        finally:
            if gateway is not None:
                gateway.close()
        core.write_json(folder / 'STATUS.json', {'stage': command, 'state': 'complete'})
        return result


def resolve(folder, retry=False):
    """Receipts sent without an answer. --retry archives them (requests/.archived/, ARCHIVED.jsonl) so the next
    run sends them again; one whose outcome was unknown may then be paid twice, which ARCHIVED.jsonl records."""
    folder = Path(folder)
    with (folder / 'RUN.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows = transport.pending(folder)
        result = {'pending': rows}
        if retry and rows:
            result['archived'] = transport.archive(folder, [r['request_id'] for r in rows],
                                                   'resolve --retry: sent, outcome unknown')
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare', help='Freeze a new plan; no API or model load')
    p.add_argument('--system', choices=SYSTEMS, required=True)
    for name in ('output', 'source', 'contracts', 'annotations'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--model', default=qwen.MODEL)
    p.add_argument('--endpoint', default=qwen.ENDPOINT)
    p.add_argument('--compatibility', choices=['openai', 'deepseek', 'mimo'], default='openai',
                   help='request convention of the endpoint: deepseek and mimo turn thinking off (default openai)')
    p.add_argument('--claims', nargs='+', help='claim ids, c1 c2 or c1,c2')
    p.add_argument('--native-eval-k', type=int,
                   help='turnrag, sessionrag: items of the native readout (default 16)')
    p.add_argument('--embedding', choices=['e5-small-v2', 'all-MiniLM-L6-v2'],
                   help='turnrag, sessionrag: retrieval encoder (default e5-small-v2)')
    p.add_argument('--accept-coverage', action='store_true',
                   help='sessionrag: keep a native k whose context shows over 25%% of the episode')
    for name in ('run', 'build', 'delete', 'answer', 'report', 'status'):
        sub.add_parser(name).add_argument('--run', required=True)
    p = sub.add_parser('resolve', help='List receipts sent without an answer; --retry sends them again next run')
    p.add_argument('--run', required=True)
    p.add_argument('--retry', action='store_true')
    args = parser.parse_args()
    if getattr(args, 'claims', None):          # c1 c2 and c1,c2 both name claims c1 and c2
        args.claims = [c for item in args.claims for c in item.split(',') if c]
    qwen.install()
    local.install()                      # a model served on this machine: no key, Qwen3 thinking off
    if os.environ.get('DASHSCOPE_API_KEY') and not os.environ.get('MEMCUT_MODEL_API_KEY'):
        os.environ['MEMCUT_MODEL_API_KEY'] = os.environ['DASHSCOPE_API_KEY']
    if not os.environ.get('MEMCUT_MODEL_API_KEY') and local.is_local(getattr(args, 'endpoint', '') or ''):
        os.environ['MEMCUT_MODEL_API_KEY'] = 'EMPTY'
    if args.command == 'prepare':
        plan = prepare(args)
        result = {'run': args.output, 'system': args.system, 'claims': len(plan['claims']), 'api_calls': 0}
    elif args.command == 'status':
        path = Path(args.run) / 'STATUS.json'
        result = core.read(path) if path.exists() else {'state': 'prepared'}
        result['pending_dispatches'] = transport.pending(args.run)
    elif args.command == 'resolve':
        result = resolve(args.run, args.retry)
    else:
        execute(args.run, args.command)
        result = core.read(Path(args.run) / 'STATUS.json')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
