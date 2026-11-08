#!/usr/bin/env python3
"""Retention for runs of the aligned entry: the share of unrelated facts a system still answers after a deletion,
measured by the MemCut protocol's probe rule, so every system's run has a retention value next to its
residual and native readouts.

  python run.py aligned-retention --run RUN [--conditions S2@10] [--probes 5]

The run's answer stage must be complete: the Full native readouts of the planned claims pick the probes. For a
claim, the probes are the other planned claims whose evidence (carrier) and disclosure turns do not meet its own,
in the fixed order sha256('<EP>:<claim>|<EP>:<probe>') the protocol defines, and of them the first --probes that
Full answers natively (the probe's own Full native readout: both fixed trials choose the target). Each probe is then
asked on the claim's state of each condition: the state's bank restored from its snapshot, the system's own native
retrieval for the probe's question, answered and scored by the system's own answer stage (same model, the aligned
answer cap, the same rules; an unusable answer is unavailable, never a miss). Retention = probes still answered /
probes answered, over the probes with a usable answer. Plan an episode's claims together (the default): a claim
outside the plan cannot be a probe.

Writes retention_aligned/<claim>/<condition>.json and RETENTION.json (the summary experiments/paper_tables.py
reads). Requests go through the run's receipts (requests/): a re-run sends only what was not answered, under the
aligned entry's paid lock and unresolved-dispatch check. This tool leaves the aligned entry's own files alone, so
plans prepared before it still run.
"""
import argparse
import asyncio
from contextlib import ExitStack
import fcntl
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from memcut.aligned_s012 import SYSTEMS, transport  # noqa: E402
from memcut.aligned_s012.answers import answer_cap, readout  # noqa: E402
from memcut.aligned_s012.runtime import ANSWER_POLICY, identity, runtime  # noqa: E402
from memcut.model_adapters import qwen  # noqa: E402
from memcut.native_memory.contract import digest  # noqa: E402
from memcut.source_anchored_v2 import evaluation  # noqa: E402
from memcut.source_anchored_v2 import runner as core  # noqa: E402
from tools.memcut_s012 import paid_lock_path, unresolved  # noqa: E402

POLICY = {'version': 'aligned-retention-v1',
          'probes': 'other planned claims whose evidence and disclosure turns do not meet the claim\'s, in '
                    'sha256("<EP>:<claim>|<EP>:<probe>") order; the first n that Full answers natively (2/2)',
          'readout': "native, by the system's own answer stage (2/2; unusable answers are unavailable)",
          'rate': 'probes answered / probes with a usable answer'}
OUT = 'retention_aligned'


def episode_of(plan, folder):
    """EP01, EP14, ...: from the source file's name, its id, or the run directory's name (as paper_tables reads it)."""
    src = Path(plan['inputs']['source']['path'])
    names = [src.name, str(core.read(src).get('id', '')) if src.exists() else '', Path(folder).name]
    for text in names:
        m = re.search(r'EP\d{2}', text)
        if m:
            return m.group(0)
    raise ValueError(f'{folder}: cannot tell its episode (no EPnn in {src.name}, its id or the directory name)')


def candidates(plan, claim, episode):
    """The claims that may be probes of `claim`, in the protocol's fixed order."""
    mine = set(claim['evidence_turns']) | set(claim['disclosure_turns'])
    others = [c for c in plan['claims'] if c['claim_id'] != claim['claim_id']
              and not (set(c['evidence_turns']) | set(c['disclosure_turns'])) & mine]
    return sorted(others, key=lambda c: hashlib.sha256(
        f"{episode}:{claim['claim_id']}|{episode}:{c['claim_id']}".encode()).hexdigest())


def full_native(folder, claim_id):
    """True / False: the claim's Full native readout; None: not answered or unavailable."""
    row = Path(folder) / 'claims' / claim_id / 'Full' / 'native.json'
    return bool(core.read(row)['result']['recovered']) if row.exists() else None


def evaluator(system):
    if system == 'memorybank':                  # the aligned entry's MemoryBank answer stage (runtime.runtime)
        from memcut.answer_execution import early_stop
        return early_stop.evaluate
    return evaluation.evaluate


class Banks:
    """A condition's bank restored from its snapshot, as the system's own stages restore it."""

    def __init__(self, system, base, folder, plan, completions):
        self.system, self.base = system, base
        if system == 'memorybank':
            self.temp = tempfile.TemporaryDirectory(prefix='memorybank-retention-')
            self.seed, self.n = base.template(folder, plan), itertools.count()
        else:
            self.factory = base.Factory(plan, completions) if system == 'simplemem' else base.Factory(plan)

    def open(self, snapshot):
        if self.system == 'memorybank':
            return self.base.branch(self.seed, Path(self.temp.name) / f'{next(self.n)}.pkl', snapshot)
        return self.factory.create(snapshot)

    def close(self):
        if self.system == 'memorybank':
            self.temp.cleanup()
        elif hasattr(self.factory, 'close'):
            self.factory.close()


async def native(bank, stem):
    result = await bank.native_context(stem)
    text = result.native_context['prompt']
    return {'context': text, 'context_sha256': digest(text), 'empty': False, 'state_identity': bank.state_identity()}


async def measure(system, base, folder, plan, completions, conditions, n):
    folder = Path(folder)
    episode = episode_of(plan, folder)
    evaluate, cache = evaluator(system), core.role_cache(completions, 'answer')
    banks = Banks(system, base, folder, plan, completions)
    try:
        with answer_cap(plan):
            for claim in plan['claims']:
                order = candidates(plan, claim, episode)
                probes = [c for c in order if full_native(folder, c['claim_id'])][:n]
                for condition in conditions:
                    target = folder / OUT / claim['claim_id'] / f'{condition}.json'
                    done = core.read(target) if target.exists() else {}
                    if done.get('plan_sha256') == digest(plan) and done.get('probes') == [p['claim_id'] for p in probes]:
                        continue                         # measured with these probes (another --probes asks again)
                    state = core.validate_state(folder, plan, claim, condition)
                    bank = banks.open(folder / 'claims' / claim['claim_id'] / condition / 'state.pkl')
                    try:
                        if bank.state_identity() != state['state_identity']:
                            raise ValueError(f"{claim['claim_id']} {condition}: the restored bank is not the saved state")
                        records = []
                        for p in probes:
                            ctx = await native(bank, p['stem'])
                            result, failure = readout(evaluate, p, ctx, cache, plan['models']['answer'])
                            records.append({'probe': p['claim_id'], 'context_sha256': ctx['context_sha256'],
                                            'recovered': None if failure else bool(result['recovered']),
                                            **({'unavailable': failure} if failure else {'result': result})})
                    finally:
                        if hasattr(bank, 'close'):
                            bank.close()
                    core.write_json(target, {'claim_id': claim['claim_id'], 'condition': condition, 'episode': episode,
                                             'plan_sha256': digest(plan), 'policy': POLICY,
                                             'state_identity': state['state_identity'], 'candidates': len(order),
                                             'probes': [p['claim_id'] for p in probes], 'records': records})
    finally:
        banks.close()
    return summary(folder, plan, conditions)


def summary(folder, plan, conditions):
    folder, out = Path(folder), {'policy': POLICY, 'plan_sha256': digest(plan), 'conditions': {}}
    for condition in conditions:
        claims, kept, asked, unavailable, without = [], 0, 0, 0, []
        for claim in plan['claims']:
            path = folder / OUT / claim['claim_id'] / f'{condition}.json'
            if not path.exists():
                continue
            rows = core.read(path)['records']
            done = [r['recovered'] for r in rows if r['recovered'] is not None]
            unavailable += len(rows) - len(done)
            if not rows:
                without.append(claim['claim_id'])
            kept, asked = kept + sum(done), asked + len(done)
            claims.append({'claim_id': claim['claim_id'], 'kept': sum(done), 'n': len(done)})
        out['conditions'][condition] = {'claims': claims, 'kept': kept, 'n': asked,
                                        'retention': kept / asked if asked else None, 'unavailable': unavailable,
                                        'claims_without_probes': without}
    core.write_json(folder / 'RETENTION.json', out)
    return out


def run(folder, conditions=('S2@10',), n=5):
    folder = Path(folder)
    raw = core.read(folder / 'PLAN.json')
    system = raw.get('alignment', {}).get('system')
    if system not in SYSTEMS or raw['alignment'] != identity(system):
        raise ValueError('Use a run of the aligned entry prepared by this version of it (PLAN.json alignment)')
    if raw.get('aligned_answer_policy') != ANSWER_POLICY:
        raise ValueError('Aligned answer policy changed; prepare a new run')
    if n < 1:
        raise ValueError('--probes must be at least 1')
    unknown = [c for c in conditions if c not in raw['conditions'] or c == 'Full']
    if unknown:
        raise ValueError(f"Conditions not in this plan (or Full): {', '.join(unknown)}; planned: {raw['conditions']}")
    missing = [c['claim_id'] for c in raw['claims']
               if not any((folder / 'claims' / c['claim_id'] / 'Full' / f).exists()
                          for f in ('native.json', 'native.unavailable.json'))]
    if missing:
        raise ValueError(f"Run the answer stage first: no Full native readout for {', '.join(missing[:8])}")
    with ExitStack() as stack:
        lock = stack.enter_context((folder / 'RUN.lock').open('a+b'))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        shared = paid_lock_path()
        shared.parent.mkdir(parents=True, exist_ok=True)
        paid_lock = stack.enter_context(shared.open('a+b'))
        fcntl.flock(paid_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        transport.archive_not_billed(folder)
        if unresolved(folder):
            raise ValueError('Unresolved dispatch retained (sent, outcome unknown); list them with '
                             f'`python run.py aligned resolve --run {folder}` and add --retry to send them again')
        base = stack.enter_context(runtime(system, native=True))
        plan = base.load_plan(folder)
        inner = base.gateway(folder, plan) if system == 'simplemem' else core.gateway(folder, plan)[0]
        gateway = transport.TransportGateway(inner)
        try:
            return asyncio.run(measure(system, base, folder, plan, gateway, list(conditions), n))
        finally:
            gateway.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', required=True)
    parser.add_argument('--conditions', nargs='+', default=['S2@10'],
                        help='conditions to measure (default S2@10, the row of Table 1); S0, S1@10, ... also')
    parser.add_argument('--probes', type=int, default=5, help='probes per claim (the protocol: 5)')
    args = parser.parse_args()
    args.conditions = [c for item in args.conditions for c in item.split(',') if c]   # S0 S2@10 or S0,S2@10
    qwen.install()
    if os.environ.get('DASHSCOPE_API_KEY') and not os.environ.get('MEMCUT_MODEL_API_KEY'):
        os.environ['MEMCUT_MODEL_API_KEY'] = os.environ['DASHSCOPE_API_KEY']
    result = run(args.run, tuple(args.conditions), args.probes)
    print(json.dumps({c: {k: v[k] for k in ('kept', 'n', 'retention', 'unavailable')}
                      for c, v in result['conditions'].items()}, ensure_ascii=False))


if __name__ == '__main__':
    main()
