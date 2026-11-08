#!/usr/bin/env python3
"""MemoryBank: original requests and 2/2 recovery, stop after valid wrong trial1.

Same commands as optimized_s012.py, scoped to MemoryBank. A separate entry keeps
all historical code hashes and banks loadable through their original entry.
"""
from contextlib import contextmanager
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from memcut.execution_optimization import cli_core as base
from memcut.answer_execution import early_stop

_prepare, _load, _runtime = base.prepare, base.load_plan, base.runtime
_report = base.core.report


def policy():
    files = list((ROOT/'memcut/answer_execution').glob('*.py')) + [Path(__file__).resolve()]
    return {**early_stop.POLICY, 'system': 'memorybank',
            'code': {str(p.relative_to(ROOT)): base.core.sha(p) for p in sorted(files)}}


def validate(plan):
    if plan['execution_optimization']['system'] != 'memorybank' or plan.get('answer_execution') != policy():
        raise ValueError('Require a new frozen MemoryBank early-stop plan; historical plans are unchanged')
    if plan.get('controls') or plan.get('retention') or plan.get('residual_plus'):
        raise ValueError('This entry supports the accepted native/residual main experiment only')


def prepare(args):
    if args.system != 'memorybank':
        raise ValueError('Use optimized_s012.py for other memory systems')
    plan = _prepare(args)
    plan['answer_execution'] = policy()
    validate(plan)
    base.core.write_json(Path(args.output)/'PLAN.json', plan)
    return plan


def load_plan(folder):
    plan = _load(folder)
    validate(plan)
    return plan


def report(folder, plan):
    # A claimed empty-memory result must correspond to the frozen readout.
    # Never let a missing trial be smuggled through as structural emptiness.
    for claim in plan['claims']:
        for condition in plan['conditions']:
            directory = Path(folder)/'claims'/claim['claim_id']/condition
            for family in ('native', 'residual'):
                path = directory/(family+'.json')
                if not path.exists():
                    continue
                state = base.core.validate_state(folder, plan, claim, condition)
                row = base.core.read(path)
                if (row.get('claim_id'), row.get('condition'), row.get('family')) != (claim['claim_id'], condition, family):
                    raise ValueError('Answer envelope differs from its frozen state')
                ctx = state['contexts'][family]
                if bool(row['result'].get('evidence_derived_empty')) != ctx['empty']:
                    raise ValueError('Structural empty score differs from the frozen memory')
                early_stop.validate_result(row['result'], claim, family)
    return _report(folder, plan)


@contextmanager
def runtime(plan, *, native=True):
    validate(plan)
    with _runtime(plan, native=native), \
         patch.object(base.scheduling, 'evaluate', early_stop.evaluate), \
         patch.object(base.core, 'evaluate', early_stop.evaluate), \
         patch.object(base.core, 'summarize', early_stop.summarize), \
         patch.object(base.core, 'report', report):
        yield


def main():
    with patch.object(base, 'prepare', prepare), patch.object(base, 'load_plan', load_plan), \
         patch.object(base, 'runtime', runtime), patch.object(base, '__doc__', __doc__):
        base.main()


if __name__ == '__main__':
    main()
