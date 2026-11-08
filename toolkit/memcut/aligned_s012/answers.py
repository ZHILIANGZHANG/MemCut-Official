"""Answer readouts that do not stop the run on one unusable answer.

A readout is scored from its fixed option rotations; an invalid or incomplete answer is unavailable, never a
zero (scoring.SCORING['recovered']). The frozen answer stages stopped the whole stage at the first such answer,
and a rerun replayed the same cached reply into the same stop, so every later claim stayed unanswered. Here a
readout whose answer is unusable (cut off at its cap, refused, unparsable, without a displayed label) is written
to <family>.unavailable.json with the request that produced it, and the stage goes on. Its scores are never
made up: the frozen report still computes a rate only on a complete cohort, and the aligned report adds the
cohort of claims available in every condition, with its rates and paired S2-S1 differences.
"""
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from memcut.native_memory.contract import digest
from memcut.native_memory.persistence import write_json
from memcut.source_anchored_v2 import evaluation
from memcut.source_anchored_v2 import runner as core

POLICY = {'version': 'aligned-unavailable-readout-v1',
          'unusable_answer': 'unavailable (recorded with its request); the stage continues; never scored',
          'rates': 'frozen metrics need a complete cohort; available_cohort uses claims available in every condition'}


class Unusable(ValueError):
    """The model answered and the frozen cache rejected the reply (cut off, refused, empty)."""


class _Recorder:
    """The answer cache, remembering the requests of one readout."""

    def __init__(self, cache):
        self.cache, self.sent, self.last, self.rejected, self.broken = cache, 0, None, None, False

    def complete(self, request, **kwargs):
        self.sent += 1
        before = len(self.cache.calls)
        try:
            response = self.cache.complete(request, **kwargs)
        except RuntimeError as exc:
            if len(self.cache.calls) > before:      # a reply came back: the cache refused its finish or text
                call = self.cache.calls[-1]
                self.rejected = {'request_id': call['request_id'], 'finish_reason': call['finish_reason'],
                                 'error': str(exc)}
                raise Unusable(str(exc)) from exc
            self.broken = True                       # nothing came back: a transport failure stops the run
            raise
        except BaseException:
            self.broken = True                       # a cache or profile check, never an answer
            raise
        self.last = response
        return response

    def __getattr__(self, name):
        return getattr(self.cache, name)


def readout(evaluate, claim, context, cache, model):
    """(result, None) for a scored readout, (None, failure) for an unavailable one."""
    recorder = _Recorder(cache)
    try:
        return evaluate(claim, context, recorder, model), None
    except Unusable:
        failure = recorder.rejected
    except ValueError as exc:
        if recorder.last is None or recorder.broken:  # a frozen-state or cache check, not an answer
            raise
        failure = {'request_id': recorder.last['request_id'], 'finish_reason': recorder.last['finish_reason'],
                   'error': str(exc)}
    trial = claim['rotations'][recorder.sent - 1]['trial']
    return None, {'trial': trial, 'trials_answered_before': recorder.sent - 1, **failure}


@contextmanager
def answer_cap(plan):
    cap = plan['aligned_answer_policy']['max_tokens']
    original = evaluation.answer_request
    with patch.object(evaluation, 'answer_request', lambda *a, **k: {**original(*a, **k), 'max_tokens': cap}):
        yield


def answer(folder, plan, completions, *, evaluate, claim_ids=None):
    if plan['controls']:
        raise ValueError('The aligned entry plans no controls')
    folder = Path(folder)
    cache = core.role_cache(completions, 'answer')
    unavailable = 0
    with answer_cap(plan):
        for claim in core.stage_claims(plan, claim_ids):
            for condition in plan['conditions']:
                state = core.validate_state(folder, plan, claim, condition)
                for family, context in state['contexts'].items():
                    target = folder / 'claims' / claim['claim_id'] / condition / f'{family}.json'
                    envelope = {'claim_id': claim['claim_id'], 'condition': condition, 'family': family,
                                'plan_sha256': digest(plan)}
                    result, failure = readout(evaluate, claim, context, cache, plan['models']['answer'])
                    if failure is None:
                        core.sealed_write(target, {**envelope, 'result': result})
                    else:
                        unavailable += 1
                        core.sealed_write(target.with_name(f'{family}.unavailable.json'), {
                            **envelope, 'unavailable': failure, 'context_sha256': context['context_sha256'],
                            'state_identity': context['state_identity']})
    return {'unavailable_readouts': unavailable}


def _families(plan):
    return ['native', 'residual'] + (['residual_plus'] if plan['residual_plus'] else [])


def available_cohort(folder, plan):
    """Per readout family: the claims scored in every condition, their rates and paired S2-S1 differences."""
    folder = Path(folder)
    ids = [c['claim_id'] for c in plan['claims']]
    scored, missing = {}, []
    for cid in ids:
        for condition in plan['conditions']:
            for family in _families(plan):
                path = folder / 'claims' / cid / condition / f'{family}.json'
                if path.exists():
                    scored[cid, condition, family] = core.read(path)['result']['recovered']
                else:
                    gone = path.with_name(f'{family}.unavailable.json')
                    missing.append({'claim_id': cid, 'condition': condition, 'family': family,
                                    **(core.read(gone)['unavailable'] if gone.exists() else {'reason': 'not answered'})})
    families = {}
    for family in _families(plan):
        cohort = [c for c in ids if all((c, s, family) in scored for s in plan['conditions'])]
        full_residual = {c for c in cohort if scored.get((c, 'Full', 'residual'))}
        rates, paired = {}, {}
        for condition in plan['conditions']:
            values = [int(scored[c, condition, family]) for c in cohort]
            conditional = [int(scored[c, condition, family]) for c in cohort if c in full_residual]
            rates[condition] = {'rate': sum(values) / len(values) if values else None,
                                'ci95': evaluation.bootstrap(values),
                                'conditional': {'denominator': len(conditional),
                                                'rate': sum(conditional) / len(conditional) if conditional else None}}
            if condition.startswith('S2@'):
                other = condition.replace('S2@', 'S1@')
                differences = [int(scored[c, condition, family]) - int(scored[c, other, family]) for c in cohort]
                paired[f'{condition}-minus-{other}'] = {
                    'mean': sum(differences) / len(differences) if differences else None,
                    'ci95': evaluation.bootstrap(differences)}
        families[family] = {'claims': len(cohort), 'excluded_claims': sorted(set(ids) - set(cohort)),
                            'rates': rates, 'paired_differences': paired}
    return {'policy': POLICY, 'planned_claims': len(ids), 'unavailable_readouts': missing, 'families': families}


def report(original):
    def aligned_report(folder, plan):
        result = original(folder, plan)
        result['available_cohort'] = available_cohort(folder, plan)
        write_json(Path(folder) / 'SUMMARY.json', result)
        return result
    return aligned_report
