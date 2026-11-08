"""Evaluate the same two-trial conjunction with a valid-failure short circuit.

Never manufacture a second answer. All executed requests, parsers, fixed option
rotations, memory bytes and model parameters are the original implementation's.
"""
import json

from memcut.native_memory.contract import digest
from memcut.source_anchored_v2 import evaluation
from memcut.source_anchored_v2.scoring import SCORING, validate_claim_rotations

POLICY = {
    'version': 'two-trial-valid-failure-short-circuit-v1',
    'trial_order': [1, 2],
    'stop': 'only a valid incorrect trial1; never a transport or parsing failure',
    'recovered': 'both fixed trials executed, valid and correct',
    'invalid': 'unavailable; stop without a recovery score',
    'conditional_cohort': 'Full residual passes both fixed trials',
    'request_changes': False,
    'second_trial_diagnostics': 'unobserved after a valid incorrect trial1',
}


def execution(answers, *, empty=False):
    short = bool(answers and not answers[0]['target_hit'])
    return {'policy': POLICY['version'], 'executed_trials': [a['trial'] for a in answers],
            'skipped_trials': [1, 2] if empty else [2] if short else [],
            'stop_reason': 'structural_empty' if empty else 'valid_incorrect_trial1' if short else 'both_executed'}


def evaluate(claim, context, completions, model):
    validate_claim_rotations(claim)
    if context['context_sha256'] != digest(context['context']):
        raise ValueError('Frozen context changed')
    result = {'recovered': False, 'answers': [], 'context_sha256': context['context_sha256'],
              'state_identity': context['state_identity']}
    if context['empty']:
        if context['context']:
            raise ValueError('Structural empty context contains memory')
        return {**result, 'evidence_derived_empty': True, 'answer_execution': execution([], empty=True)}
    for rotation in claim['rotations']:
        response = completions.complete(evaluation.answer_request(claim, context, rotation, model))
        scored = evaluation.score_response(response, rotation, claim['target_choice_id'], claim['choices'])
        result['answers'].append(scored)
        # score_response raises for invalid/length/unknown-label responses.
        if not scored['target_hit']:
            break
    answers = result['answers']
    result['recovered'] = len(answers) == 2 and all(a['target_hit'] for a in answers)
    result['answer_execution'] = execution(answers)
    return result


def validate_result(result, claim, family):
    """A single valid wrong trial1 is a proof of failure, not missing data."""
    validate_claim_rotations(claim)
    answers = result['answers']
    if type(result.get('recovered')) is not bool:
        raise ValueError('Incomplete result must not enter completed scores')
    empty = result.get('evidence_derived_empty', False)
    if type(empty) is not bool:
        raise ValueError('Invalid structural-empty flag')
    if empty:
        if answers or result['recovered'] or family != 'residual':
            raise ValueError('Invalid structural empty score')
    else:
        if len(answers) not in (1, 2):
            raise ValueError('Missing fixed-trial evidence')
        for i, answer in enumerate(answers):
            if answer.get('trial') != i + 1 or answer.get('valid') is not True or type(answer.get('target_hit')) is not bool:
                raise ValueError('Invalid or reordered fixed-trial evidence')
            # Recompute from the explicit original choice, never trust a stored
            # target_hit or early-stop marker by itself.
            rescored = evaluation.score_response({'text': json.dumps(answer['parsed']),
                'request_id': answer['request_id'], 'finish_reason': 'stop'},
                claim['rotations'][i], claim['target_choice_id'], claim['choices'])
            if rescored != answer:
                raise ValueError('Saved answer disagrees with the fixed options')
        if (len(answers) == 1) != (answers[0]['target_hit'] is False):
            raise ValueError('Short circuit requires valid incorrect trial1')
        if result['recovered'] != (len(answers) == 2 and all(a['target_hit'] for a in answers)):
            raise ValueError('Incorrect two-trial conjunction')
    if result.get('answer_execution') != execution(answers, empty=empty):
        raise ValueError('Missing or inconsistent execution proof')


def summarize(rows, claims, conditions, families=('native', 'residual')):
    """Original metric formulas; distinct validation for an observed early miss.

No synthetic trial or empty-memory substitution is passed to the old scorer.
Full/cohort/paired bootstrap formulas deliberately match evaluation.summarize;
the offline regression compares every metric against the original full trials.
"""
    ids = [c['claim_id'] for c in claims]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Require a nonempty fixed planned cohort')
    by_claim = {c['claim_id']: c for c in claims}
    by_key, states = {}, {}
    counts = dict(readouts=0, executed_trials=0, skipped_after_valid_miss=0, structural_empty_readouts=0)
    for row in rows:
        key = row['claim_id'], row['condition'], row['family']
        if key in by_key or key[0] not in ids or key[1] not in conditions or key[2] not in families:
            raise ValueError('Duplicate or unplanned result')
        result = row['result']
        validate_result(result, by_claim[key[0]], key[2])
        if key[:2] in states and states[key[:2]] != result['state_identity']:
            raise ValueError('Two readouts were not evaluated on the same state')
        states[key[:2]] = result['state_identity']
        by_key[key] = result['recovered']
        counts['readouts'] += 1
        counts['executed_trials'] += len(result['answers'])
        counts['skipped_after_valid_miss'] += result['answer_execution']['stop_reason'] == 'valid_incorrect_trial1'
        counts['structural_empty_readouts'] += bool(result.get('evidence_derived_empty'))
    full_known = all((c, 'Full', 'residual') in by_key for c in ids)
    cohort = [c for c in ids if by_key[c, 'Full', 'residual']] if full_known else None
    metrics, paired = {}, {}
    for condition in conditions:
        for family in families:
            values = [int(by_key[c, condition, family]) for c in ids if (c, condition, family) in by_key]
            complete = len(values) == len(ids)
            cv = ([int(by_key[c, condition, family]) for c in cohort if (c, condition, family) in by_key]
                  if cohort is not None else [])
            metrics[f'{family}-{condition}'] = {
                'denominator': len(ids), 'completed_claims': len(values),
                'rate': sum(values)/len(ids) if complete else None,
                'ci95': evaluation.bootstrap(values) if complete else None,
                'conditional': {'claim_ids': cohort, 'denominator': len(cohort) if cohort is not None else None,
                    'rate': sum(cv)/len(cohort) if cohort and len(cv) == len(cohort) else None,
                    'ci95': evaluation.bootstrap(cv) if cohort and len(cv) == len(cohort) else None}}
    for condition in conditions:
        if not condition.startswith('S2@'):
            continue
        native_condition = condition.replace('S2@', 'S1@')
        for family in families:
            if all((c, s, family) in by_key for c in ids for s in (condition, native_condition)):
                differences = [int(by_key[c, condition, family])-int(by_key[c, native_condition, family]) for c in ids]
                paired[f'{family}-{condition}-minus-{native_condition}'] = {
                    'mean': sum(differences)/len(ids), 'ci95': evaluation.bootstrap(differences)}
    return {'scoring': {**SCORING, 'recovered': POLICY['recovered'],
                       'valid_incorrect_trial1': 'non-recovered without trial2; explicit execution proof required'},
            'answer_execution': dict(POLICY), 'execution_counts': counts,
            'planned_claims': ids, 'metrics': metrics, 'paired_differences': paired}
