"""Current evaluation policy, separate from immutable native bank metadata.

The package-level PROTOCOL dictionaries are retained as legacy storage identity
for compatible paid banks. Runners overlay this policy when freezing NEW plans;
old plans/results cannot silently acquire a different scoring rule.
"""
import copy

from memcut.native_memory.contract import digest

ROTATION_COUNT = 2
TRIAL_IDS = {1, 2}
SCORING = {
    'version': 'two-fixed-permutations-all-correct-v1',
    'rotation_count': ROTATION_COUNT,
    'selection': 'source trial1 and trial2, fixed before any model answer',
    'recovered': 'both valid answers correct; missing/invalid is unavailable',
    'conditional_cohort': 'Full residual 2/2, shared across both readouts',
}


def evaluation_protocol(native_protocol):
    return {**copy.deepcopy(native_protocol),
            'version': native_protocol['version'] + '-eval2of2-v1',
            'scoring': copy.deepcopy(SCORING),
            'score': 'two frozen permutations both correct; missing/invalid is unavailable',
            'conditional_cohort': SCORING['conditional_cohort']}


def select_rotations(rotations, choice_ids):
    """Keep the first two authored permutations; never select by answer quality."""
    ids = list(choice_ids)
    if rotations is None:
        rotations = [{'trial':i+1, 'display_to_choice':dict(zip('abcde'[:len(ids)],ids[i:]+ids[:i]))}
                     for i in range(ROTATION_COUNT)]
    if (len(rotations) not in {2,3} or any(type(r.get('trial')) is not int for r in rotations)
            or {r['trial'] for r in rotations} != set(range(1,len(rotations)+1))):
        raise ValueError('Require source trial1/trial2, optionally historical trial3')
    normalized = []
    for r in sorted(rotations,key=lambda r:r['trial']):
        mapping = r['display_to_choice']
        if (set(mapping) != set('abcde'[:len(ids)]) or set(mapping.values()) != set(ids)):
            raise ValueError('Incomplete choice permutation')
        normalized.append({'trial':r['trial'], 'display_to_choice':dict(sorted(mapping.items()))})
    if len({digest(r['display_to_choice']) for r in normalized}) != len(normalized):
        raise ValueError('Choice permutations must be distinct')
    return normalized[:ROTATION_COUNT]


def validate_claim_rotations(claim):
    rotations = claim['rotations']
    if len(rotations) != ROTATION_COUNT or {r['trial'] for r in rotations} != TRIAL_IDS:
        raise ValueError('Current scoring requires exactly two frozen rotations; replan historical runs')
    if select_rotations(rotations,[c['id'] for c in claim['choices']]) != rotations:
        raise ValueError('Frozen rotations must use canonical trial order')
