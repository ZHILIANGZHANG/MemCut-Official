"""Versioned scheduling optimizations; pinned native formation stays intact."""

POLICY = 'ordered-concurrent-formation-v1'


def execution(workers=4, checkpoint_turns=80):
    if type(workers) is not int or not 1 <= workers <= 16:
        raise ValueError('Formation workers must be between 1 and 16')
    if type(checkpoint_turns) is not int or not 20 <= checkpoint_turns <= 320 or checkpoint_turns % 20:
        raise ValueError('Checkpoint interval must be a multiple of 20, from 20 to 320')
    return {'policy': POLICY, 'workers': workers, 'checkpoint_turns': checkpoint_turns}
