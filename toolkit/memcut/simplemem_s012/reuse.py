"""Reuse the existing audited branch importer with SimpleMem identity rules."""
from contextlib import contextmanager
import copy
from unittest.mock import patch

from memcut.execution_optimization import branches
from . import runner


def binding(plan):
    if plan.get('system') != 'simplemem' or plan.get('controls') or plan.get('residual_plus'):
        raise ValueError('SimpleMem branch reuse requires the standard two-readout experiment')
    value = copy.deepcopy(plan)
    for key in ('claims', 'reuse_full', 'execution', 'answer_policy',
                'receipt_catalog_sha256', 'branch_catalog_sha256'):
        value.pop(key, None)
    for key in ('models', 'providers', 'request_profiles'):
        value[key].pop('answer')
    value['input_normalization'].pop('changes')  # selected claims are checked independently
    return value


@contextmanager
def runtime():
    with patch.object(branches, '_binding', binding), patch.object(branches, '_source_plan', runner.load_plan):
        yield


def prepare_catalog(source, plan):
    with runtime():
        return branches.prepare_catalog(source, plan)


def import_claims(folder, plan, catalog, claim_ids=None):
    with runtime():
        return branches.import_claims(folder, plan, catalog, claim_ids=claim_ids)
