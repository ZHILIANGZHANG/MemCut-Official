"""examples/keyword_bank.py through the protocol's shared steps; no model, network or paid call."""
import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))
from keyword_bank import KeywordBank  # noqa: E402
from memcut.source_anchored_v2.candidates import build_pool, selector_input  # noqa: E402
from memcut.source_anchored_v2.evaluation import contexts  # noqa: E402


def turns():
    lines = ['I booked the dentist for Friday.', 'Noted.',
             'Our supplier Fern Labs sent the contract.', 'Good.',
             'Fern Labs is owned by my brother Bo.', 'Understood.',
             'The dentist moved to Monday.', 'Okay.',
             'Bo called about Fern Labs invoices.', 'Thanks.',
             'Lunch with Kit on Tuesday.', 'Enjoy.']
    return {f't{i + 1:02d}': {'id': f't{i + 1:02d}', 'role': 'user' if i % 2 == 0 else 'assistant',
                              'speaker': 'Ada' if i % 2 == 0 else 'Assistant', 'text': text}
            for i, text in enumerate(lines)}


@pytest.fixture
def bank():
    b = KeywordBank({}, turns())
    b.form()
    return b


def test_every_object_traces_to_the_turns_it_came_from(bank):
    assert len(bank.rows) == 6
    assert all(len(bank.trace(oid)) == 2 for oid in bank.rows)
    assert bank.reverse_trace('t05') == {bank.address('w0003')}


def test_the_candidate_pool_is_built_from_the_systems_own_retrieval(bank):
    request = "Forget that Fern Labs belongs to my brother."
    pool = asyncio.run(build_pool(bank, request, disclosure_turns=['t05'], n_obj=3, n_turn=4))
    assert pool['actual_objects'] == 3 and 't05' not in {r['turn_id'] for r in pool['turns']}
    payload, groups = selector_input(pool)                      # what S2's model would see
    assert payload['deletion_request'] == request and groups


def test_deleting_a_turn_removes_every_object_derived_from_it(bank):
    receipt = asyncio.run(bank.delete_turns(['t05'], None))
    assert receipt['deleted_objs'] == [bank.address('w0003')]
    assert receipt['collateral_raw_turns'] == ['t06']           # its window also held the reply
    assert receipt['selected_turns_have_empty_der']
    assert asyncio.run(bank.residual_objects(['t05'])) == []


def test_both_readout_contexts_come_from_the_surviving_memory(bank):
    claim = {'stem': 'Who owns Fern Labs?', 'evidence_turns': ['t03', 't09'], 'disclosure_turns': ['t05']}
    asyncio.run(bank.delete_turns(['t05'], None))
    ctx = asyncio.run(contexts(bank, claim))
    assert 'owned by my brother' not in ctx['native']['context']
    assert 'Fern Labs sent the contract' in ctx['residual']['context']
    assert 'Bo called about Fern Labs' in ctx['residual']['context']
