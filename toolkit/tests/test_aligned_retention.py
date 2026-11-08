"""Retention on the aligned entry's runs (tools/aligned_retention.py): the protocol's probe rule on the toolkit's own
states and answer stage. Offline fixtures only: synthetic embeddings, scripted replies."""
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from memcut.native_memory.persistence import write_json
from memcut.raw_turn_rag import runner as raw
from memcut.source_anchored_v2 import runner as core
from test_aligned_rag import offline_gateway, replies, requests_sent
from test_session_rag import FixtureEmbedding
from tools import aligned_retention as retention
from tools import memcut_s012 as entry


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv('MEMCUT_PAID_LOCK', str(tmp_path / 'paid.lock'))
    raw_factory = raw.Factory
    monkeypatch.setattr(raw, 'Factory', lambda plan, **_: raw_factory(plan, embedder=FixtureEmbedding(384)))


def episode(tmp_path):
    """Three claims: c1 and c2 share no turn, c2 and c3 share none, c1 and c3 share a carrier (t1)."""
    turns = [{'id': f't{i}', 'role': 'user' if i % 2 else 'assistant', 'speaker': 'Mira',
              'text': f'Rank {i}: unchanged source text {i}.'} for i in range(1, 41)]
    source = tmp_path / 'EP07.json'
    write_json(source, {'id': 'EP07_FIXTURE', 'sessions': [
        {'id': f's{j}', 'timestamp': f'2026-01-0{j + 1}T12:00:00', 'turns': turns[j * 10:(j + 1) * 10]}
        for j in range(4)]})
    claim = {'claim_id': 'c1', 'category': 'factual_composition', 'stem': 'Which option?',
             'deletion_request': 'Forget the private information.', 'target_choice_id': 'yes',
             'choices': [{'id': 'yes', 'text': 'Known option'}] + [{'id': f'n{i}', 'text': f'Wrong {i}'} for i in range(3)]}
    write_json(tmp_path / 'contracts.json', {'source_sha256': core.sha(source), 'claims': [
        claim, {**claim, 'claim_id': 'c2', 'stem': 'Which option, second?'},
        {**claim, 'claim_id': 'c3', 'stem': 'Which option, third?'}]})

    def annotation(cid, i):
        turn = turns[i - 1]
        return {'claim_id': cid, 'record_id': turn['id'], 'role': turn['role'],
                'text_sha256': hashlib.sha256(turn['text'].encode()).hexdigest()}
    carriers = {'c1': (1, 2), 'c2': (3, 4), 'c3': (1, 5)}
    write_json(tmp_path / 'annotations.json', {'source_sha256': core.sha(source),
        'direct': [annotation('c1', 30), annotation('c2', 31), annotation('c3', 32)],
        'carriers': [annotation(c, i) for c, ts in carriers.items() for i in ts],
        'distractors': [annotation(c, 29) for c in carriers]})
    return SimpleNamespace(system='turnrag', output=str(tmp_path / 'turnrag-run'), source=str(source),
                           contracts=str(tmp_path / 'contracts.json'), annotations=str(tmp_path / 'annotations.json'),
                           model='offline-fixture', endpoint='offline-fixture', claims=None, native_eval_k=None,
                           accept_coverage=False)


def test_probes_follow_the_protocol_rule_and_the_summary_counts_them(tmp_path, monkeypatch):
    run = episode(tmp_path)
    plan = entry.prepare(run)
    offline_gateway(monkeypatch, replies())
    entry.execute(run.output)
    order = {c['claim_id']: [p['claim_id'] for p in retention.candidates(plan, c, 'EP07')] for c in plan['claims']}
    assert order['c1'] == ['c2'] and order['c3'] == ['c2'] and sorted(order['c2']) == ['c1', 'c3']
    first = min(['c1', 'c3'], key=lambda p: hashlib.sha256(f'EP07:c2|EP07:{p}'.encode()).hexdigest())
    assert order['c2'][0] == first                              # the order the protocol hashes
    sent = requests_sent(run.output)
    result = retention.run(run.output, ('S2@10', 'S0'), n=1)
    s2 = result['conditions']['S2@10']
    assert s2['n'] == 3 and s2['kept'] == 3 and s2['retention'] == 1.0     # the fixture always answers the target
    row = core.read(Path(run.output) / 'retention_aligned' / 'c2' / 'S2@10.json')
    assert row['probes'] == [first] and row['episode'] == 'EP07' and row['candidates'] == 2
    assert row['records'][0]['result']['answers'][0]['target_hit']
    assert core.read(Path(run.output) / 'RETENTION.json')['conditions']['S0']['n'] == 3
    asked = requests_sent(run.output)          # a request the run already sent (same context and question) is reused
    retention.run(run.output, ('S2@10', 'S0'), n=1)             # done: nothing is asked again
    assert requests_sent(run.output) == asked >= sent
    wider = retention.run(run.output, ('S2@10',), n=2)['conditions']['S2@10']      # another probe count: measured again
    assert wider['n'] == 4 and core.read(Path(run.output) / 'retention_aligned' / 'c2' / 'S2@10.json')['probes'] == \
        order['c2']


def test_a_probe_full_does_not_answer_is_not_a_probe(tmp_path, monkeypatch):
    run = episode(tmp_path)
    entry.prepare(run)
    offline_gateway(monkeypatch, replies())
    entry.execute(run.output)
    full = Path(run.output) / 'claims' / 'c2' / 'Full' / 'native.json'
    row = core.read(full)
    row['result']['recovered'] = False
    write_json(full, row)
    result = retention.run(run.output, ('S2@10',), n=5)['conditions']['S2@10']
    assert result['claims_without_probes'] == ['c1', 'c3'] and result['n'] == 2     # c2 keeps c1 and c3


def test_the_tool_refuses_a_run_without_answers_or_an_unplanned_condition(tmp_path, monkeypatch):
    run = episode(tmp_path)
    entry.prepare(run)
    with pytest.raises(ValueError, match='answer stage first'):
        retention.run(run.output)
    offline_gateway(monkeypatch, replies())
    entry.execute(run.output)
    with pytest.raises(ValueError, match='not in this plan'):
        retention.run(run.output, ('S2@99',))
