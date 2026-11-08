"""Raw-turn and Session RAG on the aligned entry: the same transport policy, unavailable readouts and answer cap
as the four memory systems, and Session RAG's coverage gate before any paid request. Offline fixtures only:
synthetic embeddings and tokenizer, scripted replies."""
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace

import pytest

from aligned_fixtures import FixtureGateway
from memcut.aligned_s012.rag import CoverageReview
from memcut.native_memory.persistence import write_json
from memcut.raw_turn_rag import runner as raw
from memcut.session_rag import runner as session
from memcut.source_anchored_v2 import runner as core
from test_session_rag import FixtureEmbedding, WordTokenizer
from tools import memcut_s012 as entry


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv('MEMCUT_PAID_LOCK', str(tmp_path / 'paid.lock'))
    # Synthetic local encoders instead of the pinned MiniLM and its tokenizer; no package versions frozen.
    raw_factory, session_factory = raw.Factory, session.Factory
    monkeypatch.setattr(raw, 'Factory', lambda plan, **_: raw_factory(plan, embedder=FixtureEmbedding(384)))
    monkeypatch.setattr(session, 'Factory', lambda plan, **_: session_factory(
        plan, embedder=FixtureEmbedding(384), tokenizer=WordTokenizer()))
    monkeypatch.setattr(session, 'runtime_identity', lambda: {'fixture-runtime': '1'})


def episode(tmp_path):
    turns = [{'id': f't{i}', 'role': 'user' if i % 2 else 'assistant', 'speaker': 'Mira',
              'text': f'Rank {i}: unchanged source text {i}.'} for i in range(1, 41)]
    source = tmp_path / 'source.json'
    write_json(source, {'id': 'fixture', 'sessions': [
        {'id': f's{j}', 'timestamp': f'2026-01-0{j + 1}T12:00:00', 'turns': turns[j * 10:(j + 1) * 10]}
        for j in range(4)]})
    claim = {'claim_id': 'c1', 'category': 'factual_composition', 'stem': 'Which option?',
             'deletion_request': 'Forget the private information.', 'target_choice_id': 'yes',
             'choices': [{'id': 'yes', 'text': 'Known option'}] + [{'id': f'n{i}', 'text': f'Wrong {i}'} for i in range(3)]}
    write_json(tmp_path / 'contracts.json', {'source_sha256': core.sha(source), 'claims': [
        claim, {**claim, 'claim_id': 'c2', 'category': 'relational_linking'}]})

    def annotation(cid, i):
        turn = turns[i - 1]
        return {'claim_id': cid, 'record_id': turn['id'], 'role': turn['role'],
                'text_sha256': hashlib.sha256(turn['text'].encode()).hexdigest()}
    write_json(tmp_path / 'annotations.json', {'source_sha256': core.sha(source),
        'direct': [annotation('c1', 30), annotation('c2', 31)],
        'carriers': [annotation(c, i) for c in ('c1', 'c2') for i in (1, 2)],
        'distractors': [annotation(c, 29) for c in ('c1', 'c2')]})
    return source, tmp_path / 'contracts.json', tmp_path / 'annotations.json'


def args(tmp_path, system, **options):
    source, contracts, annotations = episode(tmp_path)
    return SimpleNamespace(**{'system': system, 'output': str(tmp_path / f'{system}-run'), 'source': str(source),
                              'contracts': str(contracts), 'annotations': str(annotations),
                              'model': 'offline-fixture', 'endpoint': 'offline-fixture', 'claims': None,
                              'native_eval_k': None, 'accept_coverage': False, **options})


def replies(cut_off=()):
    """Selector scores and multiple-choice answers; the answer requests numbered in `cut_off` are cut off."""
    answered = []

    def create(**request):
        system, body = request['messages'][0]['content'], request['messages'][-1]['content']
        finish = 'stop'
        if 'Score candidate turns for a targeted memory deletion' in system:
            payload = json.loads(body)
            content = {'scores': [{'id': c['id'], 'score': 10 * i / max(1, len(payload['candidates']) - 1),
                                   'reason': 'Fixture reverse ranking.'} for i, c in enumerate(payload['candidates'])]}
        else:
            assert request['max_tokens'] == 2048               # the aligned cap, not the frozen 400 or 1024
            answered.append(request)
            content = {'reason': 'Fixture.', 'answer': re.search(r'^([a-e])\. Known option$', body, re.M)[1]}
            if len(answered) in cut_off:
                finish = 'length'
        return {'choices': [{'finish_reason': finish, 'message': {'content': json.dumps(content)}}],
                'usage': {'prompt_tokens': 3, 'completion_tokens': 2}}
    create.answered = answered
    return create


def offline_gateway(monkeypatch, create):
    monkeypatch.setattr(core, 'gateway', lambda folder, plan: (FixtureGateway(Path(folder), create),) * 2)


def requests_sent(folder):
    return len(list((Path(folder) / 'requests').glob('*/response.json')))


def test_turnrag_runs_every_stage_under_the_aligned_policies(tmp_path, monkeypatch):
    run = args(tmp_path, 'turnrag')
    plan = entry.prepare(run)
    assert plan['system'] == 'raw_turn_rag' and plan['settings']['native_eval_k'] == 16
    assert plan['alignment']['system'] == 'turnrag' and plan['aligned_answer_policy']['max_tokens'] == 2048
    create = replies()
    offline_gateway(monkeypatch, create)
    summary = entry.execute(run.output)
    assert core.read(Path(run.output) / 'STATUS.json') == {'stage': 'run', 'state': 'complete'}
    for condition in plan['conditions']:
        assert summary['metrics'][f'residual-{condition}']['completed_claims'] == 2
    assert summary['available_cohort']['families']['native']['claims'] == 2
    assert not (Path(run.output) / 'full/DIAGNOSTICS.json').exists()     # nothing to diagnose: every turn is stored
    sent = requests_sent(run.output)
    entry.execute(run.output)                       # a finished run returns its summary and sends nothing
    assert requests_sent(run.output) == sent


def test_turnrag_unusable_answer_is_unavailable_and_the_run_finishes(tmp_path, monkeypatch):
    run = args(tmp_path, 'turnrag')
    entry.prepare(run)
    offline_gateway(monkeypatch, replies(cut_off={3}))
    summary = entry.execute(run.output)
    gone = list(Path(run.output).glob('claims/*/*/*.unavailable.json'))
    assert len(gone) == 1 and core.read(gone[0])['unavailable']['finish_reason'] == 'length'
    cohort = summary['available_cohort']
    assert len(cohort['unavailable_readouts']) == 1
    family = core.read(gone[0])['family']
    assert cohort['families'][family]['claims'] == 1                    # the claim is left out, never scored 0


def test_sessionrag_stops_before_any_paid_request_when_its_context_covers_the_episode(tmp_path, monkeypatch):
    run = args(tmp_path, 'sessionrag')               # k16 over four sessions: the whole episode
    plan = entry.prepare(run)
    assert plan['qa_max_tokens'] == 2048 and plan['early_stop'] is False
    assert session.load_plan(run.output)['aligned_options'] == {'accept_coverage': False}
    offline_gateway(monkeypatch, replies())
    with pytest.raises(CoverageReview, match='No paid request was sent'):
        entry.execute(run.output)
    folder = Path(run.output)
    assert requests_sent(folder) == 0 and core.read(folder / 'STATUS.json')['state'] == 'stopped'
    assert core.read(folder / 'COVERAGE.json')['summary'][0]['above_25_percent']
    assert all((folder / 'claims' / c / s / 'STATE.json').exists() for c in ('c1', 'c2') for s in ('S0', 'S1@10'))
    assert not list(folder.glob('claims/*/S2@10'))


def test_sessionrag_accepted_coverage_runs_to_a_summary_with_its_coverage(tmp_path, monkeypatch):
    run = args(tmp_path, 'sessionrag', accept_coverage=True)
    entry.prepare(run)
    create = replies()
    offline_gateway(monkeypatch, create)
    summary = entry.execute(run.output)
    assert summary['metrics']['residual-S2@10']['completed_claims'] == 2
    assert summary['coverage'][0]['k'] == 16 and summary['available_cohort']['families']['residual']['claims'] == 2
    assert create.answered and requests_sent(run.output) > len(create.answered)    # answers and the S2 selector


@pytest.mark.parametrize('system, option', [('lightmem', {'native_eval_k': 4}), ('turnrag', {'accept_coverage': True}),
                                            ('memorybank', {'accept_coverage': True})])
def test_control_only_options_are_refused_for_other_systems(tmp_path, system, option):
    run = args(tmp_path, system, **option)
    with pytest.raises(ValueError, match='turnrag|sessionrag'):
        entry.prepare(run)
    assert not Path(run.output).exists()
