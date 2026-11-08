"""tools/extend_episode.py on a small synthetic episode; no model, network or paid call."""
import copy

import pytest
from tools.extend_episode import extend


def base_episode(n_sessions=9):
    sessions = []
    for i in range(1, n_sessions + 1):
        sid = f'S{i:02d}'
        sessions.append({'id': sid, 'timestamp': f'2026-06-{i:02d}T18:00:00+01:00', 'turns': [
            {'id': f'{sid}:T01', 'role': 'user', 'speaker': 'Ada', 'text': f'Ordinary note {i}.'},
            {'id': f'{sid}:T02', 'role': 'assistant', 'speaker': 'Assistant', 'text': 'Noted.'}]})
    src = {'schema': 'memcut.long_episode.source.v1', 'id': 'EPX', 'language': 'en', 'sessions': sessions}
    ann = {'carriers': [], 'direct': [{'key': 'c1-direct', 'claim_id': 'c1', 'record_id': 'S01:T01'}],
           'distractors': []}
    return src, ann


def overlay():
    pair = lambda s, label, kind, text, **kw: {'session': s, 'label': label, 'kind': kind, 'user': text,
                                               'assistant': 'Okay.', **kw}
    turns = [pair('S01', 'c9-e1', 'carrier', 'Kit works at Fern Labs.', edges=['works']),
             pair('S03', 'c9-e2', 'carrier', 'Kit joined Fern Labs in May.', edges=['works']),
             pair('S04', 'c9-e3', 'carrier', 'Fern Labs belongs to Bo.', edges=['owns']),
             pair('S05', 'c9-e4', 'carrier', "Bo's filing lists Fern Labs.", edges=['owns']),
             pair('S06', 'c9-e5', 'carrier', 'Bo is my brother.', edges=['sibling']),
             pair('S07', 'c9-e6', 'carrier', 'My brother Bo called.', edges=['sibling']),
             pair('S09', 'c9-direct', 'direct', "Kit works at my brother Bo's company, Fern Labs.")]
    turns += [pair(f'S{2 + i % 7:02d}', f'c9-d{i + 1}', 'distractor', f'Fern Street parcel {i}.') for i in range(16)]
    return {'schema': 'memcut.case_overlay.v1', 'base_episode': 'EPX', 'variant_episode': 'EPXA',
            'variant_dialogue_id': 'EPX_A', 'world': 'test',
            'leak_patterns': [['Fern Labs', r'\bbrother\b']],
            'claim': {'claim_id': 'c9', 'category': 'relational_linking', 'claim': "Kit works at Bo's firm.",
                      'deletion_request': 'Forget where Kit works.', 'question': 'Where does Kit work?',
                      'fact_definitions': {'works': 'Kit works at Fern Labs.', 'owns': 'Bo owns Fern Labs.',
                                           'sibling': 'Bo is my brother.'},
                      'rules': [{'id': 'r', 'if_all': ['works', 'owns', 'sibling'], 'then': 'target'}],
                      'choices': [{'id': x, 'text': x} for x in 'abcde'], 'target_choice_id': 'b',
                      'insufficient_choice_id': 'a',
                      'rotations': [list('abcde'), list('aebcd'), list('adebc')]},
            'turns': turns}


def test_new_turns_are_appended_and_every_base_turn_is_unchanged():
    src, ann = base_episode()
    out = extend(src, ann, overlay())
    old = [t for s in src['sessions'] for t in s['turns']]
    new = [t for s in out['source']['sessions'] for t in s['turns']]
    assert [t for t in new if t['id'] in {o['id'] for o in old}] == old
    assert len(new) - len(old) == 2 * len(overlay()['turns'])
    assert out['_report']['placed']['c9-e1'] == 'S01:T03'
    assert out['contracts']['claims'][0]['direct_record_ids'] == ['S09:T03']


def test_minimal_sets_and_cuts_follow_the_fact_edges():
    src, ann = base_episode()
    audit = extend(src, ann, overlay())['path_audit']['claims'][0]
    assert audit['audited_minimal_source_combination_count'] == 8          # 2 x 2 x 2
    assert audit['minimum_hitting_set_size_in_audited_family'] == 2
    assert sorted(map(tuple, audit['all_minimum_hitting_sets'])) == [('c9-e1', 'c9-e2'), ('c9-e3', 'c9-e4'),
                                                                     ('c9-e5', 'c9-e6')]


@pytest.mark.parametrize('break_it, message', [
    (lambda o: o['turns'].pop(1), 'every link needs at least two'),
    (lambda o: o['turns'][0].update(edges=['works', 'owns', 'sibling']), 'covers every named fact'),
    (lambda o: o['turns'][7].update(user='Fern Labs hired my brother.'), 'leak pattern'),
    (lambda o: o['turns'][6].update(session='S05'), "shares the disclosure's session"),
    (lambda o: [o['turns'].pop() for _ in range(5)], 'distractors, outside'),
])
def test_a_design_that_breaks_the_construction_guide_is_refused(break_it, message):
    src, ann = base_episode()
    o = copy.deepcopy(overlay())
    break_it(o)
    with pytest.raises(ValueError, match=message):
        extend(src, ann, o)


def test_a_claim_id_the_base_already_uses_is_refused():
    src, ann = base_episode()
    o = overlay()
    o['claim']['claim_id'] = 'c1'
    with pytest.raises(ValueError, match='already has this claim id'):
        extend(src, ann, o)
