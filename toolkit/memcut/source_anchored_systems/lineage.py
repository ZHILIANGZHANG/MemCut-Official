"""Independent native snapshots and input ancestry, without an LLM judge."""
from __future__ import annotations

import copy
from collections import defaultdict

from memcut.native_memory.contract import digest
from memcut.source_anchored_v2.provenance import diff_objects


def observe_objects(objects):
    """The caller must enumerate storage, independently of the source table.

    Each object contains text plus its complete native stored value. Metadata
    changes are therefore visible even when neither text nor index changed.
    """
    result = {}
    for obj in objects:
        oid = obj['id']
        if oid in result or obj['kind'] not in {'atomic', 'aggregate'}:
            raise ValueError('Duplicate object ID or unknown deletion kind')
        result[oid] = {'text_sha256': digest(obj['text']),
                       'stored_sha256': digest(obj['value']), 'kind': obj['kind']}
    return result


class Lineage:
    """Transactional provenance. Current sources and historical contact differ.

    A write explicitly names the actual input turns and/or *input versions*.
    The snapshot diff decides which objects appeared, changed or disappeared.
    Regeneration replaces current sources; ordinary writes union old sources.
    """

    def __init__(self, turns, value=None):
        self.turns = tuple(turns)
        self.turn_rank = {t: i for i, t in enumerate(self.turns)}
        if len(self.turns) != len(self.turn_rank):
            raise ValueError('Duplicate source turn')
        self.data = copy.deepcopy(value) if value is not None else {
            'src': {}, 'der': {}, 'ever': {}, 'touched': {}, 'birth': {},
            'history': [], 'mutation_diffs': [], 'observed': {}, 'deleted_turns': []}

    def ordered(self, values):
        values = set(values)
        if not values <= self.turn_rank.keys():
            raise ValueError('Unknown input source turn')
        return sorted(values, key=self.turn_rank.__getitem__)

    def input(self, turns=(), parents=(), request_ids=()):
        """Capture before a native write, while all input parents still exist."""
        roots = set(turns)
        versions = []
        for oid in parents:
            roots.update(self.data['src'][oid])
            versions.append({'id': oid, 'snapshot': copy.deepcopy(self.data['observed'][oid]),
                             'turns': list(self.data['src'][oid])})
        return {'turns': self.ordered(roots), 'parents': versions,
                'request_ids': list(request_ids)}

    def commit(self, before, after, writes=None, *, rewrite=False, label=None, replace_sources=()):
        """Validate fully before replacing the ledger; unknown writes fail closed.

        `writes` also contains intentional same-text regeneration, which a hash
        difference alone cannot detect. Deleted objects need no invented input.
        """
        if before != self.data['observed']:
            raise ValueError('Native state changed outside the observed transaction')
        writes = writes or {}
        replace_sources = set(replace_sources)
        if not replace_sources <= writes.keys():
            raise ValueError('Source replacement requires an attributed write')
        diff = diff_objects(before, after)
        changed = set(diff['created'] + diff['modified'])
        if not changed <= set(writes) or not set(writes) <= set(after):
            raise ValueError('Unattributed native write or write to absent object')
        # Existing records are immutable. Copy their containers, not every
        # historical payload on every original turn (quadratic formation cost).
        new = {k: copy.copy(v) for k, v in self.data.items()}
        step = len(new['mutation_diffs'])
        for oid in diff['deleted']:
            new['history'].append({'op': 'delete', 'obj_id': oid,
                                   'turns': new['src'].pop(oid), 'step': step})
            new['ever'].pop(oid)
        positions = {oid: i for i, oid in enumerate(after)}
        deleted_turns = set(new['deleted_turns'])
        for oid in sorted(writes, key=positions.__getitem__):
            info = writes[oid]
            roots = set(self.ordered(info['turns']))
            if not roots:
                raise ValueError('A live derived object must have actual input sources')
            old = set(new['src'].get(oid, []))
            if oid in before and not rewrite and oid not in replace_sources:
                roots |= old
            if roots & deleted_turns:
                raise ValueError('A removed source was reintroduced')
            new['src'][oid] = self.ordered(roots)
            prior = set(new['ever'].get(oid, []))
            if oid not in before and oid in new['birth']:
                # A native output slot can disappear then reappear. touched is
                # defined by history of that address, not just its last life.
                prior.update(t for e in new['history'] if e['obj_id'] == oid for t in e['turns'])
            new['ever'][oid] = self.ordered(prior | old | roots)
            if oid not in new['birth']:
                new['birth'][oid] = len(new['birth'])
            new['history'].append({
                'op': 'create' if oid not in before else ('rewrite' if rewrite else 'modify'),
                'obj_id': oid, 'turns': new['src'][oid], 'step': step,
                'inputs': copy.deepcopy(info), 'text_unchanged': oid in before and before[oid] == after[oid]})
            if oid in replace_sources:
                new['history'][-1]['sources_replaced'] = True
        new['observed'] = copy.deepcopy(after)
        new['mutation_diffs'].append({
            'step': step, 'label': label, 'before_sha256': digest(before), 'after_sha256': digest(after),
            **diff, 'unchanged_writes': [oid for oid in writes if oid in before and before[oid] == after[oid]],
            'before_objects': {oid: before[oid] for oid in diff['modified'] + diff['deleted']},
            'after_objects': {oid: after[oid] for oid in diff['created'] + diff['modified']}})
        # Unchanged objects retain their validated edges. Only replace the
        # edges incident to changed/deleted objects; preserve native order.
        affected = set(writes) | set(diff['deleted'])
        for forward, reverse in [('src', 'der'), ('ever', 'touched')]:
            changed_turns = {t for oid in affected
                             for d in (self.data, new) for t in d[forward].get(oid, ())}
            additions = defaultdict(list)
            for oid in writes:
                for t in new[forward][oid]:
                    additions[t].append(oid)
            for t in changed_turns:
                ids = [oid for oid in self.data[reverse].get(t, ()) if oid not in affected]
                ids.extend(additions[t])
                if ids:
                    new[reverse][t] = sorted(ids, key=positions.__getitem__)
                else:
                    new[reverse].pop(t, None)
            if ([oid for oid in before if oid in after] !=
                    [oid for oid in after if oid in before]):
                # Some native stores reorder existing addresses after a write.
                new[reverse] = {t: sorted(ids, key=positions.__getitem__)
                                for t, ids in new[reverse].items()}
        if set(after) != set(new['src']) or set(after) != set(new['ever']):
            raise ValueError('Storage and provenance disagree')
        # All new roots/history are checked above. Full independent validation
        # remains mandatory at restore, checkpoint, freeze and deletion commit.
        self.data = new
        return copy.deepcopy(new['mutation_diffs'][-1])

    def validate(self, value=None, observed=None):
        d = self.data if value is None else value
        actual = d['observed'] if observed is None else observed
        if actual != d['observed'] or set(actual) != set(d['src']) or set(actual) != set(d['ever']):
            raise ValueError('Storage and provenance disagree')
        der, touched = defaultdict(set), defaultdict(set)
        deleted_turns = set(d['deleted_turns'])
        for oid, roots in d['src'].items():
            if not roots or set(roots) - set(d['ever'][oid]):
                raise ValueError('Lost source history')
            if set(roots) & deleted_turns:
                raise ValueError('Deleted turn remains in active ancestry')
            self.ordered(d['ever'][oid])
            for t in roots:
                der[t].add(oid)
            for t in d['ever'][oid]:
                touched[t].add(oid)
        if dict(der) != {t: set(v) for t, v in d['der'].items()}:
            raise ValueError('src/der invariant failed')
        if dict(touched) != {t: set(v) for t, v in d['touched'].items()}:
            raise ValueError('touched must include exactly the live historical objects')

    def mark_deleted(self, selected):
        selected = self.ordered(set(selected) | set(self.data['deleted_turns']))
        if any(self.data['der'].get(t) for t in selected):
            raise ValueError('Cannot commit an incomplete deletion closure')
        self.data['deleted_turns'] = selected
        self.validate()

    def residual_ids(self, turns):
        ids = {oid for t in turns for k in ('der', 'touched') for oid in self.data[k].get(t, [])}
        return sorted(ids, key=self.data['birth'].__getitem__)
