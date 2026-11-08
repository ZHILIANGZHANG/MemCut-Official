"""Incremental native write diffs, with independent full checkpoint validation.

Only the append-only ingestion path uses these scoped diffs. Frozen reads,
deletion, restoration and checkpoints retain the original full-store audit.
"""
import copy
from collections import defaultdict

from memcut.native_memory.contract import digest
from memcut.source_anchored_v2.provenance import diff_objects
from ..lineage import Lineage, observe_objects


class DeltaLineage(Lineage):
    def commit_delta(self, before, after, writes, *, label, order_key):
        d = self.data
        scope = set(before) | set(after)
        if {oid: d['observed'][oid] for oid in scope if oid in d['observed']} != before:
            raise ValueError('Native write-set differs from committed observation')
        diff = diff_objects(before, after)
        if not set(diff['created'] + diff['modified']) <= writes.keys() or not writes.keys() <= after.keys():
            raise ValueError('Unattributed native write')
        # Prepare all fallible data before mutating the ledger.
        step = len(d['mutation_diffs'])
        prepared = {}
        for oid in sorted(writes, key=order_key):
            info = writes[oid]
            roots = set(self.ordered(info['turns'])) | set(d['src'].get(oid, ()))
            if not roots or roots & set(d['deleted_turns']):
                raise ValueError('Missing or deleted native source')
            prior = set(d['ever'].get(oid, ()))
            if oid not in before and oid in d['birth']:
                prior.update(t for e in d['history'] if e['obj_id'] == oid for t in e['turns'])
            prepared[oid] = (self.ordered(roots), self.ordered(prior | roots), copy.deepcopy(info))
        record = {'step': step, 'label': label, 'scope': 'native-write-set-v1',
            'scope_ids': sorted(scope),
            'before_sha256': digest(before), 'after_sha256': digest(after), **diff,
            'unchanged_writes': [oid for oid in writes if oid in before and before[oid] == after[oid]],
            'before_objects': {oid: before[oid] for oid in diff['modified'] + diff['deleted']},
            'after_objects': {oid: after[oid] for oid in diff['created'] + diff['modified']}}
        affected = set(writes) | set(diff['deleted'])
        reverse = {}
        for forward, backward, index in [('src', 'der', 0), ('ever', 'touched', 1)]:
            additions = defaultdict(list)
            touched = {t for oid in affected for t in d[forward].get(oid, ())}
            for oid, values in prepared.items():
                for t in values[index]:
                    touched.add(t)
                    additions[t].append(oid)
            reverse[backward] = {t: sorted(
                [oid for oid in d[backward].get(t, ()) if oid not in affected] + additions[t],
                key=order_key) for t in touched}
        for oid in diff['deleted']:
            d['history'].append({'op': 'delete', 'obj_id': oid, 'turns': d['src'].pop(oid), 'step': step})
            d['ever'].pop(oid)
            d['observed'].pop(oid)
        for oid, (roots, ever, info) in prepared.items():
            d['src'][oid], d['ever'][oid] = roots, ever
            if oid not in d['birth']:
                d['birth'][oid] = len(d['birth'])
            d['history'].append({'op': 'modify' if oid in before else 'create',
                'obj_id': oid, 'turns': roots, 'step': step, 'inputs': info,
                'text_unchanged': oid in before and before[oid] == after[oid]})
        d['observed'].update(copy.deepcopy(after))
        for backward, rows in reverse.items():
            for t, ids in rows.items():
                if ids:
                    d[backward][t] = ids
                else:
                    d[backward].pop(t, None)
        d['mutation_diffs'].append(record)


class DeltaBank:
    def _delta(self, before, after, writes, label):
        if not isinstance(self.lineage, DeltaLineage):
            self.lineage = DeltaLineage(self.turns, self.lineage.data)
        self.lineage.commit_delta(observe_objects(before), observe_objects(after), writes,
                                  label=label, order_key=self._order_key)


def replay_diffs(records):
    """Verify full and incremental histories without silently weakening either."""
    state = {}
    for change in records:
        scoped = change.get('scope') == 'native-write-set-v1'
        if change.get('scope') not in (None, 'native-write-set-v1'):
            raise ValueError('Unknown native diff scope')
        def view():
            return {k: state[k] for k in change['scope_ids'] if k in state} if scoped else state
        if digest(view()) != change['before_sha256']:
            raise ValueError('Native diff history has an invalid before state')
        if any(state.get(oid) != old for oid, old in change['before_objects'].items()):
            raise ValueError('Native diff history changed a prior object')
        for oid in change['deleted']:
            state.pop(oid)
        state.update(change['after_objects'])
        if digest(view()) != change['after_sha256']:
            raise ValueError('Native diff history has an invalid after state')
    return state
