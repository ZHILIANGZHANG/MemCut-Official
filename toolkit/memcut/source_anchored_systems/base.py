"""Atomic snapshots and source-closure operations over actual native stores."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import pickle
import os
import math
from dataclasses import dataclass, field

from memcut.native_memory.contract import MemoryObject, SearchResult, digest
from memcut.native_memory.persistence import write_json
from . import VERSION
from .lineage import Lineage, observe_objects


@dataclass(frozen=True)
class SourceObject(MemoryObject):
    kind: str = 'aggregate'
    meta: dict = field(default_factory=dict)

    @property
    def obj_id(self):
        return self.address

    @property
    def obj_type(self):
        return self.layer


class SourceBank:
    """Adapters own native serialization, enumeration, retrieval and generation.

    Source text and formation recipes are research metadata. Neither search nor
    the two readouts may enumerate that metadata as active memory.
    """
    system = None

    def __init__(self, settings, turns=None):
        self.settings = copy.deepcopy(settings)
        self.turns = copy.deepcopy(turns or {})
        self.lineage = Lineage(self.turns)
        self.recipes = {}
        self.progress = {'stage': 'NEW', 'next_turn': 0}
        self.frozen = False

    @property
    def meta(self):
        return {**self.lineage.data, 'turn_order': list(self.turns), 'turns': self.turns}

    def trace(self, oid):
        return set(self.lineage.data['src'].get(oid, []))

    def reverse_trace(self, turn):
        return set(self.lineage.data['der'].get(turn, []))

    def observation(self):
        return observe_objects(self.native_objects())

    async def list_all(self):
        return [SourceObject(self.system, o['layer'], o['native_id'], o['text'], digest(o['value']),
                             o.get('visibility', 'searchable'), o['kind'], copy.deepcopy(o['value']))
                for o in self.native_objects()]

    async def residual_objects(self, turns):
        ids = self.lineage.residual_ids(turns)
        objects = {o.address: o for o in await self.list_all()}
        return [objects[oid] for oid in ids]

    def state_identity(self):
        return digest({'native': self.observation(), 'lineage': self.lineage.data,
                       'frozen': self.frozen, 'settings': self.settings, 'recipes': self.recipes})

    def read_identity(self):
        """Exact within-branch guard, including native vectors and active buffers."""
        return digest(self.dump_native())

    def freeze_updates(self, flag=True):
        if flag and self.progress['stage'] != 'FULL':
            raise ValueError('Cannot freeze unfinished formation')
        self.frozen = bool(flag)

    def export(self):
        return {'format': VERSION, 'system': self.system, 'settings': self.settings,
                'turns': self.turns, 'lineage': self.lineage.data, 'native': self.dump_native(),
                'recipes': self.recipes, 'progress': self.progress, 'frozen': self.frozen}

    def restore_value(self, value):
        if value['format'] != VERSION or value['system'] != self.system or value['settings'] != self.settings:
            raise ValueError('Snapshot system or formation settings differ')
        value = copy.deepcopy(value)
        self.turns, self.recipes, self.progress, self.frozen = (
            value['turns'], value['recipes'], value['progress'], value['frozen'])
        self.lineage = Lineage(self.turns, value['lineage'])
        self.load_native(value['native'])
        def equal(a, b, vector=False):
            if isinstance(a, dict) and isinstance(b, dict):
                return set(a)==set(b) and all(equal(a[k], b[k], vector or k in {'vector','embeddings'}) for k in a)
            if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
                return len(a)==len(b) and all(equal(x,y,vector) for x,y in zip(a,b))
            if vector and isinstance(a, (float,int)) and isinstance(b, (float,int)):
                # Qdrant normalizes float32 cosine vectors on insertion. This
                # tolerance is only for vectors, never text/source metadata.
                return math.isclose(a,b,rel_tol=2e-6,abs_tol=1e-7)
            return a==b
        if not equal(value['native'], self.dump_native()):
            raise ValueError('Native storage or index did not restore faithfully')
        self.lineage.validate(observed=self.observation())

    def snapshot(self, path):
        self.lineage.validate(observed=self.observation())
        path = Path(path)
        if path.exists():
            raise FileExistsError('Snapshots are immutable; choose a new path')
        path.parent.mkdir(parents=True, exist_ok=True)
        blob = pickle.dumps(self.export(), protocol=5)
        temp = path.with_suffix('.pending')
        with temp.open('xb') as f:
            f.write(blob)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
        write_json(path.with_suffix('.frozen.json'), {
            'sha256': hashlib.sha256(blob).hexdigest(), 'state_identity': self.state_identity(),
            'system': self.system, 'format': VERSION})
        return str(path)

    def restore(self, path):
        path = Path(path)
        blob = path.read_bytes()
        seal = json.loads(path.with_suffix('.frozen.json').read_text())
        if hashlib.sha256(blob).hexdigest() != seal['sha256']:
            raise ValueError('Changed snapshot')
        self.restore_value(pickle.loads(blob))
        if self.state_identity() != seal['state_identity']:
            raise ValueError('Restored native state differs from saved state')

    def observe_write(self, operation, inputs, *, rewrite=False, label=None):
        """`inputs` is resolved after execution if new IDs are native-generated."""
        before = self.observation()
        result = operation()
        writes = inputs(result) if callable(inputs) else inputs
        self.lineage.commit(before, self.observation(), writes, rewrite=rewrite, label=label)
        return result

    async def delete(self, obj_id):
        """Low-level whole-object removal. Turn strategies call delete_turns."""
        backup = copy.deepcopy(self.export())
        try:
            if obj_id not in self.lineage.data['src']:
                raise ValueError('Unknown native object')
            self.observe_write(lambda: self.erase_native(obj_id), {}, label='explicit object delete')
        except BaseException:
            self.restore_value(backup)
            raise

    async def regenerate(self, obj_id, src_turn_ids, completions):
        """An aggregate rewrite may share one native generation with siblings."""
        if not self.frozen or self.observation().get(obj_id, {}).get('kind') != 'aggregate':
            raise ValueError('Regeneration requires a live aggregate in a frozen branch')
        current = self.trace(obj_id)
        remaining = set(src_turn_ids)
        if not remaining <= current:
            raise ValueError('Regeneration cannot introduce an unrelated source')
        backup = copy.deepcopy(self.export())
        try:
            excluded = set(self.lineage.data['ever'][obj_id]) - remaining
            self.regenerate_groups({obj_id}, excluded | set(self.meta['deleted_turns']), completions)
            if self.trace(obj_id) - remaining:
                raise ValueError('Regenerated object retained excluded sources')
            self.lineage.validate(observed=self.observation())
        except BaseException:
            self.restore_value(backup)
            raise

    async def delete_turns(self, turns, completions):
        if not self.frozen:
            raise ValueError('Deletion requires a frozen Full-derived branch')
        selected = set(self.lineage.ordered(turns)) - set(self.lineage.data['deleted_turns'])
        all_selected = selected | set(self.lineage.data['deleted_turns'])
        backup = copy.deepcopy(self.export())
        before = self.observation()
        affected = {oid for t in selected for oid in self.lineage.data['der'].get(t, [])}
        atomic = {o['id'] for o in self.native_objects() if o['id'] in affected and o['kind'] == 'atomic'}
        collateral = {t for oid in atomic for t in self.trace(oid)} - all_selected
        event_start = len(self.lineage.data['history'])
        diff_start = len(self.lineage.data['mutation_diffs'])
        try:
            # Atomic erasure must not silently cascade into aggregate deletion.
            if atomic:
                def erase_atomic():
                    for oid in sorted(atomic):
                        self.erase_native(oid)
                self.observe_write(erase_atomic, {}, label='source atomic erase batch')
            if affected - atomic:
                self.regenerate_groups(affected - atomic, all_selected, completions)
            self.lineage.mark_deleted(all_selected)
            self.lineage.validate(observed=self.observation())
        except BaseException:
            # Paid responses remain cached; all native objects/indices and
            # provenance return to the pre-transaction state together.
            self.restore_value(backup)
            raise
        after = self.observation()
        events = self.lineage.data['history'][event_start:]
        return {
            'deleted_turns': self.lineage.ordered(selected), 'all_deleted_turns': self.lineage.data['deleted_turns'],
            'deleted_objs': [oid for oid in before if oid not in after],
            'rewritten_objs': list(dict.fromkeys(e['obj_id'] for e in events if e['op'] == 'rewrite')),
            'created_objs': [oid for oid in after if oid not in before],
            'collateral_raw_turns': self.lineage.ordered(collateral),
            'mutation_diffs': self.lineage.data['mutation_diffs'][diff_start:],
            'selected_turns_have_empty_der': all(not self.reverse_trace(t) for t in all_selected),
            'before_sha256': digest(before), 'after_sha256': digest(after)}

    async def native_context(self, query):
        before = self.state_identity()
        result = await self.search(query, top_k=self.settings['native_eval_k'])
        result.native_context['prompt'] = self.system_render(result)
        if self.state_identity() != before:
            raise ValueError('Native retrieval mutated the frozen memory')
        return result

    def close(self):
        pass
