"""Exact vector retrieval and physical row deletion over raw turn objects."""
import copy
import json

import numpy as np

from memcut.native_memory.contract import SearchResult, digest
from memcut.source_anchored_systems.base import SourceBank, SourceObject
from memcut.source_anchored_v2.whole_object_deletion import POLICY
from .embedding import validate_vectors, with_prefix


class RawTurnBank(SourceBank):
    system = 'raw_turn_rag'

    def __init__(self, settings, turns, embedder):
        super().__init__(settings, turns)
        self.embedder = embedder
        self.rows = {}

    def native_objects(self):
        return [{'id': oid, 'native_id': row['turn_id'], 'layer': 'turn', 'kind': 'atomic',
                 'text': row['text'], 'value': copy.deepcopy(row)} for oid, row in self.rows.items()]

    def dump_native(self):
        return {'rows': copy.deepcopy(self.rows)}

    def load_native(self, value):
        rows = copy.deepcopy(value['rows'])
        for oid, row in rows.items():
            expected = json.dumps([self.system, 'turn', row['turn_id']], separators=(',', ':'))
            if oid != expected or row['turn_id'] not in self.turns or row['text'] != self.turns[row['turn_id']]['text']:
                raise ValueError('Stored object is not its original turn')
            if self.trace(oid) != {row['turn_id']}:
                raise ValueError('Raw-turn objects require exactly their own source turn')
        validate_vectors([r['vector'] for r in rows.values()] if rows else np.empty((0, self.settings['embedding']['dimensions'])),
                         len(rows), self.settings['embedding']['dimensions'])
        self.rows = rows

    def form(self):
        if self.rows or self.frozen or self.progress['stage'] != 'NEW':
            raise ValueError('Formation requires a new empty bank')
        texts = with_prefix(self.settings['embedding'], [t['text'] for t in self.turns.values()], 'document')
        vectors = validate_vectors(self.embedder.encode(texts),
                                   len(self.turns), self.settings['embedding']['dimensions'])
        rows, writes = {}, {}
        for (tid, turn), vector in zip(self.turns.items(), vectors):
            oid = json.dumps([self.system, 'turn', tid], separators=(',', ':'))
            rows[oid] = {'turn_id': tid, 'text': turn['text'], 'vector': vector.tolist()}
            writes[oid] = self.lineage.input([tid])
        self.observe_write(lambda: setattr(self, 'rows', rows), writes, label='raw turns stored; no generation')
        self.progress.update(stage='FULL', next_turn=len(self.turns))
        self.lineage.validate(observed=self.observation())
        self.freeze_updates(True)

    async def search(self, query, *, top_k=20):
        if type(top_k) is not int or top_k <= 0:
            raise ValueError('top_k must be positive')
        ids = list(self.rows)
        if ids:
            q = validate_vectors(self.embedder.encode(with_prefix(self.settings['embedding'], [query], 'query')), 1,
                                 self.settings['embedding']['dimensions'])[0]
            matrix = np.asarray([self.rows[oid]['vector'] for oid in ids], dtype=np.float32)
            # Avoid platform BLAS floating-point flags; accumulate deterministically.
            scores = np.einsum('ij,j->i', matrix, q, dtype=np.float64, optimize=False)
            if not np.isfinite(scores).all():
                raise ValueError('Non-finite cosine score')
            order = np.argsort(-scores, kind='stable')[:top_k]
        else:
            order, scores = [], []
        objects, seeds = [], []
        for rank, index in enumerate(order, 1):
            oid, score = ids[index], float(scores[index])
            row = self.rows[oid]
            obj = SourceObject(self.system, 'turn', row['turn_id'], row['text'], digest(row),
                               kind='atomic', meta={'turn_id': row['turn_id']})
            objects.append(obj)
            # Shared candidate code consumes -distance as the relevance score.
            seeds.append({'address': oid, 'rank': rank, 'distance': -score, 'cosine': score})
        return SearchResult(query, ranked={'retrieved_document': objects}, native_context={},
            costs={'native_seed_hits': seeds, 'requested_top_k': top_k, 'actual_hits': len(objects),
                   'retrieval': self.settings['retrieval'], 'llm_calls': 0})

    def system_render(self, result):
        return '\n'.join(o.text for o in result.ranked['retrieved_document'])

    def erase_native(self, obj_id):
        del self.rows[obj_id]  # Text and vector leave the live index together.

    async def delete_turns(self, selected):
        if not self.frozen:
            raise ValueError('Deletion requires a frozen Full-derived branch')
        added = set(self.lineage.ordered(selected)) - set(self.meta['deleted_turns'])
        affected = {oid for t in added for oid in self.reverse_trace(t)}
        backup, before = self.export(), self.observation()
        start = len(self.meta['mutation_diffs'])
        try:
            if affected:
                def erase():
                    for oid in sorted(affected):
                        self.erase_native(oid)
                self.observe_write(erase, {}, label=POLICY)
            self.lineage.mark_deleted(added)
            after = self.observation()
            if set(before) - set(after) != affected or any(after[o] != before[o] for o in after):
                raise ValueError('Deletion changed unselected objects')
            self.lineage.validate(observed=after)
        except BaseException:
            self.restore_value(backup)
            raise
        return {'deletion_policy': POLICY, 'deletion_llm_calls': 0,
                'selected_turns': self.lineage.ordered(added), 'all_deleted_turns': self.meta['deleted_turns'],
                'deleted_objs': [o for o in before if o in affected], 'created_objs': [], 'rewritten_objs': [],
                'collateral_source_turns': [], 'collateral_raw_turns': [],
                'selected_turns_have_empty_der': all(not self.reverse_trace(t) for t in self.meta['deleted_turns']),
                'mutation_diffs': self.meta['mutation_diffs'][start:],
                'before_sha256': digest(before), 'after_sha256': digest(after)}
