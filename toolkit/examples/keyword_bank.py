"""A complete, minimal memory system for MemCut: the template for adding your own.

KeywordBank stores one object per window of consecutive turns (two by default, a user message and its reply) and
retrieves by word overlap. It needs no model, no network and no third-party package, so it shows the whole adapter
contract with nothing else in the way. docs/ADDING_A_SYSTEM.md walks through it; tests/test_example_bank.py runs it
through formation, the S1/S2 candidate pool, source-closure deletion and both readout contexts.

What a system implements (everything else comes from SourceBank):
- form()             build the full memory from self.turns and record, for every object, the turns it came from
- search(q, top_k)   the system's own retrieval; returns a SearchResult with its ranked objects and native scores
- system_render(r)   the text the system would put in front of its reader for a retrieval result
- native_objects()   enumerate what is stored (the auditor's view: residual readout and deletion accounting)
- erase_native(oid)  physically remove one stored object
- dump_native() / load_native(v)   serialize the native store for snapshots and branches
Systems whose objects merge several turns into rewritten text (summaries, profiles) also implement
regenerate_groups(); this example keeps every object atomic, so deleting a turn removes its whole window.
"""
from __future__ import annotations

import copy
import json
import math
import re

from memcut.native_memory.contract import SearchResult, digest
from memcut.source_anchored_systems.base import SourceBank, SourceObject

WORD = re.compile(r"[a-z0-9]+")


def words(text: str) -> set[str]:
    return set(WORD.findall(text.lower()))


class KeywordBank(SourceBank):
    system = 'keyword_window'

    def __init__(self, settings, turns):
        super().__init__({'window': 2, 'native_eval_k': 8, **settings}, turns)
        self.rows: dict[str, dict] = {}

    # --- the native store -------------------------------------------------------------------------------------
    def address(self, native_id: str) -> str:
        return json.dumps([self.system, 'window', native_id], separators=(',', ':'))   # == MemoryObject.address

    def native_objects(self):
        return [{'id': oid, 'native_id': row['native_id'], 'layer': 'window', 'kind': 'atomic',
                 'text': row['text'], 'value': copy.deepcopy(row)} for oid, row in self.rows.items()]

    def dump_native(self):
        return {'rows': copy.deepcopy(self.rows)}

    def load_native(self, value):
        self.rows = copy.deepcopy(value['rows'])

    def erase_native(self, obj_id):
        del self.rows[obj_id]

    # --- formation: every object records the turns it came from ----------------------------------------------
    def form(self):
        if self.rows or self.frozen or self.progress['stage'] != 'NEW':
            raise ValueError('Formation requires a new empty bank')
        ids, size = list(self.turns), self.settings['window']
        rows, writes = {}, {}
        for start in range(0, len(ids), size):
            span = ids[start:start + size]
            native_id = f'w{start // size + 1:04d}'
            oid = self.address(native_id)
            rows[oid] = {'native_id': native_id, 'turn_ids': span,
                         'text': '\n'.join(f"{self.turns[t]['speaker']}: {self.turns[t]['text']}" for t in span)}
            writes[oid] = self.lineage.input(span)                # the provenance S0/S1/S2 deletion follows
        self.observe_write(lambda: setattr(self, 'rows', rows), writes, label='keyword windows stored')
        self.progress.update(stage='FULL', next_turn=len(ids))
        self.lineage.validate(observed=self.observation())
        self.freeze_updates(True)

    # --- retrieval: the system's own ranking, used by S1, S2's pool and the native readout -------------------
    async def search(self, query, *, top_k=20):
        q = words(query)
        scored = []
        for position, (oid, row) in enumerate(self.rows.items()):
            w = words(row['text'])
            score = len(q & w) / math.sqrt(len(w) or 1)
            scored.append((-score, position, oid, score))
        scored.sort()
        objects, seeds = [], []
        for rank, (_, _, oid, score) in enumerate(scored[:top_k], 1):
            row = self.rows[oid]
            objects.append(SourceObject(self.system, 'window', row['native_id'], row['text'], digest(row),
                                        kind='atomic'))
            seeds.append({'address': oid, 'rank': rank, 'distance': -score})   # the pool reads -distance as score
        return SearchResult(query, ranked={'retrieved_document': objects}, native_context={},
                            costs={'native_seed_hits': seeds, 'requested_top_k': top_k, 'llm_calls': 0})

    def system_render(self, result):
        return '\n\n'.join(o.text for o in result.ranked['retrieved_document'])
