"""SimpleMem facts, actual storage snapshots and whole-object source closure."""
import copy
import json
from pathlib import Path
import tempfile
import pickle
import hashlib

from memcut.native_memory.contract import MemoryObject, SearchResult, digest
from memcut.source_anchored_systems.base import SourceBank, SourceObject
from memcut.source_anchored_v2.whole_object_deletion import POLICY
from .client import NativeClient, NativeAbort
from .native import Store, Retriever, native_render, stored_text, MemoryEntry


def address(ident):
    return json.dumps(['simplemem', 'memory_entry', ident], separators=(',', ':'))


class SimpleMemBank(SourceBank):
    system = 'simplemem'

    def __init__(self, settings, turns, *, embedder, retrieval_cache=None, retrieval_model=None):
        super().__init__(settings, turns)
        self.directory = tempfile.TemporaryDirectory(prefix='memcut-simplemem-')
        self.store = Store(self.directory.name, embedder, settings['storage']['table_name'])
        self.retrieval_cache, self.retrieval_model = retrieval_cache, retrieval_model
        self.progress.update(next_window=0, formation_batches=[])

    def native_objects(self):
        result = []
        for row in self.store.table.to_arrow().to_pylist():
            value = {k: v for k, v in row.items() if k != 'vector'}
            entry = MemoryEntry.model_validate(value)
            result.append({'id': address(entry.entry_id), 'native_id': entry.entry_id,
                'layer': 'memory_entry', 'kind': 'aggregate', 'text': stored_text(entry), 'value': value})
        return result

    def dump_native(self):
        return {'rows': self.store.table.to_arrow().to_pylist(),
                'fts_ready': self.store.backend._fts_initialized}

    def read_identity(self):
        # Hash all native columns/vectors, using canonical Arrow bytes instead
        # of converting hundreds of thousands of float32 values to JSON twice
        # for each query. No field or frozen-state check is omitted.
        import pyarrow as pa
        table = self.store.table.to_arrow().combine_chunks()
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
        content = hashlib.sha256(sink.getvalue()).hexdigest()
        return digest({'native_arrow_sha256': content, 'fts_ready': self.store.backend._fts_initialized})

    def load_native(self, value):
        self.store.backend.clear()
        rows = value['rows']
        if rows:
            # Reuse exact saved float32 vectors. No model or embedding request.
            self.store.table.add(copy.deepcopy(rows))
        if value['fts_ready']:
            self.store.backend.build_fts()

    def commit_batch(self, batch):
        if self.frozen or batch['window_index'] != self.progress['next_window']:
            raise ValueError('Formation commits must be ordered and unfrozen')
        entries = [MemoryEntry.model_validate(r['entry'], strict=True) for r in batch['entries']]
        writes = {}
        for row, entry in zip(batch['entries'], entries):
            info = self.lineage.input(row['turns'], request_ids=[batch['request_id']])
            info.update(source_ids=row['source_ids'], source_memory_ids=row['source_memory_ids'],
                contributing_parent_versions=row['parents'], window_index=batch['window_index'])
            writes[address(entry.entry_id)] = info
        backup = pickle.loads(pickle.dumps(self.export(), protocol=5))
        try:
            self.observe_write(lambda: self.store.add_entries(entries), writes,
                               label='simplemem-inline-source-extraction')
            self.progress['formation_batches'].append(copy.deepcopy(batch))
            self.progress['next_window'] += 1
            ranks = {t: i+1 for i, t in enumerate(self.turns)}
            self.progress['next_turn'] = max(self.progress['next_turn'], max(ranks[t] for t in batch['input_turn_ids']))
            self.progress['stage'] = 'FORMING'
        except BaseException:
            self.restore_value(backup)
            raise

    def finish(self):
        if self.progress['next_turn'] != len(self.turns):
            raise ValueError('Cannot freeze incomplete source ingestion')
        self.store.backend.build_fts()
        self.lineage.validate(observed=self.observation())
        self.progress['stage'] = 'FULL'
        self.freeze_updates()

    def erase_native(self, oid):
        self.erase_many([oid])

    def erase_many(self, ids):
        native_ids = []
        for oid in ids:
            system, layer, ident = json.loads(oid)
            if system != self.system or layer != 'memory_entry' or oid not in self.meta['src']:
                raise ValueError('Unknown native memory address')
            native_ids.append(ident)
        if native_ids:
            # Dense/structured use native delete masks. This pinned Tantivy
            # path stores positional doc_ids, so its local index must be rebuilt
            # once per closure transaction after row positions change.
            predicate = 'entry_id IN (' + ','.join(self.store.backend._quote(i) for i in native_ids) + ')'
            self.store.table.delete(predicate)
            self.store.backend.build_fts()

    async def delete_turns(self, selected, completions=None):
        if not self.frozen:
            raise ValueError('Deletion requires an independent frozen Full branch')
        added = set(self.lineage.ordered(selected)) - set(self.meta['deleted_turns'])
        all_selected = added | set(self.meta['deleted_turns'])
        affected = {o for t in added for o in self.reverse_trace(t)}
        backup = pickle.loads(pickle.dumps(self.export(), protocol=5))
        before = self.observation()
        collateral = {t for o in affected for t in self.trace(o)} - all_selected
        diff_start = len(self.meta['mutation_diffs'])
        try:
            if affected:
                self.observe_write(lambda: self.erase_many(sorted(affected)), {}, label=POLICY)
            self.lineage.mark_deleted(all_selected)
            after = self.observation()
            self.lineage.validate(observed=after)
            if set(before)-set(after) != affected or any(after[o] != before[o] for o in after):
                raise ValueError('Whole-object deletion left a target or changed a survivor')
        except BaseException:
            self.restore_value(backup)
            raise
        return {'deletion_policy': POLICY, 'deletion_llm_calls': 0,
            'deleted_turns': self.lineage.ordered(added), 'all_deleted_turns': self.lineage.ordered(all_selected),
            'deleted_objs': [o for o in before if o in affected], 'rewritten_objs': [], 'created_objs': [],
            'collateral_raw_turns': [], 'collateral_source_turns': self.lineage.ordered(collateral),
            'selected_turns_have_empty_der': all(not self.reverse_trace(t) for t in all_selected),
            'mutation_diffs': self.meta['mutation_diffs'][diff_start:]}

    async def regenerate(self, *args, **kwargs):
        raise ValueError('Current SimpleMem protocol never regenerates deleted objects')

    async def search(self, query, top_k=None):
        if not self.frozen:
            raise ValueError('Only a complete frozen bank can enter evaluation')
        if top_k is not None and (type(top_k) is not int or top_k <= 0):
            raise ValueError('Object cap must be a positive integer')
        if self.retrieval_cache is None:
            raise ValueError('Native SimpleMem retrieval needs its frozen planning model')
        before = self.read_identity()
        self.store.calls.clear()
        client = NativeClient(self.retrieval_cache, self.retrieval_model,
                              max_tokens=self.settings['retrieval']['max_tokens'])
        retriever = Retriever(client, self.store, self.settings['retrieval'], semantic_k=top_k,
            reference_time=self.settings['retrieval_reference_time'])
        try:
            # With zero live records every native path is structurally empty.
            all_entries = retriever.retrieve(query) if self.store.backend.count() else []
        except NativeAbort as exc:
            raise RuntimeError('SimpleMem native retrieval failed; no zero score inferred') from exc
        entries = all_entries if top_k is None else all_entries[:top_k]
        live = {o['native_id']: o for o in self.native_objects()}
        if len({e.entry_id for e in entries}) != len(entries) or any(e.entry_id not in live for e in all_entries):
            raise ValueError('Native retrieval contains a duplicate or a deleted fact')
        visible = [MemoryObject(self.system, 'memory_entry', entry.entry_id,
                    native_render([entry]).split('\n', 1)[1], digest(live[entry.entry_id]['value'])) for entry in entries]
        if self.read_identity() != before:
            raise ValueError('Native retrieval mutated the frozen store')
        # Scores are native union positions, not fabricated cross-channel
        # distances. The common candidate reducer consumes negative distance.
        hits = [{'address': o.address, 'rank': i+1, 'distance': float(i)} for i, o in enumerate(visible)]
        return SearchResult(query, {'retrieved_document': visible}, native_context={
            'prompt': native_render(entries), 'entry_ids': [e.entry_id for e in entries]}, costs={
            'native_seed_hits': hits, 'score_semantics': 'negative native ordered-union rank; not vector distance',
            'requested_object_cap': top_k, 'union_before_cap': [e.entry_id for e in all_entries],
            'native_calls': sorted(copy.deepcopy(self.store.calls), key=digest), 'planning_requests': client.requests,
            'prefix_stop': retriever.prefix_stop,
            'union_complete': retriever.prefix_stop is None,
            'retrieval_policy': self.settings['retrieval']['policy']})

    def system_render(self, result):
        return result.native_context['prompt']

    def close(self):
        self.store = None
        self.directory.cleanup()
