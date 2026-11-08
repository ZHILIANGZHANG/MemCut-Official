"""LightMem flat native pipeline with versioned source attribution."""
from __future__ import annotations

from contextlib import contextmanager
import copy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import uuid

from memcut.native_memory.contract import MemoryObject, SearchResult, digest
from memcut.native_memory.lightmem_v1 import native_modules, payload_record, UPSTREAM
from memcut.native_memory.lightmem_formation_v1 import settings as native_settings
from memcut.source_anchored_v2.completions import parse_json
from .base import SourceBank
from .nemori import NativeFailure
from .lightmem_sources import POLICY, EXTRACTION_INSTRUCTION, UPDATE_INSTRUCTION, fact_sources, numbers


class LightMemBank(SourceBank):
    system = 'lightmem'

    def __init__(self, settings, turns=None, *, assets, shared=None):
        super().__init__(settings, turns)
        self.Qdrant, self.QConfig, self.Retriever = native_modules()
        from lightmem.memory import lightmem as lm
        import tiktoken
        self.lm, self.assets = lm, assets
        self.temp = tempfile.TemporaryDirectory(prefix='memcut-lightmem-')
        cfg = native_settings(self.temp.name, assets, model=settings['model'], base_url='https://unused.invalid')
        cfg['memory_manager']['configs']['temperature'] = 0
        if shared is None:
            self.memory = lm.LightMemory.from_config(cfg)
            self.shared = self.memory
        else:
            # Share only model weights. Every mutable buffer, manager, store,
            # config and provenance table belongs to this branch.
            self.memory = copy.copy(shared)
            self.memory.config = copy.deepcopy(shared.config)
            self.memory.manager = copy.copy(shared.manager)
            self.memory.senmem_buffer_manager = copy.copy(shared.senmem_buffer_manager)
            self.memory.shortmem_buffer_manager = copy.copy(shared.shortmem_buffer_manager)
            self.memory.senmem_buffer_manager.buffer = []
            self.memory.senmem_buffer_manager.big_buffer = []
            self.memory.shortmem_buffer_manager.buffer = []
            self.memory.senmem_buffer_manager.token_count = self.memory.shortmem_buffer_manager.token_count = 0
            self.memory.token_stats = {k: 0 for k in shared.token_stats}
            for name in ('embedding_retriever', 'summary_retriever'):
                collection = 'details' if name == 'embedding_retriever' else 'summaries'
                store = self.Qdrant(self.QConfig(collection_name=collection,
                    path=str(Path(self.temp.name)/collection), embedding_model_dims=384, on_disk=True))
                setattr(self.memory, name, store)
            self.shared = shared
        self.memory.shortmem_buffer_manager.tokenizer = tiktoken.get_encoding('o200k_base')
        spec = importlib.util.spec_from_file_location('memcut_lightmem_s012_prompts', UPSTREAM/'experiments/locomo/prompts.py')
        self.prompts = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.prompts)
        spec = importlib.util.spec_from_file_location('memcut_lightmem_s012_render', UPSTREAM/'experiments/locomo/retrievers.py')
        self.render = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.render)
        self.completions, self.request_ids = None, []
        self.request_by_messages = {}
        self.generation_failures = []
        self.dispatch_lock = threading.Lock()
        self.memory.manager.generate_response = self._generate
        self.topic_index = 0
        self.recipes = {'batches': {}, 'updates': {}, 'full_live_ids': []}

    @property
    def fine_sources(self):
        return self.settings.get('source_policy') == POLICY

    def extraction_prompt(self):
        prompt = self.prompts.METADATA_GENERATE_PROMPT_locomo
        return prompt + EXTRACTION_INSTRUCTION if self.fine_sources else prompt

    @staticmethod
    def oid(layer, ident):
        return json.dumps(['lightmem', layer, str(ident)], separators=(',', ':'))

    def _generate(self, messages, response_format=None, tools=None, **kwargs):
        if tools:
            raise NativeFailure('Unexpected native generation tools')
        with self.dispatch_lock:
            try:
                result = self.completions.complete({'model': self.settings['model'], 'messages': messages,
                    'temperature': 0, 'top_p': .1, 'max_tokens': self.settings['generation_max_tokens'],
                    **({'response_format': response_format} if response_format else {})})
                self.request_ids.append(result['request_id'])
                self.request_by_messages[digest(messages)] = result['request_id']
                data = parse_json(result['text'])
                if 'data' in data:
                    facts = data['data']
                    if not isinstance(facts, list) or not all(isinstance(f, dict) and
                        isinstance(f.get('fact'), str) and str(f.get('source_id', '')).isdigit() for f in facts):
                        raise ValueError('Malformed native extracted facts')
                elif data.get('action') not in {'ignore', 'update', 'delete'}:
                    raise ValueError('Malformed native consolidation action')
                if data.get('action') == 'update' and not isinstance(data.get('new_memory'), str):
                    raise ValueError('Missing consolidated text')
                return result['text'], {}
            except Exception as exc:
                self.generation_failures.append(type(exc).__name__)
                raise NativeFailure(type(exc).__name__) from None

    def rows(self, *, with_vectors=True):
        return self.memory.embedding_retriever.get_all(with_vectors=with_vectors)

    def _buffers(self):
        sen, short = self.memory.senmem_buffer_manager, self.memory.shortmem_buffer_manager
        return {'sensory_buffer': sen.buffer, 'sensory_big_buffer': sen.big_buffer,
                'short_buffer': [m for segment in short.buffer for m in segment]}

    def native_objects(self):
        result = []
        for row in self.rows(with_vectors=False):
            result.append({'id': self.oid('long_term', row['id']), 'layer': 'long_term', 'native_id': str(row['id']),
                'kind': 'aggregate', 'text': json.dumps(payload_record(row['payload']), ensure_ascii=False, separators=(',', ':')),
                'value': row['payload']})
        if self.memory.summary_retriever.get_all(with_vectors=False):
            raise ValueError('Unexpected extra layer in pinned flat LightMem')
        for layer, messages in self._buffers().items():
            for m in messages:
                # Empty synthetic assistant placeholders carry no user text.
                if not m.get('content'):
                    continue
                tid = m['_memcut_source_id']
                ident = f"{tid}/{m['role']}"
                result.append({'id': self.oid(layer, ident), 'layer': layer, 'native_id': ident,
                    'kind': 'atomic', 'text': m['content'], 'value': m, 'visibility': 'stored_only'})
        return result

    def dump_native(self):
        m = self.memory
        return {'rows': self.rows(), 'sensory': {k: copy.deepcopy(getattr(m.senmem_buffer_manager, k))
                    for k in ('buffer', 'big_buffer', 'token_count')},
                'short': {k: copy.deepcopy(getattr(m.shortmem_buffer_manager, k)) for k in ('buffer', 'token_count')},
                'topic_index': self.topic_index, 'token_stats': copy.deepcopy(m.token_stats)}

    def load_native(self, value):
        store = self.memory.embedding_retriever
        for row in self.rows(with_vectors=False):
            store.delete(str(row['id']))
        if value['rows']:
            self._insert(value['rows'])
        for group, manager in [('sensory', self.memory.senmem_buffer_manager), ('short', self.memory.shortmem_buffer_manager)]:
            for k, v in value[group].items():
                setattr(manager, k, copy.deepcopy(v))
        self.topic_index = value['topic_index']
        self.memory.token_stats = copy.deepcopy(value['token_stats'])

    def _insert(self, rows):
        self.memory.embedding_retriever.insert(ids=[str(r['id']) for r in rows],
            vectors=[r['vector'] for r in rows], payloads=[r['payload'] for r in rows])

    def erase_native(self, oid):
        _, layer, ident = json.loads(oid)
        if layer != 'long_term':
            raise ValueError('Only flushed Full can be deleted; sensory buffers must be empty')
        self.memory.embedding_retriever.delete(ident)
        if self.memory.embedding_retriever.get(ident):
            raise NativeFailure('Native point deletion did not commit')

    async def delete_turns(self, selected, completions=None):
        if self.fine_sources:
            from memcut.source_anchored_v2.whole_object_deletion import delete_source_turns
            return await delete_source_turns(self, selected)
        return await super().delete_turns(selected, completions)

    @contextmanager
    def _trace_extraction(self):
        lm = self.lm
        from lightmem.memory import utils
        assign, convert, create = lm.assign_sequence_numbers_with_timestamps, lm.convert_extraction_results_to_memory_entries, utils._create_memory_entry_from_fact
        batches, fact_batch, created, sequence, declared = [], {}, [], {}, {}
        self.new_writes = {}

        def observed_assign(extract_list, *args, **kwargs):
            result = assign(extract_list, *args, **kwargs)
            sequence.update({m['sequence_number']:m['_memcut_source_id']
                for batch in extract_list for seg in batch for m in seg})
            for batch in extract_list:
                roots = list(dict.fromkeys(m['_memcut_source_id'] for seg in batch for m in seg if m.get('content')))
                key = digest(['extract', roots, len(self.recipes['batches'])])
                self.recipes['batches'][key] = {'roots': roots, 'segments': copy.deepcopy(batch),
                    'slots': [], 'base_rows': [], 'slot_sources': {}, 'metadata_roots': []}
                batches.append(key)
            return result

        def observed_convert(extracted_results, *args, **kwargs):
            for i, result in enumerate(extracted_results):
                if not result or not result.get('output_prompt'):
                    raise NativeFailure('Upstream extraction swallowed a generation failure')
                key = self.request_by_messages.get(digest(result['input_prompt']))
                if key is None:
                    raise NativeFailure('Extraction result does not match an actual request')
                self.recipes['batches'][batches[i]]['request_ids'] = [key]
                for item in result.get('cleaned_result') or []:
                    for fact in item if isinstance(item, list) else [item]:
                        fact_batch[id(fact)] = batches[i]
                        if self.fine_sources:
                            declared[id(fact)] = fact_sources(fact, self.recipes['batches'][batches[i]]['segments'])
            return convert(extracted_results, *args, **kwargs)

        def observed_create(fact, *args, **kwargs):
            note = create(fact, *args, **kwargs)
            if note is None:
                raise NativeFailure('Native converter dropped a returned fact')
            origin = sequence.get(int(fact['source_id'])*2)
            if origin is None:
                raise NativeFailure('Native metadata has no actual source message')
            created.append((note, fact_batch[id(fact)], origin, declared.get(id(fact))))
            return note

        lm.assign_sequence_numbers_with_timestamps, lm.convert_extraction_results_to_memory_entries = observed_assign, observed_convert
        utils._create_memory_entry_from_fact = observed_create
        start = len(self.request_ids)
        try:
            yield
            # No extraction on most turns: do not scroll the growing vector
            # store just to discover that there are no newly created notes.
            store = self.memory.embedding_retriever
            fresh = store.client.retrieve(collection_name=store.collection_name,
                ids=[str(note.id) for note, _, _, _ in created], with_vectors=True, with_payload=True) if created else []
            current = {str(r.id): r.model_dump() for r in fresh}
            for note, key, origin, sources in created:
                recipe = self.recipes['batches'][key]
                recipe['slots'].append(str(note.id))
                recipe['base_rows'].append(copy.deepcopy(current[str(note.id)]))
                roots = self.lineage.ordered(sources if self.fine_sources else set(recipe['roots']) | {origin})
                recipe['slot_sources'][str(note.id)] = roots
                recipe['metadata_roots'] = self.lineage.ordered(set(recipe['metadata_roots']) | {origin})
                self.new_writes[self.oid('long_term', note.id)] = self.lineage.input(roots,
                    request_ids=recipe['request_ids'])
                if self.fine_sources:
                    recipe.setdefault('slot_origin_turns', {})[str(note.id)] = origin
                    self.new_writes[self.oid('long_term', note.id)].update(
                        attribution_policy=POLICY, read_turns=list(recipe['roots']),
                        native_metadata_turn=origin, batch_key=key)
        finally:
            lm.assign_sequence_numbers_with_timestamps, lm.convert_extraction_results_to_memory_entries = assign, convert
            utils._create_memory_entry_from_fact = create

    def ingest(self, turn_id, completions, *, final=False):
        if self.frozen:
            raise ValueError('Cannot ingest into frozen memory')
        self.completions = completions
        self.generation_failures = []
        t = self.turns[turn_id]
        before = self.observation()
        meta = {'time_stamp': t['timestamp'], 'speaker_id': t.get('speaker', t['role']),
                'speaker_name': t.get('speaker', t['role']), '_memcut_source_id': turn_id}
        self.lm.GLOBAL_TOPIC_IDX = self.topic_index
        with self._trace_extraction():
            self.memory.add_memory([{'role': 'user', 'content': t['text'], **meta},
                {'role': 'assistant', 'content': '', **meta}],
                METADATA_GENERATE_PROMPT=self.extraction_prompt(),
                force_segment=final, force_extract=final)
        if self.generation_failures:
            raise NativeFailure('Native generation failed inside worker')
        self.topic_index = self.lm.GLOBAL_TOPIC_IDX
        writes = self.new_writes
        for obj in self.native_objects():
            if obj['layer'] != 'long_term':
                writes[obj['id']] = self.lineage.input([obj['value']['_memcut_source_id']])
        self.lineage.commit(before, self.observation(), writes, label=f'ingest {turn_id}')

    def finish(self, completions):
        if any(self._buffers().values()):
            raise ValueError('The final original turn must force native segmentation and extraction')
        self.completions = completions
        self.generation_failures = []
        store = self.memory.embedding_retriever
        before = self.observation()
        self.memory.construct_update_queue_all_entries(top_k=20, keep_top_n=10, max_workers=1)
        after = self.observation()
        writes = {oid: self.lineage.input(self.trace(oid)) for oid in after if before.get(oid) != after[oid]}
        self.lineage.commit(before, after, writes, label='native offline candidate queues')
        queued = self.rows(with_vectors=False)
        if any('update_queue' not in r['payload'] for r in queued):
            raise NativeFailure('Native offline queue construction did not finish')
        expected_targets = {str(c['id']) for r in queued for c in r['payload'].get('update_queue', [])
                            if c['score'] >= .9}
        original = self.memory.manager._call_update_llm
        events = {}

        def observed_update(prompt, target, sources):
            start = len(self.request_ids)
            if self.fine_sources:
                # Stable content/source ordering keeps exact-request caches
                # reusable even when an uncommitted native UUID is regenerated.
                sources = sorted(sources, key=lambda r: (r['payload']['memory'],
                    self.lineage.ordered(self.trace(self.oid('long_term', r['id'])))))
                entries = [target, *sources]
                shown = []
                for i, row in enumerate(entries):
                    value = copy.deepcopy(row)
                    value['payload']['memory'] = f'[memory {i}] ' + row['payload']['memory']
                    shown.append(value)
                value = original(prompt + UPDATE_INSTRUCTION, shown[0], shown[1:])
                selected = numbers(value.get('source_memory_ids')) if value['action'] == 'update' else [0]
                if not set(selected) <= set(range(len(entries))):
                    raise NativeFailure('Consolidation cited an unavailable input memory')
            else:
                value = original(prompt, target, sources)
                selected = list(range(1 + len(sources)))
            ids = [str(r['id']) for r in [target, *sources]]
            inputs = self.lineage.input(parents=[self.oid('long_term', ids[i]) for i in selected],
                request_ids=self.request_ids[start:])
            if self.fine_sources:
                inputs.update(attribution_policy=POLICY,
                    read_parents=self.lineage.input(parents=[self.oid('long_term', i) for i in ids])['parents'])
            events[str(target['id'])] = {'target': ids[0], 'sources': ids[1:],
                'inputs': inputs, 'result': value}
            return value

        before = self.observation()
        self.memory.manager._call_update_llm = observed_update
        try:
            self.memory.offline_update_all_entries(score_threshold=.9, max_workers=1)
        finally:
            self.memory.manager._call_update_llm = original
        if self.generation_failures or set(events) != expected_targets:
            raise NativeFailure('Native offline consolidation swallowed an incomplete worker')
        after = self.observation()
        writes = {}
        rows = {str(r['id']): r for r in self.rows(with_vectors=False)}
        for ident, event in events.items():
            action = event['result']['action']
            if action == 'update':
                if ident not in rows or rows[ident]['payload']['memory'] != event['result']['new_memory']:
                    raise NativeFailure('Native consolidation update failed')
                writes[self.oid('long_term', ident)] = event['inputs']
            elif action == 'delete' and ident in rows:
                raise NativeFailure('Native consolidation delete failed')
        self.lineage.commit(before, after, writes, label='native final offline consolidation',
                            replace_sources=writes.keys() if self.fine_sources else ())
        self.recipes['updates'] = events
        self.recipes['full_live_ids'] = list(rows)
        self.progress['stage'] = 'FULL'
        self.freeze_updates()

    def _reextract(self, recipe, selected):
        from lightmem.memory import utils
        batch = []
        for segment in recipe['segments']:
            keep = []
            for m in segment:
                tid = m['_memcut_source_id']
                if tid not in selected:
                    row = copy.deepcopy(m)
                    row['content'] = self.turns[tid]['text'] if row['role'] == 'user' else ''
                    keep.append(row)
            if keep:
                batch.append(keep)
        if not batch:
            return []
        extraction, ts, weekdays, speakers, topics = utils.assign_sequence_numbers_with_timestamps([batch],
            topic_id_mapping=[[i for i in range(len(batch))]])
        results = self.memory.manager.meta_text_extract(extract_list=extraction, messages_use='user_only',
            topic_id_mapping=[[i for i in range(len(batch))]], extraction_mode='flat',
            custom_prompts={'factual': self.prompts.METADATA_GENERATE_PROMPT_locomo})
        if any(not r or not r.get('output_prompt') for r in results):
            raise NativeFailure('Incomplete native regeneration')
        notes = utils.convert_extraction_results_to_memory_entries(results, ts, weekdays, speakers,
            topics, max_source_ids=[sum(m['role']=='user' for seg in batch for m in seg)-1], logger=self.memory.logger)
        expected = sum(len(group) if isinstance(group,list) else 1
                       for result in results for group in result.get('cleaned_result') or [])
        if len(notes) != expected:
            raise NativeFailure('Native regeneration silently dropped a returned fact')
        return [{'id': n.id, 'payload': {k: v for k, v in asdict(n).items()
                 if k not in {'id','hit_time','update_queue'} and (k!='bam_tags' or v)},
                 'vector': self.memory.text_embedder.embed(n.memory)} for n in notes]

    def regenerate_groups(self, affected, selected, completions):
        if self.fine_sources:
            raise ValueError('Fine-source protocol uses whole-object deletion; no regeneration')
        from lightmem.memory.prompts import UPDATE_PROMPT
        self.completions = completions
        start = len(self.request_ids)
        bases, sources = {}, {}
        changed_ids = set()
        for key, recipe in self.recipes['batches'].items():
            remaining = [t for t in recipe['roots'] if t not in selected]
            changed = bool((set(recipe['roots']) | set(recipe.get('metadata_roots', []))) & selected)
            output = self._reextract(recipe, selected) if changed else recipe['base_rows']
            for i, row in enumerate(output):
                ident = recipe['slots'][i] if i < len(recipe['slots']) else str(uuid.uuid5(uuid.NAMESPACE_URL, key+f'/slot/{i}'))
                if changed:
                    row['id'] = ident
                bases[ident] = row
                sources[ident] = remaining if changed else recipe.get('slot_sources', {}).get(ident, remaining)
                if changed:
                    changed_ids.add(ident)
            if changed:
                changed_ids.update(recipe['slots'])
        current = {str(r['id']) for r in self.rows(with_vectors=False)}
        affected_ids = {json.loads(oid)[2] for oid in affected}
        # New extraction outputs have no old address. Preserve them rather than
        # truncating a native one-to-many result to the old number of facts.
        original_slots = {i for r in self.recipes['batches'].values() for i in r['slots']}
        affected_ids |= changed_ids - original_slots
        targets, target_roots = {}, {}
        for ident in sorted(affected_ids):
            row = copy.deepcopy(bases.get(ident))
            roots = set(sources.get(ident, []))
            update = self.recipes['updates'].get(ident)
            if update and update['result']['action'] == 'update':
                candidates = [bases[i] for i in update['sources'] if i in bases]
                for i in update['sources']:
                    roots.update(sources.get(i, []))
                if row is None and candidates:
                    row = copy.deepcopy(candidates[0])
                    row['id'] = ident
                    row['payload']['memory'] = ''
                if row is not None and candidates:
                    result = self.memory.manager._call_update_llm(UPDATE_PROMPT, row, candidates)
                    if result['action'] == 'delete':
                        row = None
                    elif result['action'] == 'update':
                        row['payload']['memory'] = result['new_memory']
                        row['vector'] = self.memory.text_embedder.embed(result['new_memory'])
            if row is not None and roots and row['payload'].get('memory'):
                row['payload'].pop('update_queue', None)
                targets[ident], target_roots[ident] = row, roots
        before = self.observation()
        for ident in affected_ids & set(current):
            self.erase_native(self.oid('long_term', ident))
        if targets:
            self._insert(list(targets.values()))
        self.lineage.commit(before, self.observation(), {self.oid('long_term', i):
            self.lineage.input(target_roots[i], request_ids=self.request_ids[start:]) for i in targets},
            rewrite=True, label='native remaining-input extraction/consolidation replay')

    async def search(self, query, *, top_k=20):
        before = self.read_identity()
        rows = self.rows()
        result = self.Retriever(self.memory.text_embedder).retrieve(rows, query, limit=top_k)
        if len(result) != min(top_k, len(rows)):
            raise NativeFailure('Native retrieval failed; not an empty-evidence score')
        native = {str(r['id']): r for r in rows}
        objects = [MemoryObject(self.system, 'long_term', str(r['id']), r['payload']['memory'],
            digest(native[str(r['id'])]['payload'])) for r in result]
        if self.read_identity() != before:
            raise ValueError('Retrieval modified frozen LightMem')
        return SearchResult(query, {'retrieved_document': objects}, native_context={'results': result},
            costs={'native_seed_hits': [{'address': o.address, 'rank': i+1, 'distance': -r['score']}
                for i,(o,r) in enumerate(zip(objects,result))], 'requested_top_k': top_k,
                'native_operation': 'locomo.VectorRetriever.retrieve'})

    def system_render(self, result):
        return self.render.format_related_memories(result.native_context['results'])

    def close(self):
        self.memory.embedding_retriever.client.close()
        self.memory.summary_retriever.client.close()
        self.temp.cleanup()
