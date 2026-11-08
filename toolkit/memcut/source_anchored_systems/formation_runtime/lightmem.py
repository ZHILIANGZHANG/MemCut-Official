"""Native preprocessing, bounded extraction pipeline, ordered native writes."""
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import threading

from memcut.native_memory.contract import digest
from memcut.native_memory.lightmem_v1 import payload_record
from memcut.source_anchored_v2.completions import parse_json
from ..lightmem import LightMemBank
from ..lightmem_sources import POLICY, UPDATE_INSTRUCTION, numbers
from ..nemori import NativeFailure
from .concurrency import ConcurrentCompletions, ordered_map, Timings
from .delta import DeltaBank


class ExtractionQueued(BaseException):
    """Suspend upstream add_memory after native buffers/segmentation are ready."""


class FastLightMemBank(DeltaBank, LightMemBank):
    def __init__(self, *args, execution, **kwargs):
        super().__init__(*args, **kwargs)
        self.execution, self.timings = execution, Timings()
        self._pool = ThreadPoolExecutor(max_workers=execution['workers'], thread_name_prefix='lightmem-extract')
        self._pending = deque()
        self._cache = None
        self._thread = threading.local()

    def _order_key(self, oid):
        _, layer, ident = json.loads(oid)
        if layer == 'long_term':
            return (0, ident)
        return ({'sensory_buffer': 1, 'sensory_big_buffer': 2, 'short_buffer': 3}[layer],
                self.lineage.turn_rank[ident.rsplit('/', 1)[0]])

    def _buffer_objects(self):
        return [{'id': self.oid(layer, f"{m['_memcut_source_id']}/{m['role']}"),
                 'kind': 'atomic', 'text': m['content'], 'value': copy.deepcopy(m)}
                for layer, messages in self._buffers().items() for m in messages if m.get('content')]

    def _row_objects(self, rows):
        return [{'id': self.oid('long_term', row['id']), 'kind': 'aggregate',
                 'text': json.dumps(payload_record(row['payload']), ensure_ascii=False, separators=(',', ':')),
                 'value': row['payload']} for row in rows]

    def _set_writer(self, completions):
        if self._cache is not completions:
            self._cache = completions
            self.completions = ConcurrentCompletions(completions, self.execution['workers'])

    def _generate(self, messages, response_format=None, tools=None, **kwargs):
        if tools:
            raise NativeFailure('Unexpected native generation tools')
        try:
            with self.timings.measure('model_request_worker'):
                result = self.completions.complete({'model': self.settings['model'], 'messages': messages,
                    'temperature': 0, 'top_p': .1, 'max_tokens': self.settings['generation_max_tokens'],
                    **({'response_format': response_format} if response_format else {})})
            # Only bookkeeping holds the lock, never the network request.
            with self.dispatch_lock:
                self.request_ids.append(result['request_id'])
                self.request_by_messages[digest(messages)] = result['request_id']
            if hasattr(self._thread, 'requests'):
                self._thread.requests.append(result['request_id'])
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
            self.completions.failed.set()
            with self.dispatch_lock:
                self.generation_failures.append(type(exc).__name__)
            raise NativeFailure(type(exc).__name__) from None

    def ingest(self, turn_id, completions, *, final=False):
        if self.frozen:
            raise ValueError('Cannot ingest into frozen memory')
        self._set_writer(completions)
        before = self._buffer_objects()
        t = self.turns[turn_id]
        meta = {'time_stamp': t['timestamp'], 'speaker_id': t.get('speaker', t['role']),
                'speaker_name': t.get('speaker', t['role']), '_memcut_source_id': turn_id}
        assign = self.lm.assign_sequence_numbers_with_timestamps
        extract = self.memory.manager.meta_text_extract
        captured = {}

        def capture_assign(batches, *args, **kwargs):
            captured.update(batches=copy.deepcopy(batches), args=copy.deepcopy(args), kwargs=copy.deepcopy(kwargs))
            return assign(batches, *args, **kwargs)

        def queue_extract(*args, **kwargs):
            if not captured:
                raise NativeFailure('Native extraction lacks its actual input metadata')
            packet = {**captured, 'turn_id': turn_id,
                      'future': self._pool.submit(extract, *copy.deepcopy(args), **copy.deepcopy(kwargs))}
            self._pending.append(packet)
            raise ExtractionQueued()

        self.lm.GLOBAL_TOPIC_IDX = self.topic_index
        self.lm.assign_sequence_numbers_with_timestamps = capture_assign
        self.memory.manager.meta_text_extract = queue_extract
        try:
            with self.timings.measure('native_compression_and_segmentation'):
                try:
                    self.memory.add_memory([{'role': 'user', 'content': t['text'], **meta},
                        {'role': 'assistant', 'content': '', **meta}],
                        METADATA_GENERATE_PROMPT=self.extraction_prompt(), force_segment=final, force_extract=final)
                except ExtractionQueued:
                    pass
        finally:
            self.lm.assign_sequence_numbers_with_timestamps = assign
            self.memory.manager.meta_text_extract = extract
        self.topic_index = self.lm.GLOBAL_TOPIC_IDX
        after = self._buffer_objects()
        writes = {o['id']: self.lineage.input([o['value']['_memcut_source_id']]) for o in after}
        with self.timings.measure('incremental_source_commit'):
            self._delta(before, after, writes, f'native buffers {turn_id}')
        if len(self._pending) >= self.execution['workers']:
            self._commit_one()
        if final:
            self.flush_pending()

    def _commit_one(self):
        packet = self._pending[0]
        results = packet['future'].result()
        with self.timings.measure('ordered_extraction_commit'):
            old_keys = set(self.recipes['batches'])
            with self._trace_extraction():
                extraction, ts, days, speakers, topics = self.lm.assign_sequence_numbers_with_timestamps(
                    packet['batches'], *packet['args'], **packet['kwargs'])
                self.lm.process_extraction_results(extracted_results=results, token_stats=self.memory.token_stats,
                    result_dict={'add_input_prompt': [], 'add_output_prompt': [], 'api_call_nums': 0},
                    call_id=f"extract_{packet['turn_id']}", logger=self.memory.logger)
                notes = self.lm.convert_extraction_results_to_memory_entries(results, ts, days, speakers, topics,
                    max_source_ids=[sum(m.get('role') == 'user' for seg in b for m in seg) - 1 for b in extraction],
                    logger=self.memory.logger)
                # Native insertion and vector semantics unchanged; no new LLM call.
                self.memory.offline_update(notes)
            added = [key for key in self.recipes['batches'] if key not in old_keys]
            for i, key in enumerate(added):
                self.recipes['batches'][key].update(trigger_turn=packet['turn_id'],
                    native_topic_ids=packet['kwargs']['topic_id_mapping'][i],
                    native_request_messages=copy.deepcopy(results[i]['input_prompt']))
            ids = [json.loads(oid)[2] for oid in self.new_writes]
            store = self.memory.embedding_retriever
            rows = store.client.retrieve(collection_name=store.collection_name, ids=ids,
                with_payload=True, with_vectors=False) if ids else []
            self._delta([], self._row_objects([r.model_dump() for r in rows]), self.new_writes,
                        f"native extracted facts {packet['turn_id']}")
        self._pending.popleft()

    def flush_pending(self):
        while self._pending:
            self._commit_one()
        if self.generation_failures:
            raise NativeFailure('Native extraction worker failed')

    def export(self):
        if self._pending:
            raise ValueError('Drain extraction jobs before exporting a committed state')
        return super().export()

    def finish(self, completions):
        self._set_writer(completions)
        self.flush_pending()
        if any(self._buffers().values()):
            raise ValueError('Final source turn must flush the native buffers')
        store = self.memory.embedding_retriever
        before = self.observation()
        # Native query/filter/ties stay intact; CPU Qdrant reads/writes remain
        # serialized. The expensive independent model work below is concurrent.
        with self.timings.measure('native_update_queues'):
            self.memory.construct_update_queue_all_entries(top_k=20, keep_top_n=10, max_workers=1)
        after = self.observation()
        writes = {oid: self.lineage.input(self.trace(oid)) for oid in after if before.get(oid) != after[oid]}
        self.lineage.commit(before, after, writes, label='native offline candidate queues')
        rows = self.rows()
        if any('update_queue' not in r['payload'] for r in rows):
            raise NativeFailure('Native offline queues incomplete')
        # Same reverse-queue relation and order, computed once instead of an
        # O(objects² × queue length) scan for every possible target.
        reverse = defaultdict(list)
        for row in rows:
            seen = set()
            for candidate in row['payload']['update_queue']:
                ident = str(candidate['id'])
                if candidate['score'] >= .9 and ident not in seen:
                    reverse[ident].append(row)
                    seen.add(ident)
        targets = [r for r in rows if str(r['id']) in reverse]
        if set(reverse) - {str(r['id']) for r in targets}:
            raise NativeFailure('Offline queue points outside the native store')

        def generate(target):
            from lightmem.memory.prompts import UPDATE_PROMPT
            sources = reverse[str(target['id'])]
            self._thread.requests = []
            if self.fine_sources:
                sources = sorted(sources, key=lambda r: (r['payload']['memory'],
                    self.lineage.ordered(self.trace(self.oid('long_term', r['id'])))))
                entries = [target, *sources]
                shown = copy.deepcopy(entries)
                for i, row in enumerate(shown):
                    row['payload']['memory'] = f'[memory {i}] ' + row['payload']['memory']
                value = self.memory.manager._call_update_llm(UPDATE_PROMPT + UPDATE_INSTRUCTION, shown[0], shown[1:])
                if value is None:
                    raise NativeFailure('Native consolidation returned no result')
                selected = numbers(value.get('source_memory_ids')) if value['action'] == 'update' else [0]
                if not set(selected) <= set(range(len(entries))):
                    raise NativeFailure('Consolidation cited unavailable memory')
            else:
                value = self.memory.manager._call_update_llm(UPDATE_PROMPT, target, sources)
                selected = list(range(1 + len(sources)))
            ids = [str(r['id']) for r in [target, *sources]]
            inputs = self.lineage.input(parents=[self.oid('long_term', ids[i]) for i in selected],
                                        request_ids=self._thread.requests)
            if self.fine_sources:
                inputs.update(attribution_policy=POLICY,
                    read_parents=self.lineage.input(parents=[self.oid('long_term', i) for i in ids])['parents'])
            return target, {'target': ids[0], 'sources': ids[1:], 'inputs': inputs, 'result': value}

        before = self.observation()
        events, writes = {}, {}
        with self.timings.measure('parallel_offline_consolidation'):
            for target, event in ordered_map(generate, targets, self.execution['workers']):
                ident, value = str(target['id']), event['result']
                events[ident] = event
                usage = value['usage']
                self.memory.token_stats['update_calls'] += 1
                for name in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
                    self.memory.token_stats['update_' + name] += usage.get(name, 0)
                if value['action'] == 'delete':
                    store.delete(ident)
                    if store.get(ident):
                        raise NativeFailure('Native consolidation delete failed')
                elif value['action'] == 'update':
                    payload = {**target['payload'], 'memory': value['new_memory']}
                    # Match upstream exactly: updates keep the original vector.
                    store.update(vector_id=ident, vector=target['vector'], payload=payload)
                    stored = store.get(ident)
                    if stored is None or stored.payload['memory'] != value['new_memory']:
                        raise NativeFailure('Native consolidation update failed')
                    writes[self.oid('long_term', ident)] = event['inputs']
        if self.generation_failures or len(events) != len(targets):
            raise NativeFailure('Native consolidation incomplete')
        self.lineage.commit(before, self.observation(), writes, label='native final offline consolidation',
                            replace_sources=writes.keys() if self.fine_sources else ())
        self.recipes['updates'] = events
        self.recipes['full_live_ids'] = [str(r['id']) for r in self.rows(with_vectors=False)]
        self.lineage.validate(observed=self.observation())
        self.progress['stage'] = 'FULL'
        self.freeze_updates()

    def close(self):
        if self._pending and isinstance(self.completions, ConcurrentCompletions):
            self.completions.failed.set()
        for packet in self._pending:
            packet['future'].cancel()
        self._pool.shutdown(wait=True, cancel_futures=True)
        super().close()
