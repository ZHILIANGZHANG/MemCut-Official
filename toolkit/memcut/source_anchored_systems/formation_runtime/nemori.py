"""Parallel native generation; ordered native deduplication and persistence."""
import copy
from datetime import datetime
import json

from ..nemori import NemoriBank, NativeLLM, NativeFailure
from .concurrency import ConcurrentCompletions, ordered_map, Timings
from .delta import DeltaBank


class FastNemoriBank(DeltaBank, NemoriBank):
    def __init__(self, *args, execution, **kwargs):
        super().__init__(*args, **kwargs)
        self.execution, self.timings = execution, Timings()
        self._writer = self._cache = None
        self._positions = {}
        self._lexical_dirty = set()

    def _order_key(self, oid):
        _, layer, ident = json.loads(oid)
        return ({'buffer': 0, 'episodic': 1, 'semantic': 2}[layer], self._positions[oid])

    def load_native(self, value):
        super().load_native(value)
        self._positions = {o['id']: i for i, o in enumerate(super().native_objects())}
        self._lexical_dirty.clear()

    def _object(self, layer, obj):
        ident = getattr(obj, {'buffer': 'message_id', 'episodic': 'episode_id', 'semantic': 'memory_id'}[layer])
        oid = self.oid(layer, ident)
        self._positions.setdefault(oid, len(self._positions))
        if layer == 'episodic':
            text = obj.title + '\n' + obj.content
            if obj.original_messages:
                text += '\n' + '\n'.join(f"{m['role']}: {m['content']}" for m in obj.original_messages)
        else:
            text = obj.content
        return {'id': oid, 'kind': 'aggregate' if layer == 'semantic' else 'atomic',
                'text': text, 'value': obj.to_dict()}

    def _lexical_append(self, name, objects):
        engine = self.engine.bm25_search
        data = getattr(engine, name + '_data').setdefault(self.user, [])
        tokens = getattr(engine, name + '_tokenized_texts').setdefault(self.user, [])
        for obj in objects:
            text = f'{obj.title} {obj.content}' if name == 'episode' else obj.content
            encoded = engine._tokenize(text)
            if not encoded:
                raise NativeFailure('Native lexical tokenization returned no tokens')
            data.append(obj)
            tokens.append(encoded)
        self._lexical_dirty.add(name)

    def flush_pending(self):
        from rank_bm25 import BM25Okapi
        engine = self.engine.bm25_search
        for name in sorted(self._lexical_dirty):
            getattr(engine, name + '_indices')[self.user] = BM25Okapi(
                getattr(engine, name + '_tokenized_texts')[self.user])
        self._lexical_dirty.clear()

    def _store(self, layer, objects):
        repository = self.episodes if layer == 'episodic' else self.semantics
        for obj in objects:
            repository.save(obj)
        if layer == 'episodic':
            self.engine.chroma_search.index_episodes(self.user, objects)
        else:
            self.engine.chroma_search.index_semantic_memories(self.user, objects)
        self._lexical_append('episode' if layer == 'episodic' else 'semantic', objects)
        rows = [self._object(layer, obj) for obj in objects]
        expected = {json.loads(r['id'])[2] for r in rows}
        if set(self._collection(layer).get(ids=list(expected), include=[])['ids']) != expected:
            raise NativeFailure('Native vector batch failed to persist')
        # Repositories retain these exact native objects; verify the saved IDs.
        stored = repository.list_by_user(self.user)[-len(objects):]
        actual = [self._object(layer, obj) for obj in stored]
        if rows != actual:
            raise NativeFailure('Native repository batch failed to persist')
        return actual

    def _process_buffer(self):
        messages = self.buffer.get_messages()
        with self.timings.measure('segmentation'):
            groups = self.segmenter.segment_batch(messages)
        flat = [i for g in groups for i in g]
        if any(type(i) is not int for i in flat) or sorted(flat) != list(range(1, len(messages) + 1)):
            raise NativeFailure('Native segmentation must cover every input once')

        def generate(item):
            number, group = item
            # Per-job clients isolate request IDs; all prompts/generators stay native.
            llm = NativeLLM(self.settings['model'], self._writer)
            generator, semantic = copy.copy(self.generator), copy.copy(self.semantic_generator)
            generator.llm_client = semantic.llm_client = llm
            batch = [messages[i - 1] for i in group]
            reason = f'Batch segmentation (group {number + 1}/{len(groups)})'
            with self.timings.measure('episode_and_semantic_generation_worker'):
                ep = generator.generate_episode(self.user, batch, reason)
                ep.created_at = min(m.timestamp for m in batch)
                episode_requests = list(llm.requests)
                memories = semantic.check_and_generate_semantic_memories(self.user, ep, [], [])
            return ep, memories, [m.message_id for m in batch], episode_requests, llm.requests[len(episode_requests):]

        def safe_generate(item):
            try:
                return generate(item)
            except BaseException:
                self._writer.failed.set()
                raise

        existing = self.semantics.list_by_user(self.user)
        for ep, memories, roots, episode_requests, semantic_requests in ordered_map(
                safe_generate, enumerate(groups), self.execution['workers']):
            with self.timings.measure('ordered_native_commit'):
                eoid = self.oid('episodic', ep.episode_id)
                rows = self._store('episodic', [ep])
                self._delta([], rows, {eoid: self.lineage.input(roots, request_ids=episode_requests)}, 'native episode')
                accepted = []
                for mem in memories:
                    # The comparison set grows in the original serial order.
                    if not self._is_duplicate(mem, existing + accepted):
                        accepted.append(mem)
                self.recipes[ep.episode_id] = {'episode': ep.to_dict(), 'roots': roots,
                                              'slots': [m.memory_id for m in accepted]}
                if accepted:
                    inputs = self.lineage.input(parents=[eoid], request_ids=semantic_requests)
                    rows = self._store('semantic', accepted)
                    self._delta([], rows, {r['id']: inputs for r in rows}, 'native direct semantic batch')
                    existing.extend(accepted)
                self.llm.requests.extend(episode_requests + semantic_requests)
        before = [self._object('buffer', m) for m in self.buffer.get_messages()]
        self.buffer.clear()
        self._delta(before, [], {}, 'native consumed source buffer')
        self.flush_pending()

    def ingest(self, turn_id, completions, *, final=False):
        from src.models import Message
        if self.frozen:
            raise ValueError('Formation cannot write frozen memory')
        if self._cache is not completions:
            self._cache, self._writer = completions, ConcurrentCompletions(completions, self.execution['workers'])
        self.llm.completions = self._writer
        t = self.turns[turn_id]
        message = Message(role=t['role'], content=t['text'], timestamp=datetime.fromisoformat(t['timestamp']),
            message_id=turn_id, metadata={'original_speaker': t.get('speaker', t['role'])})
        self.buffer.add_message(message)
        row = self._object('buffer', self.buffer.get_messages()[-1])
        self._delta([], [row], {row['id']: self.lineage.input([turn_id])}, f'native buffer ingest {turn_id}')
        if self.segmenter.should_create_episode(self.buffer.size())[0]:
            self._process_buffer()

    def finish(self, completions):
        if self._cache is not completions:
            self._cache, self._writer = completions, ConcurrentCompletions(completions, self.execution['workers'])
        self.llm.completions = self._writer
        if not self.buffer.is_empty():
            self._process_buffer()
        self.flush_pending()
        self.lineage.validate(observed=self.observation())
        self.progress['stage'] = 'FULL'
        self.freeze_updates()
