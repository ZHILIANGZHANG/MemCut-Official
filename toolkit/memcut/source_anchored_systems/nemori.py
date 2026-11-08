"""Native Nemori v0.1.1, local repositories and real Chroma/BM25 retrieval.

The upstream's direct per-episode semantic mode is the lightweight formation
profile. Scheduling is serial at committed source boundaries; prompts,
segmentation, generation, representations and retrieval remain upstream code.
"""
from __future__ import annotations

import copy
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import uuid

from memcut.native_memory.contract import MemoryObject, SearchResult, digest
from memcut.source_anchored_v2.completions import parse_json
from .base import SourceBank
from .profiles import ROOT, nemori_modules, profile


class NativeFailure(BaseException):
    """Escape upstream broad Exception handlers; never fabricate empty memory."""


class NativeLLM:
    def __init__(self, model, completions=None):
        self.model, self.completions = model, completions
        self.requests = []

    def generate_json_response(self, prompt, system_prompt=None, temperature=0,
                               max_tokens=8192, category=None, **ignored):
        messages = ([{'role': 'system', 'content': system_prompt}] if system_prompt else [])
        messages.append({'role': 'user', 'content': prompt})
        try:
            response = self.completions.complete({'model': self.model, 'messages': messages,
                'temperature': 0, 'max_tokens': max_tokens, 'response_format': {'type': 'json_object'}})
            self.requests.append(response['request_id'])
            value = parse_json(response['text'])
            if category == 'episode_generation':
                if any(not isinstance(value.get(k), str) or not value[k].strip() for k in ('title', 'content')):
                    raise ValueError('Invalid native episode response')
            elif category == 'batch_segmentation':
                if (not isinstance(value.get('episodes'), list) or not value['episodes'] or
                    any(not isinstance(g, dict) or not g.get('indices') or
                        not isinstance(g['indices'], list) or any(type(i) is not int for i in g['indices'])
                        for g in value['episodes'])):
                    raise ValueError('Invalid native segmentation response')
            elif category == 'semantic_direct_extraction':
                if not all(isinstance(v, str) or (isinstance(v, list) and all(isinstance(s, str) for s in v))
                           for v in value.values()):
                    raise ValueError('Invalid native semantic response')
            return value
        except Exception as exc:
            raise NativeFailure(type(exc).__name__) from None


class LocalEmbedding:
    def __init__(self, path):
        from sentence_transformers import SentenceTransformer
        self.encoder = SentenceTransformer(path, device='cpu', local_files_only=True)
        self.model, self.embedding_dim = 'all-MiniLM-L6-v2', 384
        self.cache = {}
        self.prepared = {}

    def embed_texts(self, texts):
        missing = list(dict.fromkeys(t for t in texts if t not in self.cache))
        if missing:
            vectors = self.encoder.encode(missing, normalize_embeddings=False, show_progress_bar=False)
            self.cache.update(zip(missing, vectors.tolist()))
            import numpy as np
            for text in missing:
                vector = self.cache[text]
                array = np.asarray(vector)
                self.prepared[id(vector)] = (vector, array, np.linalg.norm(array))
        return SimpleNamespace(embeddings=[self.cache[t] for t in texts], usage={})

    def embed_text(self, text):
        return self.embed_texts([text]).embeddings[0]

    def cosine_similarity(self, a, b):
        import numpy as np
        # Only cache this embedder's immutable output lists. Foreign vectors
        # still use the original calculation; the native threshold is unchanged.
        def prepare(vector):
            saved = self.prepared.get(id(vector))
            if saved is not None and saved[0] is vector:
                return saved[1:]
            array = np.asarray(vector)
            return array, np.linalg.norm(array)
        a, an = prepare(a)
        b, bn = prepare(b)
        denominator = an * bn
        return float(np.dot(a, b) / denominator) if denominator else 0.0


class NemoriBank(SourceBank):
    system = 'nemori'

    def __init__(self, settings, turns=None, *, embedder):
        nemori_modules()
        super().__init__(settings, turns)
        from src.config import MemoryConfig
        from src.models import MessageBuffer
        from src.infrastructure.repositories import InMemoryEpisodeRepository, InMemorySemanticRepository
        from src.generation.batch_segmenter import BatchSegmenter
        from src.generation.episode_generator import EpisodeGenerator
        from src.generation.semantic_generator import SemanticGenerator
        from src.search.unified_search import UnifiedSearchEngine
        self.user = 'memcut'
        self.temp = tempfile.TemporaryDirectory(prefix='memcut-nemori-')
        self.config = MemoryConfig(openai_api_key='injected-transport-no-key', llm_model=settings['model'],
            storage_path=self.temp.name, chroma_persist_directory=str(Path(self.temp.name)/'chroma'),
            storage_backend='memory', enable_episode_merging=False, enable_prediction_correction=False,
            extract_semantic_per_episode=True, embedding_dimension=384, embedding_model='all-MiniLM-L6-v2',
            batch_threshold=settings['batch_threshold'], max_workers=1, semantic_generation_workers=1)
        self.llm, self.embedder = NativeLLM(settings['model']), embedder
        self.episodes, self.semantics = InMemoryEpisodeRepository(), InMemorySemanticRepository()
        self.buffer = MessageBuffer(self.user)
        self.segmenter = BatchSegmenter(self.llm, self.config)
        self.generator = EpisodeGenerator(self.llm, self.config)
        self.semantic_generator = SemanticGenerator(self.llm, embedder, self.config)
        self.engine = UnifiedSearchEngine(embedder, self.config, language='en')
        # Native data formatting/answer template; importing does not create a client.
        path = ROOT / profile('nemori')['repository'] / 'evaluation/locomo/search.py'
        spec = importlib.util.spec_from_file_location('memcut_nemori_eval', path)
        self.render = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.render)

    @staticmethod
    def oid(layer, ident):
        return json.dumps(['nemori', layer, ident], separators=(',', ':'))

    def native_objects(self):
        rows = []
        for m in self.buffer.get_messages():
            rows.append({'id': self.oid('buffer', m.message_id), 'layer': 'buffer',
                         'native_id': m.message_id, 'kind': 'atomic', 'text': m.content,
                         'value': m.to_dict(), 'visibility': 'stored_only'})
        for ep in self.episodes.list_by_user(self.user):
            # original_messages really reside in each native Episode. They are
            # auditor-visible stored text, not research-side source reinjection.
            text = ep.title + '\n' + ep.content
            if ep.original_messages:
                text += '\n' + '\n'.join(f"{m['role']}: {m['content']}" for m in ep.original_messages)
            rows.append({'id': self.oid('episodic', ep.episode_id), 'layer': 'episodic',
                         'native_id': ep.episode_id, 'kind': 'atomic', 'text': text, 'value': ep.to_dict()})
        for mem in self.semantics.list_by_user(self.user):
            rows.append({'id': self.oid('semantic', mem.memory_id), 'layer': 'semantic',
                         'native_id': mem.memory_id, 'kind': 'aggregate', 'text': mem.content, 'value': mem.to_dict()})
        return rows

    def _collection(self, layer):
        c = self.engine.chroma_search
        return c._get_episode_collection(self.user) if layer == 'episodic' else c._get_semantic_collection(self.user)

    def dump_native(self):
        vectors = {}
        for layer in ('episodic', 'semantic'):
            value = self._collection(layer).get(include=['embeddings', 'documents', 'metadatas'])
            vectors[layer] = {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in value.items()
                              if k in {'ids', 'embeddings', 'documents', 'metadatas'}}
        lexical = {}
        for name in ('episode','semantic'):
            engine = self.engine.bm25_search
            index = getattr(engine, name+'_indices').get(self.user)
            lexical[name] = {'data':[o.to_dict() for o in getattr(engine,name+'_data').get(self.user,[])],
                'tokens':copy.deepcopy(getattr(engine,name+'_tokenized_texts').get(self.user,[])),
                'index':copy.deepcopy(vars(index)) if index is not None else None}
        return {'episodes': [e.to_dict() for e in self.episodes.list_by_user(self.user)],
                'semantics': [m.to_dict() for m in self.semantics.list_by_user(self.user)],
                'buffer': self.buffer.to_dict(), 'vectors': vectors, 'lexical':lexical}

    def load_native(self, value):
        from src.models import Episode, SemanticMemory, MessageBuffer
        self.episodes.delete_user_data(self.user)
        self.semantics.delete_user_data(self.user)
        for v in value['episodes']:
            self.episodes.save(Episode.from_dict(v))
        for v in value['semantics']:
            self.semantics.save(SemanticMemory.from_dict(v))
        self.buffer = MessageBuffer.from_dict(value['buffer'])
        for layer, rows in value['vectors'].items():
            col = self._collection(layer)
            existing = col.get()['ids']
            if existing:
                col.delete(ids=existing)
            if rows['ids']:
                limit = self.engine.chroma_search.client.get_max_batch_size()
                for start in range(0, len(rows['ids']), limit):
                    col.add(**{k: v[start:start+limit] if v is not None else None
                               for k, v in rows.items()})
        # Rebuild only the lexical data structure. Saved embeddings are reused.
        lexical = self.engine.bm25_search
        lexical.clear_user_index(self.user)
        eps, sems = self.episodes.list_by_user(self.user), self.semantics.list_by_user(self.user)
        if eps:
            lexical.index_episodes(self.user, eps)
        if sems:
            lexical.index_semantic_memories(self.user, sems)

    def _put_episode(self, ep):
        self.episodes.save(ep)
        self.engine.chroma_search.index_episodes(self.user, [ep])
        self.engine.bm25_search.add_episode(self.user, ep)
        if ep.episode_id not in self._collection('episodic').get(ids=[ep.episode_id], include=[])['ids']:
            raise NativeFailure('Native episode indexing failed')

    def _put_semantic(self, mem):
        self.semantics.save(mem)
        self.engine.chroma_search.index_semantic_memories(self.user, [mem])
        self.engine.bm25_search.add_semantic_memory(self.user, mem)
        if mem.memory_id not in self._collection('semantic').get(ids=[mem.memory_id], include=[])['ids']:
            raise NativeFailure('Native semantic indexing failed')

    def _put_semantics(self, memories):
        for memory in memories:
            self._put_semantic(memory)

    def _is_duplicate(self, mem, existing):
        from src.core.memory_system import MemorySystem
        native = SimpleNamespace(config=self.config, embedding_client=self.embedder,
            performance_optimizer=SimpleNamespace(cached_call=lambda fn, key, text: fn(text)))
        return MemorySystem._is_duplicate_semantic_memory_fallback(native, self.user, mem,
                                                                   existing_memories=existing)

    def erase_native(self, oid):
        _, layer, ident = json.loads(oid)
        if layer == 'episodic':
            ok = self.episodes.delete(self.user, ident)
            self.engine.chroma_search.remove_episode(self.user, ident)
            self.engine.bm25_search.remove_episode(self.user, ident)
        else:
            ok = self.semantics.delete(self.user, ident)
            self.engine.chroma_search.remove_semantic_memory(self.user, ident)
            self.engine.bm25_search.remove_semantic_memory(self.user, ident)
        if not ok or ident in self._collection(layer).get(ids=[ident], include=[])['ids']:
            raise NativeFailure('Native deletion did not remove storage and vector')
        lexical = self.engine.bm25_search
        live = (lexical.episode_data.get(self.user,[]) if layer=='episodic' else
                lexical.semantic_data.get(self.user,[]))
        if any((o.episode_id if layer=='episodic' else o.memory_id)==ident for o in live):
            raise NativeFailure('Native deletion left a lexical index record')

    def _process_buffer(self):
        messages = self.buffer.get_messages()
        groups = self.segmenter.segment_batch(messages)
        flat = [i for g in groups for i in g]
        if any(type(i) is not int for i in flat) or sorted(flat) != list(range(1, len(messages)+1)):
            raise NativeFailure('Native segmentation must cover each actual input exactly once')
        for number, group in enumerate(groups):
            batch = [messages[i-1] for i in group]
            roots = [m.message_id for m in batch]
            start = len(self.llm.requests)
            reason = f'Batch segmentation (group {number+1}/{len(groups)})'
            ep = self.generator.generate_episode(self.user, batch, reason)
            ep.created_at = min(m.timestamp for m in batch)  # deterministic observation clock
            eoid = self.oid('episodic', ep.episode_id)
            self.observe_write(lambda: self._put_episode(ep),
                {eoid: self.lineage.input(roots, request_ids=self.llm.requests[start:])}, label='native episode')
            start = len(self.llm.requests)
            memories = self.semantic_generator.check_and_generate_semantic_memories(self.user, ep, [], [])
            accepted = []
            for mem in memories:
                if not self._is_duplicate(mem, self.semantics.list_by_user(self.user) + accepted):
                    accepted.append(mem)
            memories = accepted
            recipe_id = ep.episode_id
            self.recipes[recipe_id] = {'episode': ep.to_dict(), 'roots': roots, 'slots': [m.memory_id for m in memories]}
            if memories:
                # One actual model output is one observed write transaction.
                # Keep every output address and its exact parent input version;
                # no read or generation occurs between the native saves.
                inputs = self.lineage.input(parents=[eoid], request_ids=self.llm.requests[start:])
                self.observe_write(lambda: self._put_semantics(memories),
                    {self.oid('semantic', mem.memory_id): inputs for mem in memories},
                    label='native direct semantic batch')
        self.observe_write(self.buffer.clear, {}, label='native consumed source buffer')

    def ingest(self, turn_id, completions, *, final=False):
        from src.models import Message
        if self.frozen:
            raise ValueError('Formation cannot write a frozen state')
        t = self.turns[turn_id]
        self.llm.completions = completions
        message = Message(role=t['role'], content=t['text'],
            timestamp=datetime.fromisoformat(t['timestamp']), message_id=turn_id,
            metadata={'original_speaker': t.get('speaker', t['role'])})
        self.observe_write(lambda: self.buffer.add_message(message),
            {self.oid('buffer', turn_id): self.lineage.input([turn_id])}, label=f'native buffer ingest {turn_id}')
        if self.segmenter.should_create_episode(self.buffer.size())[0]:
            self._process_buffer()

    def finish(self, completions):
        self.llm.completions = completions
        if not self.buffer.is_empty():
            self._process_buffer()
        self.progress['stage'] = 'FULL'
        self.freeze_updates()

    def regenerate_groups(self, affected, selected, completions):
        from src.models import Episode, Message
        self.llm.completions = completions
        for recipe_id, recipe in self.recipes.items():
            live_ids = {self.oid('semantic', i) for i in recipe.get('live_slots', recipe['slots'])} & set(self.lineage.data['src'])
            if not live_ids & affected:
                continue
            remaining = [t for t in recipe['roots'] if t not in selected]
            start = len(self.llm.requests)
            new = []
            if remaining:
                # Recompute the title as well: an old title can itself disclose
                # the removed fact. The temporary episode is never reinserted.
                old_ep = Episode.from_dict(recipe['episode'])
                msgs = [Message.from_dict(m) for m in old_ep.original_messages if m['message_id'] in remaining]
                ep = self.generator.generate_episode(self.user, msgs, old_ep.boundary_reason)
                ep.episode_id, ep.created_at = old_ep.episode_id, min(m.timestamp for m in msgs)
                new = self.semantic_generator.generate_semantic_memories(self.user, [ep])
            # Native one-to-many output is replayed once. Stable output slots
            # retain IDs; disappearing slots are deleted, additional slots get
            # deterministic fresh IDs. No semantic matching call or old text.
            slots = recipe['slots']
            for i, mem in enumerate(new):
                if i < len(slots):
                    mem.memory_id = slots[i]
                else:
                    mem.memory_id = str(uuid.uuid5(uuid.NAMESPACE_URL,
                        digest([recipe_id, i, 'semantic-output-slot'])))
                mem.source_episodes = [recipe_id] if self.oid('episodic', recipe_id) in self.lineage.data['src'] else []
            before = self.observation()
            for oid in sorted(live_ids):
                self.erase_native(oid)
            for mem in new:
                self._put_semantic(mem)
            inputs = self.lineage.input(remaining, request_ids=self.llm.requests[start:])
            self.lineage.commit(before, self.observation(),
                {self.oid('semantic', m.memory_id): inputs for m in new}, rewrite=True, label='native semantic regeneration')
            recipe['live_slots'] = [m.memory_id for m in new]

    def _visible(self, rows):
        result = []
        stored = {o['id']: o for o in self.native_objects()}
        for row in rows:
            layer = 'episodic' if row.get('episode_id') else 'semantic'
            ident = row.get('episode_id') or row['memory_id']
            obj = stored[self.oid(layer, ident)]
            result.append(MemoryObject(self.system, layer, ident, row['content'], digest(obj['value'])))
        return result

    async def search(self, query, *, top_k=20):
        before = self.read_identity()
        rows = self.engine.search(self.user, query, top_k=top_k, memory_types=['episodic', 'semantic'])
        total = len(self.episodes.list_by_user(self.user)) + len(self.semantics.list_by_user(self.user))
        if len(rows) != min(top_k, total):
            raise NativeFailure('Native hybrid retrieval returned an incomplete object set')
        objects = self._visible(rows)
        hits = [{'address': o.address, 'rank': i+1, 'distance': -float(r.get('fused_score', r.get('score', 0)))}
                for i, (o, r) in enumerate(zip(objects, rows))]
        if before != self.read_identity():
            raise ValueError('Native search mutated memory')
        return SearchResult(query, {'retrieved_document': objects}, native_context={'results': rows},
            costs={'native_seed_hits': hits, 'requested_top_k': top_k, 'native_operation': 'UnifiedSearchEngine.search'})

    async def native_context(self, query):
        before = self.read_identity()
        c = self.engine.chroma_search
        episodes = c.search_episodes(self.user, query, self.settings['native_eval_k'])
        semantics = c.search_semantic_memories(self.user, query, self.settings['native_eval_semantic_k'])
        if (len(episodes) != min(self.settings['native_eval_k'], len(self.episodes.list_by_user(self.user))) or
            len(semantics) != min(self.settings['native_eval_semantic_k'], len(self.semantics.list_by_user(self.user)))):
            raise NativeFailure('Native vector QA retrieval failed')
        by_id = {e.episode_id: e for e in self.episodes.list_by_user(self.user)}
        for row in episodes:
            ep = by_id[row['episode_id']]
            row.update(timestamp=ep.timestamp.isoformat(), original_messages=ep.original_messages)
        ctx = {'episodic': episodes, 'semantic': semantics}
        result = SearchResult(query, {'retrieved_document': self._visible(episodes + semantics)}, native_context=ctx,
            costs={'native_operation': 'ChromaSearchEngine vector search', 'top_k_episodes': self.settings['native_eval_k'],
                   'top_k_semantic': self.settings['native_eval_semantic_k']})
        result.native_context['prompt'] = self.system_render(result)
        if self.read_identity() != before:
            raise ValueError('Native QA retrieval mutated memory')
        return result

    def system_render(self, result):
        ctx = result.native_context
        return self.render.ANSWER_PROMPT.render(question=result.query,
            episodic=self.render.format_memory_lines(ctx.get('episodic', []),
                include_original_limit=self.settings['native_eval_original_top_k']),
            semantic=self.render.format_memory_lines(ctx.get('semantic', [])))

    def close(self):
        if getattr(self, '_closed', False):
            return
        self._closed = True
        self.engine.executor.shutdown(wait=True)
        # Each branch owns a unique temporary persistence directory. Release
        # only its system; Chroma's global client registry otherwise retains
        # closed branches and their HNSW indices until the process exits.
        from chromadb.api.shared_system_client import SharedSystemClient
        client = self.engine.chroma_search.client
        system = client._system
        system.stop()
        if SharedSystemClient._identifier_to_system.get(client._identifier) is system:
            SharedSystemClient._identifier_to_system.pop(client._identifier)
        self.temp.cleanup()
