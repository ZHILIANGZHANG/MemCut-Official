"""Native merging and prediction/correction on snapshot-compatible local stores.

Keep the pinned v0.1.1 generators and their scheduling. A full batch of `batch_threshold` messages is segmented
and its first group may merge into an earlier episode, as v0.1.1 does; what is left at the end goes through
v0.1.1's LoCoMo flush (memory.flush -> force_episode_creation): one "Force episode creation" episode, no
segmentation and no merge check (fewer than episode_min_messages make none). This is mechanism alignment, not a
v0.2.0 prompt reproduction.

Also as upstream: ids have the native uuid format (deterministic uuid5 here, so a replay reuses its receipts), and
the native answer prompt shows a semantic memory with the time LoCoMo's searcher fills in
(LocomoSearcher._hydrate_semantic_results). Two things upstream does silently are counted in
progress['diagnostics']: a merge decision naming no candidate, and a merge or correction reply missing a field
that upstream then defaults. BM25 must run on spaCy's en_core_web_sm; without it upstream tokenizes to empty
strings and its hybrid search silently becomes vector-only, so the bank refuses to start.
"""
from datetime import datetime
from types import SimpleNamespace
import uuid
from unittest.mock import patch

from memcut.native_memory.contract import digest
from memcut.source_anchored_systems.nemori import NemoriBank, NativeLLM, NativeFailure


def native_id(*parts):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, digest(list(parts))))


class PredictionLLM(NativeLLM):
    def __init__(self, model, completions=None, *, count=None):
        super().__init__(model, completions)
        self.count = count or (lambda name, n=1: None)

    def generate_json_response(self, *args, **kwargs):
        value = super().generate_json_response(*args, **kwargs)
        category = kwargs.get('category')
        if category in ('semantic_coldstart', 'semantic_correction'):
            if (not isinstance(value, dict) or not isinstance(value.get('statements', []), list)
                    or any(not isinstance(s, str) for s in value.get('statements', []))):
                raise NativeFailure('Invalid native semantic statements')
            if 'statements' not in value:
                self.count('statements_missing')
        elif category == 'merge_content' and isinstance(value, dict):
            if not isinstance(value.get('content'), str) or not isinstance(value.get('title'), str):
                self.count('merge_content_defaulted')
        return value

    def generate_text_response(self, prompt, system_prompt=None, temperature=0,
                               max_tokens=8192, **ignored):
        messages = ([{'role': 'system', 'content': system_prompt}] if system_prompt else [])
        messages.append({'role': 'user', 'content': prompt})
        try:
            response = self.completions.complete(dict(model=self.model, messages=messages,
                temperature=0, max_tokens=max_tokens))
            self.requests.append(response['request_id'])
            return response['text']
        except Exception as exc:
            raise NativeFailure(type(exc).__name__) from exc


def lemmatizing(nlp):
    """spaCy's English pipeline with lemmas (en_core_web_sm), not the blank fallback upstream drops to."""
    return nlp is not None and 'lemmatizer' in getattr(nlp, 'pipe_names', ())


class AlignedNemoriBank(NemoriBank):
    def __init__(self, settings, turns=None, *, embedder, execution=None):
        super().__init__(settings, turns, embedder=embedder)
        if not lemmatizing(self.engine.bm25_search.nlp):
            self.close()
            raise NativeFailure("Nemori's BM25 needs spaCy's en_core_web_sm (python -m spacy download "
                                'en_core_web_sm in the Nemori environment); without it upstream silently '
                                'searches by vectors only')
        from src.generation.episode_merger import EpisodeMerger
        from src.generation.semantic_generator import SemanticGenerator
        self.config.enable_episode_merging = True
        self.config.enable_prediction_correction = True
        self.config.extract_semantic_per_episode = False
        self.llm = PredictionLLM(settings['model'], count=self._count)
        self.segmenter.llm_client = self.generator.llm_client = self.llm
        storage = SimpleNamespace(get_episode=lambda ident, user:
            next((ep for ep in self.episodes.list_by_user(user) if ep.episode_id == ident), None))
        self.merger = EpisodeMerger(self.llm, embedder, self.config, storage, self.engine.chroma_search)
        decide = self.merger._decide_merge

        def counted_decision(new_episode, candidates):
            should, target, reason = decide(new_episode, candidates)
            if should and target and target not in {c.episode_id for c in candidates}:
                self._count('merge_target_unmatched')   # upstream then keeps the episode unmerged
            return should, target, reason
        self.merger._decide_merge = counted_decision
        self.semantic_generator = SemanticGenerator(self.llm, embedder, self.config,
                                                    self.engine.chroma_search)

    def _count(self, name, n=1):
        diagnostics = self.progress.setdefault('diagnostics', {})
        diagnostics[name] = diagnostics.get(name, 0) + n

    def load_native(self, value):
        super().load_native(value)
        # MessageBuffer.from_dict re-adds every message, which stamps last_updated with the wall clock; a
        # checkpoint taken with the final partial batch still buffered would then never restore.
        self.buffer.last_updated = datetime.fromisoformat(value['buffer']['last_updated'])

    def _episode(self, messages, reason):
        start = len(self.llm.requests)
        ep = self.generator.generate_episode(self.user, messages, reason)
        ep.created_at = min(m.timestamp for m in messages)
        # IDs appear in later native merge prompts. Stable identities let an interrupted batch reuse its
        # exact receipts without new calls; the uuid format is the one the merge prompt shows upstream.
        ep.episode_id = native_id('aligned-episode', [m.message_id for m in messages], ep.title, ep.content)
        return ep, self.llm.requests[start:]

    def _semantics(self, ep, eoid):
        start = len(self.llm.requests)
        existing = self.semantics.list_by_user(self.user)
        engine = self.semantic_generator.prediction_correction_engine
        retrieve = engine._retrieve_relevant_statements
        used = []

        def traced_retrieve(*args, **kwargs):
            texts = retrieve(*args, **kwargs)
            # Only statements actually supplied to prediction are parents.
            used.extend(self.oid('semantic', m.memory_id) for m in existing if m.content in texts)
            return texts

        with patch.object(engine, '_retrieve_relevant_statements', traced_retrieve):
            generated = self.semantic_generator.check_and_generate_semantic_memories(
                self.user, ep, self.episodes.list_by_user(self.user), existing)
        accepted = []
        for index, mem in enumerate(generated):
            if not self._is_duplicate(mem, existing + accepted):
                mem.memory_id = native_id('aligned-semantic', ep.episode_id, index, mem.content)
                accepted.append(mem)
        return accepted, used, self.llm.requests[start:]

    def _commit(self, ep, roots, request_ids, old_id=None, label='native episode/merge'):
        eoid = self.oid('episodic', ep.episode_id)
        parents = [self.oid('episodic', old_id)] if old_id else []
        inputs = self.lineage.input(roots, parents=parents, request_ids=request_ids)

        def store_episode():
            if old_id:
                self.erase_native(self.oid('episodic', old_id))
            self._put_episode(ep)

        self.observe_write(store_episode, {eoid: inputs}, label=label)
        accepted, used, semantic_requests = self._semantics(ep, eoid)
        self.recipes[ep.episode_id] = dict(episode=ep.to_dict(), roots=inputs['turns'],
                                           slots=[m.memory_id for m in accepted])
        if accepted:
            inputs = self.lineage.input(parents=[eoid, *dict.fromkeys(used)], request_ids=semantic_requests)
            self.observe_write(lambda: self._put_semantics(accepted),
                {self.oid('semantic', m.memory_id): inputs for m in accepted},
                label='native prediction/correction')

    def _process_buffer(self):
        messages = self.buffer.get_messages()
        start = len(self.llm.requests)
        groups = self.segmenter.segment_batch(messages)
        segmentation_requests = self.llm.requests[start:]
        flat = [i for group in groups for i in group]
        if any(type(i) is not int for i in flat) or sorted(flat) != list(range(1, len(messages) + 1)):
            raise NativeFailure('Segmentation must cover the input messages exactly once')
        for number, group in enumerate(groups):
            batch = [messages[i - 1] for i in group]
            ep, episode_requests = self._episode(batch, f'Batch segmentation (group {number + 1}/{len(groups)})')
            old_id = None
            start = len(self.llm.requests)
            if number == 0:
                merged, merged_ep, old_id = self.merger.check_and_merge(ep)
                if merged:
                    merged_ep.episode_id = native_id('aligned-merge', old_id, ep.episode_id,
                                                     merged_ep.title, merged_ep.content)
                    merged_ep.created_at = ep.created_at
                    # upstream stamps the wall clock here; the source time keeps a rebuilt Full identical
                    merged_ep.metadata['merge_timestamp'] = ep.created_at.isoformat()
                    ep = merged_ep
                else:
                    old_id = None
            self._commit(ep, [m.message_id for m in batch],
                         segmentation_requests + episode_requests + self.llm.requests[start:], old_id)
        self.observe_write(self.buffer.clear, {}, label='native consumed source buffer')

    def _flush_buffer(self):
        messages = self.buffer.get_messages()
        if len(messages) >= self.config.episode_min_messages:
            ep, requests = self._episode(messages, 'Force episode creation')
            self._commit(ep, [m.message_id for m in messages], requests, label='native forced episode (flush)')
        else:
            self._count('flush_below_episode_min_messages', len(messages))
        self.observe_write(self.buffer.clear, {}, label='native consumed source buffer')

    def finish(self, completions):
        self.llm.completions = completions
        if not self.buffer.is_empty():
            self._flush_buffer()
        self.progress['stage'] = 'FULL'
        self.freeze_updates()

    def system_render(self, result):
        # LoCoMo's searcher puts a semantic memory's stored time on its row before formatting; Chroma rows
        # carry it only in their metadata, so without this every semantic line reads "- [] ...".
        self.render.LocomoSearcher._hydrate_semantic_results(None, result.native_context.get('semantic', []))
        return super().system_render(result)
