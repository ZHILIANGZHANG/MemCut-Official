"""Whole-session max-cosine retrieval with exact source-turn text removal.

Only local embeddings are used in formation and deletion. Historical source
metadata remain auditable; every readout reads current stored turn bodies.
"""
import copy
from dataclasses import asdict
import json

import numpy as np

from memcut.native_memory.contract import SearchResult, digest
from memcut.raw_turn_rag.embedding import validate_vectors
from memcut.source_anchored_systems.base import SourceBank, SourceObject
from memcut.source_anchored_v2.candidates import candidate_protocol
from memcut.source_anchored_v2 import PROTOCOL
from .text import chunk_session, coverage_identity, coverage_tokens, load_tokenizer, token_count


DELETION_POLICY = 'exact-original-turn-removal-session-rechunk-local-reembed-v1'


def _address(session_id):
    return json.dumps(['session_rag', 'session', session_id], separators=(',', ':'))


class SessionBank(SourceBank):
    system = 'session_rag'

    def __init__(self, settings, turns, embedder, tokenizer=None):
        super().__init__(settings, turns)
        if not turns or any(not isinstance(tid, str) or not isinstance(t.get('text'), str)
                            or not isinstance(t.get('session_id'), str) or not t['session_id']
                            for tid, t in turns.items()):
            raise ValueError('Require ordered original turns with nonempty session IDs')
        # A session may contain many turns, but cannot reappear after a later
        # session: shared import_source establishes the original order.
        sessions, previous = set(), None
        for turn in turns.values():
            sid = turn['session_id']
            if sid != previous and sid in sessions:
                raise ValueError('Session turns must be contiguous in source order')
            sessions.add(sid)
            previous = sid
        if type(settings['embedding']['max_seq_length']) is not int:
            raise ValueError('Invalid embedding sequence length')
        self.embedder = embedder
        self.tokenizer = tokenizer if tokenizer is not None else load_tokenizer(settings['embedding'])
        self.rows = {}
        self._source_sha256 = digest(self.turns)

    @property
    def split_sessions(self):
        return sum(len(row['chunks']) > 1 for row in self.rows.values())

    def _live_groups(self):
        groups = {}
        deleted = set(self.meta['deleted_turns'])
        for tid, turn in self.turns.items():
            if tid not in deleted:
                groups.setdefault(turn['session_id'], []).append(tid)
        return groups

    def _make_row(self, sid, tids):
        # Store each original body explicitly so residual never has to obtain
        # text from the complete, immutable auditor-only source copy.
        stored = [{'turn_id': tid, 'text': self.turns[tid]['text']} for tid in tids]
        return {'session_id': sid, 'turn_ids': list(tids), 'turns': stored,
                'text': '\n'.join(t['text'] for t in stored),
                'chunks': chunk_session(tids, self.turns, self.tokenizer,
                                        self.settings['embedding']['max_seq_length'])}

    def _encode_rows(self, rows):
        chunks = [chunk for row in rows.values() for chunk in row['chunks']]
        vectors = validate_vectors(self.embedder.encode([c['text'] for c in chunks]),
                                   len(chunks), self.settings['embedding']['dimensions'])
        for chunk, vector in zip(chunks, vectors):
            chunk['vector'] = vector.tolist()

    def native_objects(self):
        return [{'id': oid, 'native_id': row['session_id'], 'layer': 'session', 'kind': 'aggregate',
                 'text': row['text'], 'value': copy.deepcopy(row)} for oid, row in self.rows.items()]

    def dump_native(self):
        return {'rows': copy.deepcopy(self.rows)}

    def _validate_rows(self, rows):
        groups = self._live_groups()
        if self.progress['stage'] == 'NEW' and not rows:
            return
        if list(rows) != [_address(sid) for sid in groups]:
            raise ValueError('Live sessions do not exactly cover undeleted original turns')
        dimensions = self.settings['embedding']['dimensions']
        vectors = []
        for oid, row in rows.items():
            sid = row['session_id']
            expected = self._make_row(sid, groups[sid])
            plain = copy.deepcopy(row)
            for chunk in plain['chunks']:
                vectors.append(chunk.pop('vector'))
            if plain != expected or self.trace(oid) != set(groups[sid]):
                raise ValueError('Stored session text, chunks or active sources differ from live source turns')
        validate_vectors(vectors if vectors else np.empty((0, dimensions)), len(vectors), dimensions)

    def load_native(self, value):
        if set(value) != {'rows'}:
            raise ValueError('Unknown session snapshot fields')
        rows = copy.deepcopy(value['rows'])
        self._validate_rows(rows)
        self.rows = rows

    def restore_value(self, value):
        if digest(value.get('turns')) != self._source_sha256:
            raise ValueError('Snapshot source differs from original source')
        backup = copy.deepcopy(self.export())
        try:
            super().restore_value(value)
        except BaseException:
            super().restore_value(backup)
            raise

    def form(self):
        if self.rows or self.frozen or self.progress['stage'] != 'NEW':
            raise ValueError('Formation requires a new empty bank')
        rows = {_address(sid): self._make_row(sid, tids) for sid, tids in self._live_groups().items()}
        self._encode_rows(rows)
        writes = {oid: self.lineage.input(row['turn_ids']) for oid, row in rows.items()}
        self.observe_write(lambda: setattr(self, 'rows', rows), writes, label='original sessions stored; no generation')
        self.progress.update(stage='FULL', next_turn=len(self.turns))
        self._validate_rows(self.rows)
        self.lineage.validate(observed=self.observation())
        self.freeze_updates(True)

    def _query_vector(self, query):
        # Match RawTurn exactly: the pinned encoder applies its ordinary query
        # max_seq_length behavior. search() records any truncated query.
        return validate_vectors(self.embedder.encode([query]), 1, self.settings['embedding']['dimensions'])[0]

    @staticmethod
    def _cosines(vectors, query):
        matrix = np.asarray(vectors, dtype=np.float32)
        scores = np.einsum('ij,j->i', matrix, query, dtype=np.float64, optimize=False)
        if not np.isfinite(scores).all():
            raise ValueError('Non-finite cosine score')
        return scores

    async def search(self, query, *, top_k=20):
        if type(top_k) is not int or top_k <= 0:
            raise ValueError('top_k must be positive')
        objects, seeds = [], []
        scored = []
        if self.rows:
            q = self._query_vector(query)
            for oid, row in self.rows.items():
                cosines = self._cosines([c['vector'] for c in row['chunks']], q)
                best = int(np.argmax(cosines))
                scored.append((oid, float(cosines[best]), best))
        # Python sort is stable: tied scores retain original session order.
        for rank, (oid, score, best) in enumerate(sorted(scored, key=lambda x: -x[1])[:top_k], 1):
            row = self.rows[oid]
            objects.append(SourceObject(self.system, 'session', row['session_id'], row['text'], digest(row),
                kind='aggregate', meta={'session_id': row['session_id'], 'turn_ids': list(row['turn_ids'])}))
            seeds.append({'address': oid, 'rank': rank, 'distance': -score, 'cosine': score,
                          'best_chunk_index': best, 'session_id': row['session_id']})
        return SearchResult(query, ranked={'retrieved_document': objects}, native_context={},
            costs={'native_seed_hits': seeds, 'requested_top_k': top_k, 'actual_hits': len(objects),
                   'sessions_retrieved': [o.native_id for o in objects], 'retrieval': self.settings['retrieval'],
                   'session_score': 'maximum chunk cosine', 'split_sessions': self.split_sessions,
                   'query_embedding_tokens': token_count(query, self.tokenizer),
                   'query_embedding_truncated': token_count(query, self.tokenizer) > self.settings['embedding']['max_seq_length'],
                   'query_embedding_policy': 'identical to RawTurn pinned encoder max_seq_length', 'llm_calls': 0})

    def system_render(self, result):
        return '\n'.join(o.text for o in result.ranked['retrieved_document'])

    async def delete(self, obj_id):
        if obj_id not in self.rows:
            raise ValueError('Unknown session object')
        return await self.delete_turns(self.rows[obj_id]['turn_ids'])

    async def regenerate(self, *args, **kwargs):
        raise ValueError('Session RAG has no LLM regeneration; use exact delete_turns')

    async def delete_turns(self, selected):
        if not self.frozen:
            raise ValueError('Deletion requires a frozen Full-derived branch')
        added = set(self.lineage.ordered(selected)) - set(self.meta['deleted_turns'])
        affected = {oid for tid in added for oid in self.reverse_trace(tid)}
        backup, before = copy.deepcopy(self.export()), self.observation()
        start = len(self.meta['mutation_diffs'])
        try:
            replacements = {}
            for oid in self.rows:
                if oid in affected:
                    row = self.rows[oid]
                    tids = [tid for tid in row['turn_ids'] if tid not in added]
                    if tids:
                        replacements[oid] = self._make_row(row['session_id'], tids)
            self._encode_rows(replacements)
            if affected:
                writes = {oid: self.lineage.input(row['turn_ids']) for oid, row in replacements.items()}
                def replace():
                    self.rows = {oid: replacements[oid] if oid in replacements else row
                                 for oid, row in self.rows.items() if oid not in affected or oid in replacements}
                self.observe_write(replace, writes, rewrite=True, label=DELETION_POLICY)
            self.lineage.mark_deleted(added)
            after = self.observation()
            if any(after.get(oid) != before[oid] for oid in before if oid not in affected):
                raise ValueError('Deletion changed an unrelated session')
            self._validate_rows(self.rows)
            self.lineage.validate(observed=after)
        except BaseException:
            self.restore_value(backup)
            raise
        return {'deletion_policy': DELETION_POLICY, 'deletion_llm_calls': 0,
                'selected_turns': self.lineage.ordered(added), 'deleted_turns': self.lineage.ordered(added),
                'all_deleted_turns': list(self.meta['deleted_turns']),
                'deleted_objs': [oid for oid in before if oid not in after],
                'rewritten_objs': [oid for oid in before if oid in affected and oid in after],
                'created_objs': [], 'collateral_source_turns': [], 'collateral_raw_turns': [],
                'selected_turns_have_empty_der': all(not self.reverse_trace(t) for t in self.meta['deleted_turns']),
                'mutation_diffs': self.meta['mutation_diffs'][start:],
                'before_sha256': digest(before), 'after_sha256': digest(after)}

    def _stored_turns(self):
        return {t['turn_id']: t['text'] for row in self.rows.values() for t in row['turns']}

    async def residual_objects(self, turns):
        # Exact raw-turn views, not inherited whole-session source closure.
        requested = set(self.lineage.ordered(turns))
        live = self._stored_turns()
        return [SourceObject(self.system, 'turn_view', tid, live[tid], digest(live[tid]), kind='atomic',
                             meta={'turn_id': tid, 'session_id': self.turns[tid]['session_id']})
                for tid in self.turns if tid in requested and tid in live]

    async def contexts(self, claim):
        before = self.state_identity()
        native = await self.native_context(claim['stem'])
        evidence = set(self.lineage.ordered(claim['evidence_turns']))
        requested = evidence | set(self.lineage.ordered(claim['disclosure_turns']))
        residual = await self.residual_objects(requested)
        native_ids = [tid for obj in native.visible() for tid in self.rows[obj.address]['turn_ids']]
        context = native.native_context['prompt']
        episode_tokens = coverage_tokens('\n'.join(t['text'] for t in self.turns.values()))
        if not episode_tokens:
            raise ValueError('Episode has zero coverage tokens')
        context_tokens = coverage_tokens(context)
        result = {
            # Native empty memory still invokes the normal MC readout, as in
            # Turn RAG. Only evidence-derived residual emptiness is structural0.
            'native': {'context': context, 'obj_ids': [o.address for o in native.visible()],
                'retrieval': asdict(native), 'empty': False, 'turn_ids': native_ids,
                'sessions_retrieved': [o.native_id for o in native.visible()],
                'session_count': len(native.visible()), 'turn_count': len(native_ids),
                'context_tokens': context_tokens, 'episode_tokens': episode_tokens,
                'episode_coverage': context_tokens / episode_tokens,
                'token_counter': coverage_identity(), 'evidence_turns_in_ctx': len(evidence & set(native_ids)),
                'evidence_turn_ids_in_ctx': [tid for tid in native_ids if tid in evidence],
                'split_sessions': self.split_sessions},
            'residual': {'context': '\n'.join(o.text for o in residual),
                'obj_ids': [o.address for o in residual], 'turn_ids': [o.native_id for o in residual],
                'empty': not residual, 'evidence_derived_empty': not residual}}
        for row in result.values():
            row.update(context_sha256=digest(row['context']), state_identity=before)
        if self.state_identity() != before:
            raise ValueError('Context construction mutated stored memory')
        return result

    async def build_pool(self, request, disclosure_turns, *, n_obj=20, n_turn=20):
        candidate_protocol(PROTOCOL, n_obj, n_turn)
        excluded = set(self.lineage.ordered(disclosure_turns))
        if not excluded <= set(self.meta['deleted_turns']):
            raise ValueError('Candidate pool requires shared S0 with Direct turns removed')
        before = self.state_identity()
        result = await self.search(request, top_k=n_obj)
        live = self._stored_turns()
        candidates = {}
        for hit in result.costs['native_seed_hits']:
            row = self.rows[hit['address']]
            for tid in row['turn_ids']:
                if tid not in excluded:
                    candidates[tid] = {'turn_id': tid, 'session_id': row['session_id'],
                                       'session_obj_id': hit['address'], 'best_rank': hit['rank']}
        scored = []
        if candidates:
            q = self._query_vector(request)
            texts = [live[tid] for tid in candidates]
            # Exactly RawTurn's text/vector cache identity. Long candidate turns
            # retain the encoder's established truncation, explicitly logged;
            # their stored body and selector-visible text are never truncated.
            vectors = validate_vectors(self.embedder.encode(texts), len(texts), self.settings['embedding']['dimensions'])
            cosines = self._cosines(vectors, q)
            for row, text, score in zip(candidates.values(), texts, cosines):
                length = token_count(text, self.tokenizer)
                scored.append({**row, 'score': float(score), 'embedding_tokens': length,
                    'embedding_truncated': length > self.settings['embedding']['max_seq_length']})
        position = {tid: i for i, tid in enumerate(self.turns)}
        selected = sorted(scored, key=lambda r: (-r['score'], r['best_rank'], position[r['turn_id']]))[:n_turn]
        objects, rows = {}, []
        for row in selected:
            tid = row['turn_id']
            oid = json.dumps([self.system, 'turn_view', tid], separators=(',', ':'))
            # Existing selector_input displays only these synthetic views. Full
            # session text in native_retrieval is audit data, not selector input.
            objects[oid] = {'text': live[tid], 'score': row['score'], 'rank': row['best_rank'],
                            'version': digest({'turn_id': tid, 'text': live[tid]}),
                            'session_id': row['session_id'], 'source_session_object': row['session_obj_id']}
            rows.append({**row, 'exposed_obj_ids': [oid]})
        if self.state_identity() != before:
            raise ValueError('Candidate construction mutated shared S0')
        pool = {'request': request, 'state_identity': before, 'objects': objects, 'turns': rows,
                'native_retrieval': asdict(result), 'candidate_turns_before_cap': len(candidates),
                'requested_objects': n_obj, 'requested_candidate_turns': n_turn,
                'actual_objects': len(result.costs['native_seed_hits']), 'actual_candidate_turns': len(rows),
                'sessions_retrieved': result.costs['sessions_retrieved'],
                'turn_scores': copy.deepcopy(rows),
                'candidate_embedding_truncated_turns_before_cap': sum(r['embedding_truncated'] for r in scored),
                'candidate_scoring': 'single whole-turn cosine; identical RawTurn encoder max_seq_length; no gold labels'}
        return {**pool, 'pool_sha256': digest(pool)}
