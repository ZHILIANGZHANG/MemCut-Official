"""Actual LanceDB/Tantivy paths, with explicit errors and deterministic merging."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import copy
import threading

import numpy as np

from memcut.raw_turn_rag.embedding import LocalEmbeddings
from .client import NativeAbort
from .upstream import modules

modules()
from simplemem.core.database.vector_store import VectorStore
from simplemem.core.database.vector_store_backend import LanceDBVectorStoreBackend
from simplemem.core.models.memory_entry import MemoryEntry
from simplemem.core.hybrid_retriever import HybridRetriever
from simplemem.core.answer_generator import AnswerGenerator


class LocalEmbedding:
    """Native embedding interface over the existing pinned local MiniLM cache."""
    def __init__(self, settings, cache_dir):
        self.local = LocalEmbeddings(settings, cache_dir)
        self.dimension = settings['dimensions']
        self.lock = threading.Lock()

    def encode_documents(self, texts):
        with self.lock:
            return self.local.encode(texts)

    def encode_single(self, text, is_query=False):
        return self.encode_documents([text])[0]


class Backend(LanceDBVectorStoreBackend):
    def _init_fts_index(self):
        # Native bulk ingestion inserts once. The resumable adapter inserts
        # ordered batches, then creates this same index once over the Full.
        # Never freeze an index covering just the first checkpoint batch.
        return

    def build_fts(self):
        if self.count():
            self.table.create_fts_index('lossless_restatement', use_tantivy=True,
                                        tokenizer_name='en_stem', replace=True)
        self._fts_initialized = True


class Store(VectorStore):
    def __init__(self, path, embedder, table_name):
        super().__init__(db_path=str(path), embedding_model=embedder, table_name=table_name,
            backend_factory=lambda dimension: Backend(str(path), table_name, dimension))
        self.calls, self.call_lock = [], threading.Lock()

    def _entries(self, kind, results, **arguments):
        entries = [MemoryEntry.model_validate({'entry_id': r.entry_id,
            **{k: v for k, v in r.metadata.items() if k in MemoryEntry.model_fields}}, strict=True) for r in results]
        if len({e.entry_id for e in entries}) != len(entries):
            raise ValueError('Native retrieval returned duplicate IDs within a path')
        with self.call_lock:
            self.calls.append({'path': kind, **arguments, 'hits': [
                {'entry_id': r.entry_id, 'score': r.score, 'rank': i+1} for i, r in enumerate(results)]})
        return entries

    def semantic_search(self, query, top_k=5):
        try:
            if not self.backend.count():
                return self._entries('semantic', [], query=query, top_k=top_k)
            vector = self.embedding_model.encode_single(query, is_query=True)
            return self._entries('semantic', self.backend.semantic_search(vector.tolist(), top_k),
                                 query=query, top_k=top_k)
        except Exception as exc:
            raise NativeAbort(f'Native semantic retrieval failed: {exc}') from exc

    def keyword_search(self, keywords, top_k=3):
        try:
            if not self.backend._fts_initialized:
                raise ValueError('Cannot query a partially built FTS index')
            return self._entries('keyword', self.backend.keyword_search(keywords, top_k),
                                 keywords=list(keywords), top_k=top_k)
        except Exception as exc:
            raise NativeAbort(f'Native lexical retrieval failed: {exc}') from exc

    def structured_search(self, persons=None, timestamp_range=None, location=None, entities=None, top_k=None):
        try:
            arguments = dict(persons=persons, timestamp_range=timestamp_range,
                             location=location, entities=entities, top_k=top_k)
            return self._entries('structured', self.backend.structured_search(**arguments), **arguments)
        except Exception as exc:
            raise NativeAbort(f'Native structured retrieval failed: {exc}') from exc


class Retriever(HybridRetriever):
    """Keep native planners, all three searches, deduplication and reflection.

    The upstream serial path defines query order. executor.map retains that
    order while overlapping local searches, avoiding as_completed rank drift.
    """
    def __init__(self, client, store, settings, *, semantic_k=None, reference_time):
        super().__init__(client, store,
            semantic_top_k=semantic_k or settings['semantic_top_k'],
            keyword_top_k=settings['keyword_top_k'], structured_top_k=settings['structured_top_k'],
            enable_planning=settings['enable_planning'], enable_reflection=settings['enable_reflection'],
            max_reflection_rounds=settings['max_reflection_rounds'],
            enable_parallel_retrieval=True, max_retrieval_workers=settings['max_retrieval_workers'])
        self.reference_time = reference_time
        self.output_cap = semantic_k
        self.prefix_stop = None

    def _retrieve_with_planning(self, query, enable_reflection=None):
        if self.output_cap is None:
            return super()._retrieve_with_planning(query, enable_reflection)
        # Native merge only appends unseen IDs; later paths/reflections cannot
        # displace an existing prefix. Stop only after that exact prefix is full.
        # QA has no final cap and always uses the complete upstream path above.
        plan = self._analyze_information_requirements(query)
        queries = self._generate_targeted_queries(query, plan)
        merged = []
        def add(entries, stage):
            nonlocal merged
            merged = self._merge_and_deduplicate_entries(merged + entries)
            if len(merged) >= self.output_cap:
                self.prefix_stop = stage
                return True
            return False
        for i, text in enumerate(queries):
            if add(self._semantic_search(text), f'semantic-query-{i+1}'):
                return merged
        analysis = self._analyze_query(query)
        if add(self._keyword_search(query, analysis), 'keyword'):
            return merged
        if add(self._structured_search(analysis), 'structured'):
            return merged
        reflection = self.enable_reflection if enable_reflection is None else enable_reflection
        return self._retrieve_with_intelligent_reflection(query, merged, plan) if reflection else merged

    def _execute_parallel_searches(self, queries):
        with ThreadPoolExecutor(max_workers=self.max_retrieval_workers) as pool:
            return [entry for result in pool.map(self._semantic_search, queries) for entry in result]

    def _execute_parallel_additional_searches(self, queries, round_num):
        return self._execute_parallel_searches(queries)

    def _parse_time_range(self, expression):
        import dateparser
        # Same native day/week interpretation, with a frozen source clock.
        parsed = dateparser.parse(expression, settings={'PREFER_DATES_FROM': 'past',
            'RELATIVE_BASE': datetime.fromisoformat(self.reference_time)})
        if not parsed:
            return None
        start, end = parsed.replace(hour=0, minute=0, second=0), parsed.replace(hour=23, minute=59, second=59)
        if 'week' in expression.lower() or '周' in expression:
            start, end = start - timedelta(days=7), end + timedelta(days=7)
        return start.isoformat(), end.isoformat()


def native_render(entries):
    return AnswerGenerator._format_contexts(None, entries)


def stored_text(entry):
    # Full stored information, without source labels, vectors or internal IDs.
    # The native QA renderer intentionally omits keywords; residual must not.
    text = native_render([entry]).split('\n', 1)[1]
    if entry.keywords:
        text += '\nKeywords: ' + ', '.join(entry.keywords)
    return text
