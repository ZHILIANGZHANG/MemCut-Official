"""SimpleMem's native leniency where MemCut was stricter than upstream, and a reproducible lexical index.

- Retrieval planner replies: a missing field takes upstream's default (HybridRetriever's .get(..., default)), so
  the retriever receives the raw reply and applies it; a field that is present must still have a usable type.
  More than three missing-information queries are executed, as upstream does.
- A keyword query Tantivy cannot parse ("Lin Yue's", "time: 5pm", "-x") gives no lexical hits, as upstream's
  VectorStore.keyword_search does, and the search log records the rejected query; it no longer stops the run.
- A time expression dateparser cannot handle gives no time range, as upstream's try/except does.
- The full-text index is written by one Tantivy thread: with several, documents land in segments in a different
  order on every rebuild (every restore and every deletion rebuilds it), and BM25 ties come back reordered.
"""
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

from memcut.simplemem_s012.client import validate_native_result  # noqa: F401 - the frozen validator, kept importable


def _strings(value, key):
    rows = value.get(key)
    if key in value and (not isinstance(rows, list) or any(not isinstance(x, str) for x in rows)):
        raise ValueError(f'Invalid native retrieval {key}')


def validate_result(value, system):
    """Reject what upstream would silently turn into junk; accept what it defaults."""
    if not isinstance(value, dict):
        raise ValueError('Native retrieval requires a JSON object')
    if 'information requirement analyst' in system:
        # Every field is read with .get(key, default) and only written into the next prompt.
        _strings(value, 'key_entities')
        _strings(value, 'relationships')
        if 'required_info' in value and not isinstance(value['required_info'], list):
            raise ValueError('Invalid native information requirements')
        if 'question_type' in value and not isinstance(value['question_type'], str):
            raise ValueError('Invalid native information plan')
        if isinstance(value.get('minimal_queries_needed'), (dict, list)):
            raise ValueError('Invalid native information plan')
    elif 'query analysis assistant' in system:
        for key in ('keywords', 'persons', 'entities'):
            _strings(value, key)
        for key in ('time_expression', 'location'):
            if value.get(key) is not None and not isinstance(value[key], str):
                raise ValueError(f'Invalid native query field: {key}')
    elif 'query generation specialist' in system:
        _strings(value, 'queries')
    elif 'information completeness evaluator' in system:
        pass    # .get('assessment', 'incomplete'); anything but "complete" reads as incomplete
    elif 'missing information query generator' in system:
        _strings(value, 'targeted_queries')
    else:
        raise ValueError('Unexpected native retrieval request type')


def keyword_search(self, keywords, top_k=3):
    from memcut.simplemem_s012.client import NativeAbort
    if not self.backend._fts_initialized:
        raise NativeAbort('Native lexical retrieval failed: Cannot query a partially built FTS index')
    try:
        results = self.backend.keyword_search(keywords, top_k)
    except ValueError as exc:   # Tantivy refused the query text: upstream returns no lexical hits
        with self.call_lock:
            self.calls.append({'path': 'keyword', 'keywords': list(keywords), 'top_k': top_k, 'hits': [],
                               'query_rejected': str(exc)[:300]})
        return []
    try:
        return self._entries('keyword', results, keywords=list(keywords), top_k=top_k)
    except Exception as exc:
        raise NativeAbort(f'Native lexical retrieval failed: {exc}') from exc


class _OneWriterThread:
    """A Tantivy index whose writer uses one thread: one segment, documents in table order."""

    def __init__(self, index):
        self._index = index

    def writer(self, heap_size=128_000_000, num_threads=0):
        return self._index.writer(heap_size=heap_size, num_threads=1)

    def __getattr__(self, name):
        return getattr(self._index, name)


@contextmanager
def native():
    """Upstream behaviours restored in the native SimpleMem stages (build, delete)."""
    import lancedb.fts
    from memcut.simplemem_s012 import native as store
    populate = lancedb.fts.populate_index
    parse = store.Retriever._parse_time_range

    def populate_index(index, *args, **kwargs):
        return populate(_OneWriterThread(index), *args, **kwargs)

    def parse_time_range(self, expression):
        try:
            return parse(self, expression)
        except Exception:  # noqa: BLE001 - upstream: "Time parsing failed", no time filter
            return None

    with ExitStack() as stack:
        stack.enter_context(patch.object(lancedb.fts, 'populate_index', populate_index))
        stack.enter_context(patch.object(store.Store, 'keyword_search', keyword_search))
        stack.enter_context(patch.object(store.Retriever, '_parse_time_range', parse_time_range))
        yield
