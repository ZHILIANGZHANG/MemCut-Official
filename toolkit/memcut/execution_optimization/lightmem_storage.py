"""Same LightMem/Qdrant operations, fewer scroll calls and SQLite commits.

This opt-in execution layer does not recreate collections or replace retrieval.
The frozen adapter still verifies every restored field, vector and source link.
Only the pinned local Qdrant backend has its per-point commits coalesced; other
backends retain their ordinary durability behavior. A failed transaction closes
the client after rollback because its in-memory points may be partially changed.
"""
from __future__ import annotations

from contextlib import contextmanager
from collections import OrderedDict
import copy
from importlib.metadata import version
import pickle
import sqlite3
import threading
from unittest.mock import patch
from weakref import WeakKeyDictionary


POLICY = 'lightmem-same-collection-batched-io-v1'
FEATURES = ('native-scroll-pages-4096', 'native-bulk-delete',
            'local-qdrant-1.15.1-restore-transaction', 'exact-local-embedding-memo-512')
QDRANT_TRANSACTION_VERSION = '1.15.1'
SCROLL_PAGE_SIZE = 4096


class _DeferredCommit:
    def __init__(self, connection):
        self.connection = connection

    def commit(self):
        # Only the owning scope can commit the complete native replacement.
        return None

    def __getattr__(self, name):
        return getattr(self.connection, name)


@contextmanager
def _native_transaction(store):
    from qdrant_client.local.qdrant_local import QdrantLocal

    local = getattr(store.client, '_client', None)
    if type(local) is not QdrantLocal or version('qdrant-client') != QDRANT_TRANSACTION_VERSION:
        yield
        return
    collection = local._get_collection(store.collection_name)
    persistence = collection.storage
    if persistence is None:
        yield
        return
    connection = persistence.storage
    if type(connection) is not sqlite3.Connection or connection.in_transaction:
        raise RuntimeError('Qdrant storage has an unexpected or unfinished transaction')
    connection.execute('BEGIN')
    persistence.storage = _DeferredCommit(connection)
    try:
        yield
        connection.commit()
    except BaseException as exc:
        try:
            connection.rollback()
        finally:
            persistence.storage = connection
            # Disk rolled back; do not allow partially updated RAM to be read.
            store.client.close()
        exc.add_note('LightMem restore rolled back its SQLite transaction and closed the '
                     'partially changed client; create a new bank before continuing.')
        raise
    finally:
        persistence.storage = connection


def rows(bank, *, with_vectors=True):
    """Identical native scroll order/fields, with fewer pagination round trips."""
    store = bank.memory.embedding_retriever
    result, offset = [], None
    while True:
        page, offset = store.client.scroll(collection_name=store.collection_name,
            scroll_filter=None, limit=SCROLL_PAGE_SIZE, with_payload=True,
            with_vectors=with_vectors, offset=offset)
        result.extend(point.model_dump() for point in page)
        if offset is None:
            return result


def load_native(bank, value):
    from qdrant_client.models import PointIdsList

    store = bank.memory.embedding_retriever
    ids = [str(row['id']) for row in bank.rows(with_vectors=False)]
    with _native_transaction(store):
        if ids:
            store.client.delete(collection_name=store.collection_name,
                points_selector=PointIdsList(points=ids), wait=True)
        if value['rows']:
            bank._insert(value['rows'])
    for group, manager in [('sensory', bank.memory.senmem_buffer_manager),
                           ('short', bank.memory.shortmem_buffer_manager)]:
        for key, item in value[group].items():
            setattr(manager, key, copy.deepcopy(item))
    bank.topic_index = value['topic_index']
    bank.memory.token_stats = copy.deepcopy(value['token_stats'])


def _memoized_embed(original, max_entries=512):
    """Cache local inference only; keep logical call stats and independent values."""
    caches, lock = WeakKeyDictionary(), threading.RLock()

    def embed(self, *args, **kwargs):
        if self.use_api or self.config.huggingface_base_url or self.model.training:
            return original(self, *args, **kwargs)
        config = self.config.model_dump()
        key = pickle.dumps((id(self.model), config, args, kwargs), protocol=5)
        with lock:
            cache = caches.setdefault(self, OrderedDict())
            if key in cache:
                self.total_calls += 1
                cache.move_to_end(key)
                return copy.deepcopy(cache[key])
            result = original(self, *args, **kwargs)
            cache[key] = copy.deepcopy(result)
            if len(cache) > max_entries:
                cache.popitem(last=False)
            return result
    return embed


@contextmanager
def runtime():
    """Install around sequential bank mutations; independent QA may run in parallel."""
    from memcut.source_anchored_systems.lightmem import LightMemBank
    from memcut.native_memory.lightmem_v1 import native_modules
    native_modules()
    from lightmem.factory.text_embedder.huggingface import TextEmbedderHuggingface

    with patch.object(LightMemBank, 'rows', rows), patch.object(LightMemBank, 'load_native', load_native), \
         patch.object(TextEmbedderHuggingface, 'embed', _memoized_embed(TextEmbedderHuggingface.embed)):
        yield


lightmem_storage = runtime
