"""Reuse frozen native BM25 tokenizations; keep Chroma and deletion unchanged.

Install only for an explicitly versioned execution plan.  No saved artifacts or
native modules are edited.  Snapshot hashes, full native equality checks and
transaction rollback remain the responsibility of the original SourceBank.
"""
from __future__ import annotations

import copy
import hashlib
import pickle
from contextlib import contextmanager, ExitStack
from unittest.mock import patch
from weakref import WeakKeyDictionary


POLICY = 'nemori-saved-lexical-tokens-batched-delete-v1'


def _tokens_from_snapshot(value):
    """Bind saved tokens to exact native objects, never to a retrieval result."""
    cache = {}
    for name, collection, id_key in (('episode', 'episodes', 'episode_id'),
                                      ('semantic', 'semantics', 'memory_id')):
        saved = value.get('lexical', {}).get(name)
        if saved is None:
            continue  # Older snapshots still use the original tokenizer.
        data, tokens = saved['data'], saved['tokens']
        if len(data) != len(tokens):
            raise ValueError('Saved lexical objects and tokenizations differ in length')
        native = {row[id_key]: row for row in value[collection]}
        ids = [row[id_key] for row in data]
        id_set = set(ids)
        if len(ids) != len(id_set) or any(native.get(row[id_key]) != row for row in data):
            raise ValueError('Saved lexical objects differ from native storage')
        if ids != [row[id_key] for row in value[collection] if row[id_key] in id_set]:
            raise ValueError('Saved lexical object order differs from native storage')
        if bool(data) != (saved['index'] is not None):
            raise ValueError('Saved lexical index/data presence differs')
        for row, encoded in zip(data, tokens):
            if not isinstance(encoded, list) or not encoded or any(not isinstance(t, str) for t in encoded):
                raise ValueError('Invalid saved native tokenization')
            text = f"{row['title']} {row['content']}" if name == 'episode' else row['content']
            frozen = tuple(encoded)
            if text in cache and cache[text] != frozen:
                raise ValueError('Equal native text has conflicting saved tokenizations')
            cache[text] = frozen
    return cache


def _check_restored_lexical(engine, user, value):
    """The rebuilt BM25 statistics must equal the frozen native index exactly."""
    for name, saved in value.get('lexical', {}).items():
        data = getattr(engine, name + '_data').get(user, [])
        tokens = getattr(engine, name + '_tokenized_texts').get(user, [])
        index = getattr(engine, name + '_indices').get(user)
        if ([obj.to_dict() for obj in data] != saved['data'] or tokens != saved['tokens'] or
                (vars(index) if index is not None else None) != saved['index']):
            raise ValueError('Reused lexical tokens did not reconstruct the frozen native index')


def _validate_saved_index(saved, validated, stats):
    """First encounter independently reconstructs statistics from native tokens."""
    from rank_bm25 import BM25Okapi
    key = hashlib.sha256(pickle.dumps((saved['tokens'], saved['index']), protocol=5)).digest()
    if key not in validated:
        expected = vars(BM25Okapi(saved['tokens'])) if saved['tokens'] else None
        if expected != saved['index']:
            raise ValueError('Saved native BM25 statistics do not match the saved tokenizations')
        validated.add(key)
        stats['lexical_index_validations'] = stats.get('lexical_index_validations', 0) + 1


def _load_saved_index(engine, name, user, objects, saved, stats, validated):
    """Restore the native index already in the sealed snapshot, independently."""
    from rank_bm25 import BM25Okapi
    id_key = 'episode_id' if name == 'episode' else 'memory_id'
    lookup = {row[id_key]: row for row in saved['data']}
    kept = []
    for obj in objects:
        ident = getattr(obj, id_key)
        if ident in lookup:
            if obj.to_dict() != lookup[ident]:
                raise ValueError('Restored native object differs from its saved lexical object')
            kept.append(obj)
        else:
            text = f'{obj.title} {obj.content}' if name == 'episode' else obj.content
            if engine._tokenize(text):
                raise ValueError('Saved lexical index omitted a native object with nonempty tokens')
    if [getattr(obj, id_key) for obj in kept] != [row[id_key] for row in saved['data']]:
        raise ValueError('Restored lexical object order differs')
    _validate_saved_index(saved, validated, stats)
    if kept:
        # BM25 has no external/native resources. Its complete native instance
        # state was serialized by dump_native and covered by the snapshot seal.
        index = BM25Okapi.__new__(BM25Okapi)
        vars(index).update(copy.deepcopy(saved['index']))
        getattr(engine, name + '_indices')[user] = index
        getattr(engine, name + '_data')[user] = kept
        getattr(engine, name + '_tokenized_texts')[user] = [list(row) for row in saved['tokens']]
        stats['restored_tokenizations'] = stats.get('restored_tokenizations', 0) + len(kept)
        stats['restored_lexical_indices'] = stats.get('restored_lexical_indices', 0) + 1


@contextmanager
def _deferred_removals(engine, stats):
    """An erase-only transaction has no reader until both indexes are rebuilt."""
    from rank_bm25 import BM25Okapi
    dirty = set()

    def remover(name, id_key):
        def remove(user, ident):
            data = getattr(engine, name + '_data')
            tokens = getattr(engine, name + '_tokenized_texts')
            indices = getattr(engine, name + '_indices')
            objects = data.get(user, [])
            encoded = tokens.get(user, [])
            if not objects:
                return False
            if len(objects) != len(encoded):
                raise ValueError('Native BM25 object/token alignment changed before deletion')
            kept = [(obj, text) for obj, text in zip(objects, encoded) if getattr(obj, id_key) != ident]
            if len(kept) == len(objects):
                return False
            stats['lexical_objects_removed'] = stats.get('lexical_objects_removed', 0) + len(objects) - len(kept)
            if kept:
                data[user] = [obj for obj, _ in kept]
                tokens[user] = [text for _, text in kept]
                dirty.add((name, user))
            else:
                for table in (data, tokens, indices):
                    table.pop(user, None)
                dirty.discard((name, user))
            return True
        return remove

    def no_read(*args, **kwargs):
        raise RuntimeError('Native retrieval cannot run inside an unfinished deletion transaction')

    with ExitStack() as stack:
        stack.enter_context(patch.object(engine, 'remove_episode', remover('episode', 'episode_id')))
        stack.enter_context(patch.object(engine, 'remove_semantic_memory', remover('semantic', 'memory_id')))
        stack.enter_context(patch.object(engine, 'search_episodes', no_read))
        stack.enter_context(patch.object(engine, 'search_semantic_memories', no_read))
        yield
        for name, user in sorted(dirty):
            encoded = getattr(engine, name + '_tokenized_texts')[user]
            getattr(engine, name + '_indices')[user] = BM25Okapi(encoded)
            stats['lexical_delete_rebuilds'] = stats.get('lexical_delete_rebuilds', 0) + 1


@contextmanager
def optimize_nemori_storage(*, stats=None):
    """Scope patches to one serial native-storage execution (not thread safe).

    Original ``load_native`` still restores repositories, buffers and Chroma in
    the original order.  At its BM25 initialization boundary, already serialized
    native indexes are restored with independent mutable data/tokens/statistics;
    all are checked against the snapshot. No retrieval method is replaced.
    Only the current whole-object deletion's erase transaction batches BM25
    rebuilds.  Per-object native repository/vector erasure and checks still run.
    Formation and individual deletion calls retain their existing behavior.
    """
    from memcut.source_anchored_systems.nemori import NemoriBank
    from memcut.source_anchored_v2.whole_object_deletion import POLICY as DELETE_POLICY
    stats = {} if stats is None else stats
    validated = set()
    tokenizations = WeakKeyDictionary()
    original_load = NemoriBank.load_native
    original_observe = NemoriBank.observe_write

    def load(bank, value):
        expected_tokens = _tokens_from_snapshot(value)
        lexical = bank.engine.bm25_search
        verified_tokens = tokenizations.setdefault(lexical, {})
        for text, expected in expected_tokens.items():
            if text not in verified_tokens:
                verified_tokens[text] = tuple(lexical._tokenize(text))
                stats['lexical_text_validations'] = stats.get('lexical_text_validations', 0) + 1
            if verified_tokens[text] != expected:
                raise ValueError('Saved lexical tokens differ from the original native tokenizer')
        with ExitStack() as stack:
            for name, method in (('episode', 'index_episodes'), ('semantic', 'index_semantic_memories')):
                saved = value.get('lexical', {}).get(name)
                if saved is not None:
                    def restore(user, objects, name=name, saved=saved):
                        return _load_saved_index(lexical, name, user, objects, saved, stats, validated)
                    stack.enter_context(patch.object(lexical, method, restore))
            result = original_load(bank, value)
        _check_restored_lexical(lexical, bank.user, value)
        return result

    def observe(bank, operation, inputs, *, rewrite=False, label=None):
        if label != DELETE_POLICY or rewrite:
            return original_observe(bank, operation, inputs, rewrite=rewrite, label=label)

        def erase_and_rebuild():
            with _deferred_removals(bank.engine.bm25_search, stats):
                return operation()
        return original_observe(bank, erase_and_rebuild, inputs, rewrite=False, label=label)

    with patch.object(NemoriBank, 'load_native', load), patch.object(NemoriBank, 'observe_write', observe):
        yield stats


runtime = optimize_nemori_storage
