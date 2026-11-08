"""Local execution only: isolated copies and selective residual materialization.

The original restore/transaction functions still own every validation and rollback.
No cached native state/hash, approximate copy, or shared mutable snapshot is used.
"""
import copy
from contextlib import ExitStack, contextmanager
import pickle
from unittest.mock import patch

from memcut.native_memory.contract import digest
from memcut.source_anchored_systems import base, lineage
from memcut.source_anchored_v2 import whole_object_deletion

POLICY = 'isolated-pickle-copies-native-storage-reuse-v1'


class SnapshotCopies:
    """Use the snapshot's existing serialization for large plain-data graphs.

    Only the known export and provenance envelopes take this path. Other copy
    calls, including custom native objects, keep Python's original semantics.
    Pickle roundtrip preserves ordering and internal aliases while isolating the
    entire graph from the caller, including on rollback and across branches.
    """
    def __getattr__(self, name):
        return getattr(copy, name)

    def deepcopy(self, value, memo=None):
        if memo is None and type(value) is dict and (
            {'format', 'system', 'native', 'lineage', 'recipes', 'turns', 'progress'} <= value.keys()
            or {'src', 'der', 'ever', 'touched', 'birth', 'history', 'mutation_diffs', 'observed', 'deleted_turns'} <= value.keys()
        ):
            return pickle.loads(pickle.dumps(value, protocol=5))
        return copy.deepcopy(value, memo)


async def residual_objects(bank, turns):
    ids = bank.lineage.residual_ids(turns)
    wanted = set(ids)
    # Still enumerate the real native storage. The provenance table identifies
    # objects, but never supplies their live text or substitutes for storage.
    objects = {}
    for row in bank.native_objects():
        if row['id'] in wanted:
            obj = base.SourceObject(bank.system, row['layer'], row['native_id'], row['text'],
                digest(row['value']), row.get('visibility', 'searchable'), row['kind'],
                copy.deepcopy(row['value']))
            objects[obj.address] = obj
    # Preserve birth order and the original KeyError on missing live objects.
    return [objects[oid] for oid in ids]


@contextmanager
def runtime(system):
    """One sequential native lifecycle per process; API answer workers use no bank."""
    if system not in {'nemori', 'lightmem'}:
        yield
        return
    with ExitStack() as stack:
        copier = SnapshotCopies()
        for module in (base, lineage, whole_object_deletion):
            stack.enter_context(patch.object(module, 'copy', copier))
        stack.enter_context(patch.object(base.SourceBank, 'residual_objects', residual_objects))
        if system == 'nemori':
            from .nemori_storage import runtime as native_runtime
        else:
            from .lightmem_storage import runtime as native_runtime
        stack.enter_context(native_runtime())
        yield
