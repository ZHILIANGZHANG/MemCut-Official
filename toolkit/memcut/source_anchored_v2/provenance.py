"""Storage observations only: no provenance lookup and no model calls."""
import json

from memcut.native_memory.contract import digest


def snapshot_fields(fields):
    """Hash authoritative native fields, including fields absent from an index.

    Empty global strings are MemoryBank's required empty containers, not live
    memories. Hash stored values too, so metadata-only changes are observable.
    """
    result = {}
    for layer, ident, text, value, visibility in fields:
        if layer in {'overall_history', 'overall_personality'} and value == '':
            continue
        oid = json.dumps(['memorybank', layer, ident], separators=(',', ':'))
        if oid in result:
            raise ValueError('Duplicate native object address')
        result[oid] = {'text_sha256': digest(text), 'stored_sha256': digest(value)}
    return result


def diff_objects(before, after):
    """Discover creates, modifications and deletions from two independent views."""
    return {
        'created': [oid for oid in after if oid not in before],
        'modified': [oid for oid in after if oid in before and after[oid] != before[oid]],
        'deleted': [oid for oid in before if oid not in after],
    }
