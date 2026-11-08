"""Native bulk windows and extraction; inline roots add no attribution request."""
import copy
import uuid

from memcut.native_memory.contract import digest
from .client import NativeClient, NativeAbort, decode_json
from .upstream import modules


def windows(turn_ids, size=40, overlap=2, *, final=True):
    if not 0 <= overlap < size:
        raise ValueError('Invalid native sliding window')
    result, pos = [], 0
    while pos + size <= len(turn_ids):
        result.append(list(turn_ids[pos:pos + size]))
        pos += size - overlap
    if final and turn_ids[pos:]:
        result.append(list(turn_ids[pos:]))
    return result


SOURCE_INSTRUCTION = '''
[Source attribution extension]
Each JSON entry must also include "source_ids": [1, ...] and "source_memory_ids": [].
Use only the printed [turn N] and [memory N] labels. List ALL and ONLY contributing
inputs to the entire entry, including restatement and metadata, pronoun resolution,
speaker identification, dates and cross-turn synthesis. source_ids references turns;
source_memory_ids references contributing previous memories. Their union must be
nonempty. Reading an input is not sufficient reason to attribute it. Do not attribute
all window turns by default. Both fields are required on EVERY entry. source_ids
and source_memory_ids are arrays of distinct integer labels, not strings. timestamp,
location and topic must each be a string or null, never an array. keywords, persons
and entities are arrays of strings. Return only the JSON array; no separate analysis.
'''


def extraction_request(turns, ids, parents=()):
    modules()
    from simplemem.core.memory_builder import MemoryBuilder
    from simplemem.core.models.memory_entry import Dialogue
    # Keep native timestamp/speaker rendering, adding only source labels.
    dialogue_text = '\n'.join(f'[turn {i}] ' + str(Dialogue(dialogue_id=i,
        speaker=turns[tid].get('speaker') or turns[tid]['role'],
        content=turns[tid]['text'], timestamp=turns[tid]['timestamp'])) for i, tid in enumerate(ids, 1))
    context = ''
    if parents:
        context = '\n[Previous Window Memory Entries (for reference to avoid duplication)]\n'
        context += ''.join(f'- [memory {i}] {p["text"]}\n' for i, p in enumerate(parents, 1))
    prompt = MemoryBuilder._build_extraction_prompt(None, dialogue_text, list(range(1, len(ids)+1)), context)
    # Extend the pinned output schema and examples consistently; the v1
    # footer-only extension was frequently ignored in actual GLM responses.
    # Edit only the template suffix, never similarly worded source dialogue.
    head, marker, schema = prompt.rpartition('[Output Format]')
    if not marker:
        raise ValueError('Pinned extraction template changed')
    changes = {
        '"topic": "topic phrase"': '"topic": "topic phrase",\n    "source_ids": [1],\n    "source_memory_ids": []',
        '[2025-11-15T14:30:00] Alice:': '[turn 1] [2025-11-15T14:30:00] Alice:',
        '[2025-11-15T14:31:00] Bob:': '[turn 2] [2025-11-15T14:31:00] Bob:',
        '"topic": "Product discussion meeting arrangement"': '"topic": "Product discussion meeting arrangement",\n    "source_ids": [1],\n    "source_memory_ids": []',
        '"topic": "Meeting preparation confirmation"': '"topic": "Meeting preparation confirmation",\n    "source_ids": [1, 2],\n    "source_memory_ids": []',
    }
    for before, after in changes.items():
        if schema.count(before) != 1:
            raise ValueError('Pinned extraction schema/example changed')
        schema = schema.replace(before, after)
    prompt = head + marker + schema
    return [
        {'role': 'system', 'content': 'You are a professional information extraction assistant, skilled at extracting structured, unambiguous information from conversations. You must output valid JSON format.' + SOURCE_INSTRUCTION},
        {'role': 'user', 'content': prompt}]


def parse_entries(text, ids, parents, *, window_identity):
    from simplemem.core.models.memory_entry import MemoryEntry
    rows = decode_json(text)
    if not isinstance(rows, list):
        raise ValueError('Extraction must return a JSON array')
    result = []
    def indices(row, key, size):
        value = row.get(key)
        if not isinstance(value, list) or any(type(i) is not int or not 1 <= i <= size for i in value) or len(value) != len(set(value)):
            raise ValueError(f'Invalid inline {key}; no whole-window fallback')
        return value
    allowed = set(MemoryEntry.model_fields) - {'entry_id'}
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) - allowed - {'source_ids', 'source_memory_ids'}:
            raise ValueError('Unknown generated memory fields')
        sources = indices(row, 'source_ids', len(ids))
        source_parents = indices(row, 'source_memory_ids', len(parents))
        if not sources and not source_parents:
            raise ValueError('Unattributed fact; no silent batch association')
        if not isinstance(row.get('lossless_restatement'), str) or not row['lossless_restatement'].strip():
            raise ValueError('Empty generated fact')
        entry = MemoryEntry.model_validate({k: v for k, v in row.items() if k in allowed}, strict=True)
        # Storage IDs never enter prompts. Deterministic IDs make exact replay,
        # insertion order and independent restoration stable across processes.
        entry.entry_id = str(uuid.uuid5(uuid.NAMESPACE_URL, digest({
            'window': window_identity, 'slot': i, 'entry': entry.model_dump(exclude={'entry_id'})})))
        roots = {ids[n-1] for n in sources}
        for n in source_parents:
            roots.update(parents[n-1]['turns'])
        result.append({'entry': entry.model_dump(), 'turns': sorted(roots),
            'source_ids': sources, 'source_memory_ids': source_parents,
            'parents': [copy.deepcopy(parents[n-1]) for n in source_parents]})
    return result


def extract(turns, ids, cache, settings, *, window_index, parents=()):
    client = NativeClient(cache, settings['model'], max_tokens=settings['write_max_tokens'], extraction=True)
    messages = extraction_request(turns, ids, parents)
    identity = digest({'index': window_index, 'turn_ids': ids, 'messages': messages})
    try:
        text = client.chat_completion(messages)
        entries = parse_entries(text, ids, parents, window_identity=identity)
    except NativeAbort as exc:
        raise RuntimeError('SimpleMem extraction failed; checkpoint and receipts retained') from exc
    return {'window_index': window_index, 'window_identity': identity,
            'input_turn_ids': list(ids), 'input_parents': copy.deepcopy(list(parents)),
            'request_id': client.requests[0]['request_id'], 'entries': entries}
