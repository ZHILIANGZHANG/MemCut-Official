"""Model-declared object sources, returned in existing native generation calls."""
import copy

POLICY = 'lightmem-inline-multisource-v1'

EXTRACTION_INSTRUCTION = '''
Source tracking extension (required for every fact in data):
Keep source_id as the primary message number for native timestamp metadata.
Also return source_ids: a nonempty JSON array of ALL message numbers whose
content contributes to this fact, including messages needed to resolve names,
pronouns, references or combine facts. Include source_id in source_ids.
Use only message numbers actually present in THIS input, exactly as printed;
they need not start at zero. Do not include unrelated messages merely because
they were processed together. Never return the whole batch by default.
Example output shape: {"data":[{"source_id":5,"source_ids":[2,5],"fact":"..."}]}.
Return facts and source_ids together in this response; no reasoning or explanation.
'''

UPDATE_INSTRUCTION = '''
Source tracking extension: each input memory is labelled [memory N].
For action=update, also return source_memory_ids: a nonempty JSON array of the
input memory numbers whose information is retained or used in new_memory.
Include memory 0 only if the target contributes information; include a candidate
only if its information contributes, not merely because it was offered.
When retaining the unchanged target, return action=ignore. For ignore/delete,
source_memory_ids is unnecessary. Keep new_memory plain text without memory
labels or source lists. Return the action, text and sources together, no reasoning.
'''


def numbers(value):
    if not isinstance(value, list) or not value:
        raise ValueError('Missing nonempty source list')
    out = []
    for item in value:
        if type(item) is int and item >= 0:
            number = item
        elif isinstance(item, str) and item.isascii() and item.isdigit():
            number = int(item)
        else:
            raise ValueError('Invalid source number')
        if number in out:
            raise ValueError('Duplicate source number')
        out.append(number)
    return out


def fact_sources(fact, messages):
    """Resolve within this actual native API batch, never by list offset."""
    visible = {m['sequence_number']//2: m['_memcut_source_id']
               for segment in messages for m in segment
               if m['role'] == 'user' and m.get('content')}
    ids = numbers(fact.get('source_ids'))
    primary = numbers([fact.get('source_id')])[0]
    if primary not in ids or not set(ids) <= visible.keys():
        raise ValueError('Fact references a source outside its actual input batch')
    return list(dict.fromkeys(visible[i] for i in ids))


def source_protocol(protocol, system):
    result = copy.deepcopy(protocol)
    if system == 'lightmem':
        result['version'] += '-lightmem-inline-sources-v1'
        result['provenance'] = 'native storage diffs plus model-declared sources in existing generation calls; full reads retained for audit'
        result['source_attribution'] = {
            'policy': POLICY,
            'extraction': 'source_ids in the native fact-generation response',
            'consolidation': 'source_memory_ids in the native update response; inherit contributing parent versions',
            'audit': 'full input batches/parents retained separately from active src/der',
            'extra_attribution_requests': 0,
            'qualification': 'model-declared attribution, not proven semantic causality',
        }
    return result


async def source_links(bank, *, query=None, turn_id=None, object_id=None, top_k=20):
    """Read-only diagnostic lookup. Source text is never added to QA/search."""
    if sum(x is not None for x in (query,turn_id,object_id)) != 1:
        raise ValueError('Choose one query, turn or object')
    before = bank.state_identity()
    if query is not None:
        found = await bank.search(query,top_k=top_k)
        ids = [obj.address for obj in found.ranked['retrieved_document']]
    elif turn_id is not None:
        if turn_id not in bank.turns:
            raise ValueError('Unknown source turn')
        ids = sorted(bank.reverse_trace(turn_id), key=bank.meta['birth'].__getitem__)
    else:
        ids = [object_id]
    native = {obj['id']:obj for obj in bank.native_objects()}
    if not set(ids) <= native.keys():
        raise ValueError('Object is not live in this snapshot')
    rows = []
    for oid in ids:
        sources = bank.lineage.ordered(bank.trace(oid))
        if any(oid not in bank.reverse_trace(t) for t in sources):
            raise ValueError('Forward/reverse source relation disagrees')
        rows.append({'object_id':oid,'layer':native[oid]['layer'],'text':native[oid]['text'],
                     'source_turn_ids':sources,'source_turns':[copy.deepcopy(bank.turns[t]) for t in sources]})
    if bank.state_identity() != before:
        raise ValueError('Source lookup modified memory')
    return {'source_policy':bank.settings.get('source_policy','input-ancestry'),
            'query':query,'turn_id':turn_id,'object_id':object_id,'objects':rows,'llm_calls':0}
