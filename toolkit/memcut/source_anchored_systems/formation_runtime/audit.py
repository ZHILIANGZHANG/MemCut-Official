"""Read-only checks of parallel LightMem prefix receipts and source mappings."""
from collections import Counter
import json
from pathlib import Path

from memcut.native_memory.contract import digest
from ..lightmem_sources import fact_sources
from .delta import replay_diffs


def lightmem_prefix(bank, folder):
    from .render import render
    folder = Path(folder)
    observed = bank.observation()
    bank.lineage.validate(observed=observed)
    if replay_diffs(bank.lineage.data['mutation_diffs']) != observed:
        raise ValueError('Diff replay differs from actual native storage')
    if bank.recipes['updates']:
        raise ValueError('Prefix audit must precede final consolidation')
    live = {str(r['id']): r for r in bank.rows(with_vectors=False)}
    rows, batches, covered = [], [], set()
    for recipe in bank.recipes['batches'].values():
        rid, = recipe['request_ids']
        request = json.loads((folder / 'requests' / rid / 'request.json').read_text())
        response = json.loads((folder / 'requests' / rid / 'response.json').read_text())
        if digest(request) != rid:
            raise ValueError('Changed request receipt')
        messages = render({'topic_id_mapping': [recipe['native_topic_ids']],
            'extract_list': [recipe['segments']], 'system_prompt': bank.extraction_prompt()}, 0)
        if messages != recipe['native_request_messages'] or messages != request['request']['messages']:
            raise ValueError('Stored extraction batch differs from actual model input')
        value = json.loads(response['choices'][0]['message']['content'])
        facts = [f for group in value['data'] for f in (group if isinstance(group, list) else [group])]
        if len(facts) != len(recipe['slots']) or len(facts) != len(recipe['base_rows']):
            raise ValueError('Extraction silently lost or added native objects')
        for ident, base, fact in zip(recipe['slots'], recipe['base_rows'], facts):
            oid = bank.oid('long_term', ident)
            sources = bank.lineage.ordered(fact_sources(fact, recipe['segments']))
            if (live[ident]['payload'] != base['payload'] or base['payload']['memory'] != fact['fact']
                    or bank.lineage.data['src'][oid] != sources or recipe['slot_sources'][ident] != sources):
                raise ValueError('Native fact/source links differ from actual generation')
            rows.append({'obj_id': oid, 'memory': fact['fact'], 'source_turn_ids': sources,
                'batch_turn_ids': recipe['roots'], 'request_id': rid,
                'source_turns': [bank.turns[t] for t in sources]})
        covered.update(recipe['roots'])
        batches.append({'request_id': rid, 'input_turns': recipe['roots'], 'facts': len(facts),
                        'trigger_turn': recipe['trigger_turn']})
    if len(rows) != len(live):
        raise ValueError('Unaccounted native objects')
    buffered = bank.lineage.ordered(m['_memcut_source_id'] for values in bank._buffers().values()
                                   for m in values if m.get('content'))
    return {'status': 'PASS', 'completed_turns': bank.progress['next_turn'],
        'long_term_objects': len(rows), 'actual_extraction_requests': len(batches),
        'diff_steps': len(bank.lineage.data['mutation_diffs']), 'covered_extraction_turns': len(covered),
        'buffer_turn_ids': buffered, 'source_count_distribution': dict(Counter(len(r['source_turn_ids']) for r in rows)),
        'source_policy': bank.settings.get('source_policy'), 'issues': [], 'objects': rows, 'batches': batches,
        'scope': 'Actual native batches, receipts and source declarations; not semantic truth certification.'}
