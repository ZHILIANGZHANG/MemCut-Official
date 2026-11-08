"""Native LightMem formation with automatic input lineage and committed resume."""
from __future__ import annotations

from contextlib import contextmanager, redirect_stdout, redirect_stderr
import copy
import json
from pathlib import Path
import threading

from .contract import digest
from .lightmem_v1 import native_modules, object_from_row, read, UPSTREAM
from .persistence import write_json


def settings(bank, assets, *, model, base_url, key='runtime-only'):
    return {
        'pre_compress': True, 'pre_compressor': {'model_name': 'llmlingua-2', 'configs': {
            'llmlingua_config': {'model_name': assets['compressor'], 'device_map': 'cpu', 'use_llmlingua2': True},
            'compress_config': {'instruction': '', 'rate': .7, 'target_token': -1}}},
        'topic_segment': True, 'precomp_topic_shared': True, 'topic_segmenter': {'model_name': 'llmlingua-2'},
        'messages_use': 'user_only', 'metadata_generate': True, 'text_summary': True,
        'extraction_mode': 'flat', 'extract_threshold': .1,
        'memory_manager': {'model_name': 'openai', 'configs': {'model': model, 'api_key': key,
            'openai_base_url': base_url, 'max_tokens': 16000, 'temperature': .1, 'top_p': .1}},
        'index_strategy': 'embedding', 'retrieve_strategy': 'embedding', 'update': 'offline',
        'text_embedder': {'model_name': 'huggingface', 'configs': {'model': assets['embedder'],
            'embedding_dims': 384, 'model_kwargs': {'device': 'cpu'}}},
        'embedding_retriever': {'model_name': 'qdrant', 'configs': {'collection_name': 'details',
            'path': str(Path(bank) / 'details'), 'embedding_model_dims': 384, 'on_disk': True}},
        'summary_retriever': {'model_name': 'qdrant', 'configs': {'collection_name': 'summaries',
            'path': str(Path(bank) / 'summaries'), 'embedding_model_dims': 384, 'on_disk': True}},
        'logging': {'level': 'ERROR', 'console_level': 'ERROR', 'file_enabled': False}}


class NativeFormationStopped(BaseException):
    """Escape upstream broad Exception handlers after a stopped model dispatch."""


class Lineage:
    def __init__(self, entries=None):
        self.entries = copy.deepcopy(entries or {})

    def created(self, obj, input_ids, origin_ids, *, event):
        if obj.native_id in self.entries:
            raise ValueError('New native object reused a lineage identity')
        self.entries[obj.native_id] = {'version': obj.version, 'content_sha256': digest(obj.text),
            'input_source_ids': sorted(set(input_ids) | set(origin_ids)),
            'origin_source_ids': sorted(set(origin_ids)), 'parents': [], 'created_event': event}

    def updated(self, obj, parents, before, *, event):
        if obj.native_id not in before or not parents:
            raise ValueError('Update has no known target/input versions')
        source_ids = set()
        for parent in parents:
            info = before.get(parent['id'])
            if info is None or parent['version'] != info['version']:
                raise ValueError('Update parent differs from actual input version')
            source_ids.update(info['input_source_ids'])
        self.entries[obj.native_id] = {**before[obj.native_id], 'version': obj.version,
            'content_sha256': digest(obj.text),
            'input_source_ids': sorted(source_ids), 'parents': parents, 'updated_event': event}

    def refresh_versions(self, rows):
        for row in rows:
            obj = object_from_row(row)
            if obj.native_id not in self.entries:
                raise ValueError('Untraced native object was stored')
            self.entries[obj.native_id]['version'] = obj.version
            self.entries[obj.native_id]['content_sha256'] = digest(obj.text)


class FormationTrace:
    """Observe the real converter and write inputs; never alter memory text."""
    def __init__(self, lineage):
        self.lineage = lineage
        self.inputs, self.sequence, self.fact_inputs, self.created_notes = [], {}, {}, []
        self.event = None

    @contextmanager
    def install(self):
        from lightmem.memory import lightmem as lm
        from lightmem.memory import utils
        assign, convert, create = lm.assign_sequence_numbers_with_timestamps, lm.convert_extraction_results_to_memory_entries, utils._create_memory_entry_from_fact

        def observe_assign(extract_list, *args, **kwargs):
            result = assign(extract_list, *args, **kwargs)
            self.inputs = [{m['_memcut_source_id'] for seg in batch for m in seg}
                           for batch in extract_list]
            self.sequence = {m['sequence_number']: m['_memcut_source_id']
                             for batch in extract_list for seg in batch for m in seg}
            return result

        def observe_convert(extracted_results, *args, **kwargs):
            self.fact_inputs = {}
            for index, result in enumerate(extracted_results):
                if not result:
                    continue
                if index >= len(self.inputs):
                    raise ValueError('Extraction response has no corresponding actual input batch')
                for group in result.get('cleaned_result') or []:
                    for fact in group if isinstance(group, list) else [group]:
                        self.fact_inputs[id(fact)] = self.inputs[index]
            return convert(extracted_results, *args, **kwargs)

        def observe_create(fact_entry, *args, **kwargs):
            note = create(fact_entry, *args, **kwargs)
            if note is not None:
                inputs = self.fact_inputs.get(id(fact_entry))
                if inputs is None:
                    raise ValueError('Untraced extracted fact')
                # Match the upstream converter's actual global sequence lookup,
                # even when a model source_id is wrong. Do not repair native facts.
                source = self.sequence.get(int(fact_entry.get('source_id', 0)) * 2)
                self.created_notes.append((note, set(inputs), [] if source is None else [source]))
            return note

        lm.assign_sequence_numbers_with_timestamps = observe_assign
        lm.convert_extraction_results_to_memory_entries = observe_convert
        utils._create_memory_entry_from_fact = observe_create
        try:
            yield
        finally:
            lm.assign_sequence_numbers_with_timestamps = assign
            lm.convert_extraction_results_to_memory_entries = convert
            utils._create_memory_entry_from_fact = create

    def commit(self, rows):
        objects = {str(r['id']): object_from_row(r) for r in rows}
        added = []
        for note, inputs, origins in self.created_notes:
            # The upstream may change a colliding UUID during insertion.
            obj = objects[str(note.id)]
            self.lineage.created(obj, inputs, origins, event=self.event)
            added.append({'id': obj.native_id, 'version': obj.version,
                          'input_source_ids': sorted(inputs), 'origin_source_ids': origins})
        self.lineage.refresh_versions(rows)
        self.created_notes.clear()
        return added


def _save_checkpoint(bank, memory, lineage, next_turn, stage):
    from lightmem.memory import lightmem as lm
    value = {'next_turn': next_turn, 'stage': stage, 'topic_idx': lm.GLOBAL_TOPIC_IDX,
        'token_stats': memory.token_stats,
        'sensory': {k: copy.deepcopy(getattr(memory.senmem_buffer_manager, k))
                    for k in ['buffer', 'big_buffer', 'token_count']},
        'short': {k: copy.deepcopy(getattr(memory.shortmem_buffer_manager, k))
                  for k in ['buffer', 'token_count']}}
    write_json(Path(bank) / 'LINEAGE.json', lineage.entries)
    write_json(Path(bank) / 'CHECKPOINT.json', value)


def form(bank, source, manifest, assets, chat, *, audit_folder, limit_turns=0, memory_factory=None):
    """Resume committed source turns; an interrupted mutation always stops replay.

    memory_factory/chat are injected only by local tests. CLI uses real native
    components and an explicitly budgeted transport. No annotations are accepted.
    """
    native_modules()
    from lightmem.memory import lightmem as lm
    import tiktoken
    import importlib.util
    spec = importlib.util.spec_from_file_location('memcut_lightmem_locomo_prompts', UPSTREAM / 'experiments/locomo/prompts.py')
    prompts = importlib.util.module_from_spec(spec); spec.loader.exec_module(prompts)
    bank, audit_folder = Path(bank), Path(audit_folder)
    audit_folder.mkdir(parents=True, exist_ok=True)
    if (bank / 'FROZEN.json').exists():
        raise ValueError('Formation cannot reopen a frozen bank')
    if (bank / 'INFLIGHT.json').exists():
        raise ValueError('Partial formation needs reconciliation; do not replay paid operations')
    if (bank / 'CHECKPOINT.json').exists() and read(bank / 'CHECKPOINT.json')['stage'] == 'FULL':
        return {'stage': 'FULL', 'turns': read(bank / 'CHECKPOINT.json')['next_turn']}
    config = settings(bank, assets, model=manifest['model'], base_url=manifest['base_url'])
    memory = (memory_factory or lm.LightMemory.from_config)(config)
    memory.shortmem_buffer_manager.tokenizer = tiktoken.encoding_for_model('gpt-4o-mini')
    serialization_lock = threading.Lock()
    failures = []

    def generate(messages, response_format=None, tools=None, tool_choice='auto'):
        if tools:
            raise NativeFormationStopped('Unexpected tools in native LightMem formation')
        with serialization_lock:
            if failures:
                raise NativeFormationStopped('A previous native model operation failed')
            try:
                response = chat({'messages': messages, 'temperature': .1, 'top_p': .1,
                    'max_tokens': 16000, **({'response_format': response_format} if response_format else {})})
                choice = response['choices'][0]
                if choice['finish_reason'] != 'stop' or not choice['message'].get('content'):
                    raise ValueError('Incomplete native generation; no output-cap retry')
                # Native parsing otherwise silently turns malformed JSON into
                # zero extracted facts or an ignored update.
                parsed = json.loads(choice['message']['content'])
                if not isinstance(parsed, dict) or not (
                    isinstance(parsed.get('data'), list) or parsed.get('action') in {'ignore', 'update', 'delete'}):
                    raise ValueError('Invalid native extraction/update response')
                if parsed.get('action') == 'update' and not isinstance(parsed.get('new_memory'), str):
                    raise ValueError('Update response has no memory text')
                return choice['message']['content'], response.get('usage', {})
            except BaseException as exc:
                failures.append(type(exc).__name__)
                raise NativeFormationStopped(type(exc).__name__) from None

    memory.manager.generate_response = generate
    original_extract = memory.manager.meta_text_extract

    def checked_extract(*args, **kwargs):
        results = original_extract(*args, **kwargs)
        if any(not r or not r.get('output_prompt') for r in results):
            failures.append('invalid_native_extraction')
            raise NativeFormationStopped('Native extraction failed before storing a usable result')
        return results

    memory.manager.meta_text_extract = checked_extract
    checkpoint = read(bank / 'CHECKPOINT.json') if (bank / 'CHECKPOINT.json').exists() else {'next_turn': 0, 'stage': 'INGEST', 'topic_idx': 0}
    lineage = Lineage(read(bank / 'LINEAGE.json') if (bank / 'LINEAGE.json').exists() else {})
    lm.GLOBAL_TOPIC_IDX = checkpoint['topic_idx']
    for name, obj in [('sensory', memory.senmem_buffer_manager), ('short', memory.shortmem_buffer_manager)]:
        for attr, value in checkpoint.get(name, {}).items():
            setattr(obj, attr, value)
    if 'token_stats' in checkpoint:
        memory.token_stats = checkpoint['token_stats']
    turns = [(session, turn) for session in source['sessions'] for turn in session['turns']]
    stop = min(limit_turns, len(turns)) if limit_turns else len(turns)
    trace = FormationTrace(lineage)
    try:
        with (audit_folder / 'native.log').open('a') as log, redirect_stdout(log), redirect_stderr(log), trace.install():
            for index in range(checkpoint['next_turn'], stop):
                session, turn = turns[index]
                event = f'turn-{index:05d}'
                write_json(bank / 'INFLIGHT.json', {'event': event, 'source_id': turn['id']})
                trace.event = event
                meta = {'time_stamp': session['timestamp'], 'speaker_id': turn['speaker'],
                        'speaker_name': turn['speaker'], '_memcut_source_id': turn['id']}
                result = memory.add_memory(
                    [{'role': 'user', 'content': turn['text'], **meta}, {'role': 'assistant', 'content': '', **meta}],
                    METADATA_GENERATE_PROMPT=prompts.METADATA_GENERATE_PROMPT_locomo,
                    force_segment=index == len(turns) - 1, force_extract=index == len(turns) - 1)
                if failures:
                    raise NativeFormationStopped('Native formation swallowed a failed model operation')
                added = trace.commit(memory.embedding_retriever.get_all(with_vectors=True))
                _save_checkpoint(bank, memory, lineage, index + 1, 'INGEST')
                write_json(audit_folder / 'turns' / (event + '.json'), {'source_id': turn['id'],
                    'created': added, 'native_api_calls': result.get('api_call_nums'),
                    'input_batches': [sorted(ids) for ids in trace.inputs] if added else []})
                (bank / 'INFLIGHT.json').unlink()
                write_json(audit_folder / 'PROGRESS.json', {'stage': 'INGEST', 'turns': index + 1,
                    'total_turns': len(turns), 'entries': len(lineage.entries)})
            if stop < len(turns):
                return {'stage': 'PREFIX', 'turns': stop, 'total_turns': len(turns)}
            if checkpoint['stage'] != 'FULL':
                write_json(bank / 'INFLIGHT.json', {'event': 'offline-update'})
                # Upstream discards executor.map results, so capture actual
                # queue writes and failures before any paid consolidation.
                store = memory.embedding_retriever
                original_write, original_search = store.update, store.search
                queue_writes = set()

                def queue_write(vector_id, *args, **kwargs):
                    try:
                        result = original_write(vector_id, *args, **kwargs)
                        queue_writes.add(str(vector_id))
                        return result
                    except BaseException as exc:
                        failures.append(type(exc).__name__)
                        raise

                def queue_search(*args, **kwargs):
                    try:
                        return original_search(*args, **kwargs)
                    except BaseException as exc:
                        failures.append(type(exc).__name__)
                        raise

                eligible = {str(r['id']) for r in store.get_all(with_vectors=True)
                            if r.get('vector') is not None and r['payload'].get('float_time_stamp') is not None}
                store.update, store.search = queue_write, queue_search
                try:
                    memory.construct_update_queue_all_entries(top_k=20, keep_top_n=10, max_workers=1)
                finally:
                    store.update, store.search = original_write, original_search
                if failures or queue_writes != eligible:
                    raise NativeFormationStopped('Incomplete native update-queue construction')
                before_rows = memory.embedding_retriever.get_all(with_vectors=True)
                lineage.refresh_versions(before_rows)
                before = copy.deepcopy(lineage.entries)
                events = []
                original_update = memory.manager._call_update_llm

                def update(prompt, target, sources):
                    try:
                        response = original_update(prompt, target, sources)
                        parents = [{'id': str(r['id']), 'version': object_from_row(r).version} for r in [target, *sources]]
                        events.append({'target_id': str(target['id']), 'parents': parents, 'response': response})
                        return response
                    except BaseException as exc:
                        failures.append(type(exc).__name__)
                        raise

                memory.manager._call_update_llm = update
                try:
                    memory.offline_update_all_entries(score_threshold=.9, max_workers=1)
                    if failures:
                        raise NativeFormationStopped('Native update swallowed a failed model operation')
                finally:
                    memory.manager._call_update_llm = original_update
                after_rows = memory.embedding_retriever.get_all(with_vectors=True)
                after = {str(r['id']): object_from_row(r) for r in after_rows}
                for event in events:
                    response = event['response'] or {}
                    if response.get('action') == 'update':
                        if (event['target_id'] not in after or
                            json.loads(after[event['target_id']].text)['text'].get('memory') != response['new_memory']):
                            raise RuntimeError('Native update response was not committed')
                        lineage.updated(after[event['target_id']], event['parents'], before, event='offline-update')
                    elif response.get('action') == 'delete' and event['target_id'] in after:
                        raise RuntimeError('Native consolidation delete was not committed')
                lineage.refresh_versions(after_rows)
                # Native final flush must leave no unevaluated conversation buffer.
                buffers = {'sensory_buffer': bool(memory.senmem_buffer_manager.buffer),
                    'sensory_big_buffer': bool(memory.senmem_buffer_manager.big_buffer),
                    'short_buffer': bool(memory.shortmem_buffer_manager.buffer)}
                write_json(audit_folder / 'OFFLINE_UPDATE.json', {'events': events, 'buffers_after_flush': buffers})
                if any(buffers.values()):
                    raise RuntimeError('Native final flush left buffered input; retain bank for inspection')
                if memory.summary_retriever.get_all():
                    raise RuntimeError('Unexpected second active memory layer in pinned flat profile')
                _save_checkpoint(bank, memory, lineage, len(turns), 'FULL')
                write_json(audit_folder / 'FORMATION_COMPLETE.json', {'stage': 'FULL', 'turns': len(turns),
                    'entries': len(after), 'native_stats': memory.get_token_statistics()})
                (bank / 'INFLIGHT.json').unlink()
        return {'stage': 'FULL', 'turns': len(turns)}
    finally:
        memory.embedding_retriever.client.close()
        memory.summary_retriever.client.close()
