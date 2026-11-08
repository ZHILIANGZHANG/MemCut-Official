"""Native MemoryBank fields plus turn provenance and transactional closure edits."""
import copy
import fcntl
import hashlib
import inspect
import json
import os
import pickle
import tempfile
import uuid
from pathlib import Path

from memcut.native_memory.contract import digest
from memcut.native_memory.memorybank import MemoryBankAdapter, make_memorybank
from . import PROTOCOL
from .completions import native_summary_request
from .provenance import diff_objects, snapshot_fields


def address(layer, ident):
    return json.dumps(['memorybank', layer, ident], separators=(',', ':'))


def import_source(source):
    """Keep the existing native pair/session layout; track each role separately."""
    turns, groups, order = {}, {}, []
    for session in source['sessions']:
        date = session['timestamp']
        rows = session['turns']
        if date in groups or not rows or len(rows) % 2:
            raise ValueError('Require unique session timestamps and complete native pairs')
        pairs = []
        for i, row in enumerate(rows):
            if row['id'] in turns or row['role'] != ('user' if i % 2 == 0 else 'assistant'):
                raise ValueError('Require unique original turn IDs and alternating roles')
            turns[row['id']] = {**row, 'timestamp': date, 'session_id': session['id']}
            order.append(row['id'])
            if i % 2 == 0:
                pairs.append([row['id'], rows[i+1]['id']])
        groups[date] = pairs
    if not turns:
        raise ValueError('Empty episode')
    return turns, groups, order


class SourceMemoryBank(MemoryBankAdapter):
    def __init__(self, native, prompts, qa, user_id, checkpoint, summaries, model):
        super().__init__(native, prompts, qa, user_id, checkpoint)
        self.summaries, self.model, self.meta = summaries, model, {}

    def save(self):
        """Native index and provenance commit in one atomic file replacement."""
        import faiss
        if self.checkpoint.with_suffix('.frozen.json').exists():
            raise ValueError('Frozen snapshot; restore into an independent path')
        self.assert_provenance()
        state = {
            'format_version': 1, 'user_id': self.user_id, 'memory': self.memory,
            'pair_ids': self.pair_ids, 'doc_addresses': self.doc_addresses,
            'retrieval_config': self.retrieval_config(),
            'index': faiss.serialize_index(self.store.index) if self.store is not None else None,
            'docstore': self.store.docstore if self.store is not None else None,
            'index_ids': self.store.index_to_docstore_id if self.store is not None else None,
            'source_anchored_v2': self.meta,
        }
        self.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        payload = pickle.dumps(state)
        with self.checkpoint.with_suffix('.lock').open('a+b') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            actual = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest() if self.checkpoint.exists() else None
            if actual != self._checkpoint_hash:
                raise RuntimeError('Checkpoint changed outside this branch')
            temp = self.checkpoint.with_suffix('.tmp')
            with temp.open('wb') as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            temp.replace(self.checkpoint)
            self._checkpoint_hash = hashlib.sha256(payload).hexdigest()

    def restore(self, snap_id=None):
        if snap_id is not None:
            src = Path(snap_id).resolve()
            if src == self.checkpoint.resolve() or self.checkpoint.exists():
                raise ValueError('Restore requires a new independent branch path')
            frozen = json.loads(src.with_suffix('.frozen.json').read_text())
            payload = src.read_bytes()
            if hashlib.sha256(payload).hexdigest() != frozen['sha256']:
                raise ValueError('Snapshot changed')
            self.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            self.checkpoint.write_bytes(payload)
        super().restore()
        state = pickle.loads(self.checkpoint.read_bytes())
        self.meta = state.get('source_anchored_v2', {})
        if (self.meta.get('protocol') != PROTOCOL or self.meta.get('model') != self.model
                or not self.meta.get('frozen_updates')):
            raise ValueError('Not a compatible source-anchored MemoryBank snapshot')
        if self.meta.get('formation_identity') != self.formation_identity():
            raise ValueError('Formation implementation or native settings changed')
        self.assert_provenance()

    def formation_identity(self):
        request = native_summary_request(self.summaries, 'identity-only capture', self.model)
        return {'source_digest': self.meta['source_digest'], 'model': self.model,
                'write_transport': self.meta['write_transport'],
                'generation': {k: v for k, v in request.items() if k != 'messages'},
                'retrieval': self.retrieval_config(),
                'native_source': {name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                                  for name, module in [('storage', self.native), ('summary', self.summaries), ('qa', self.prompts)]},
                'formation_code': digest([inspect.getsource(fn) for fn in
                    (import_source, native_summary_request, SourceMemoryBank.form,
                     SourceMemoryBank._source_pairs, SourceMemoryBank._generate,
                     SourceMemoryBank._record, SourceMemoryBank.object_snapshot,
                     SourceMemoryBank._write_sources, SourceMemoryBank._observe_write,
                     snapshot_fields, diff_objects, MemoryBankAdapter.fields,
                     MemoryBankAdapter._build_index)])}

    def snapshot(self, destination):
        from memcut.native_memory.persistence import write_json
        destination = Path(destination)
        payload = self.checkpoint.read_bytes()
        if destination.exists() and destination.read_bytes() != payload:
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            temp = destination.with_suffix('.tmp')
            with temp.open('wb') as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            temp.replace(destination)
        write_json(destination.with_suffix('.frozen.json'), {'sha256': hashlib.sha256(payload).hexdigest(),
                   'protocol': PROTOCOL, 'native_state': self.state_identity()})
        return str(destination)

    def freeze_updates(self, flag=True):
        if flag is not True:
            raise ValueError('Evaluation branches always disable autonomous writes/forgetting')
        self.meta['frozen_updates'] = True

    def state_identity(self):
        return digest({'memory': self.memory, 'pair_ids': self.pair_ids,
                       'src': self.meta['src'], 'ever': self.meta['ever'],
                       'deleted_turns': self.meta['deleted_turns']})

    def _sync(self, observed=None):
        self.meta['der'], self.meta['touched'] = {}, {}
        for oid, roots in self.meta['src'].items():
            for t in roots:
                self.meta['der'].setdefault(t, []).append(oid)
            for t in self.meta['ever'][oid]:
                self.meta['touched'].setdefault(t, []).append(oid)
        self.assert_provenance(observed)

    def assert_provenance(self, observed=None):
        for date, pairs in self.memory[self.user_id]['history'].items():
            if len(pairs) != len(self.pair_ids[date]):
                raise ValueError('Raw pairs and stable native IDs differ')
        # _observe_write already took the independent after-write snapshot.
        # Only metadata changes between that observation and this assertion;
        # reuse it here. Save/restore and external assertions still rescan.
        if observed is None:
            observed = self.object_snapshot()
        if observed != self.meta['object_snapshot']:
            raise ValueError('Unobserved native storage mutation')
        live = set(observed)
        if live != set(self.meta['src']):
            raise ValueError('Provenance does not cover exactly the live native fields')
        expected, touched = {}, {}
        known_turns, deleted_turns = set(self.meta['turns']), set(self.meta['deleted_turns'])
        for oid, roots in self.meta['src'].items():
            root_set = set(roots)
            if not root_set <= known_turns or not root_set <= set(self.meta['ever'][oid]):
                raise ValueError('Unknown/lost source turn')
            if root_set & deleted_turns:
                raise ValueError('Selected source remains in active provenance')
            for t in roots:
                expected.setdefault(t, []).append(oid)
            for t in self.meta['ever'][oid]:
                touched.setdefault(t, []).append(oid)
        if expected != self.meta['der'] or touched != self.meta['touched']:
            raise ValueError('src/der/touched invariant violated')

    def trace(self, oid):
        return set(self.meta['src'].get(oid, []))

    def reverse_trace(self, turn):
        return set(self.meta['der'].get(turn, []))

    async def list_all(self):
        live = self.object_snapshot()
        return [o for o in await super().audit_objects() if o.address in live]

    def object_snapshot(self):
        # The FAISS docstore is refreshed after a transaction, so its rendered
        # text can be stale during a write. Observe native storage directly.
        return snapshot_fields(self.fields())

    def _write_sources(self, turns=(), *, parents=(), request=None):
        roots = set(turns)
        for parent in parents:
            if parent not in self.meta['src']:
                raise ValueError('Unknown generation input parent')
            roots.update(self.trace(parent))
        if not roots or not roots <= set(self.meta['turns']):
            raise ValueError('Generation inputs must resolve to known source turns')
        return {'turns': [t for t in self.meta['turn_order'] if t in roots],
                'parents': list(parents), 'request_id': request}

    def _observe_write(self, before, inputs, *, rewrite=False):
        """Let the snapshot diff decide which objects changed, then bind inputs.

        Ordinary ingestion conservatively unions ancestry. Regeneration from
        remaining inputs replaces current ancestry, retaining historical roots.
        An explicit write can reproduce identical bytes; record that separately
        instead of falsely reporting a content change or retaining removed roots.
        """
        if before != self.meta['object_snapshot']:
            raise ValueError('Unobserved native storage mutation before write')
        after = self.object_snapshot()
        changes = diff_objects(before, after)
        changed_outputs = set(changes['created']) | set(changes['modified'])
        if changed_outputs - set(inputs):
            raise ValueError('Snapshot diff found outputs without generation inputs')
        if set(inputs) - set(after):
            raise ValueError('Declared write output is absent from native storage')
        unchanged_writes = [oid for oid in inputs if oid not in changed_outputs]
        # Validate all bindings before applying any provenance edits.
        for binding in inputs.values():
            roots = set(binding['turns'])
            if not roots or not roots <= set(self.meta['turns']):
                raise ValueError('Unknown/empty source binding')
            if any(p not in before or not self.trace(p) <= roots for p in binding['parents']):
                raise ValueError('Input ancestry omits a live parent source')
        receipt = {'step': len(self.meta['mutation_diffs']),
                   'before_sha256': digest(before), 'after_sha256': digest(after),
                   'before_count': len(before), 'after_count': len(after),
                   **changes, 'unchanged_writes': unchanged_writes,
                   'objects': {oid: {'before': before.get(oid), 'after': after.get(oid)}
                               for oid in dict.fromkeys([*changes['created'], *changes['modified'],
                                                         *changes['deleted'], *unchanged_writes])}}
        for oid in changes['deleted']:
            self._record('delete', oid, [], observation=receipt['step'])
        for oid in [*changes['created'], *changes['modified'], *unchanged_writes]:
            binding = inputs[oid]
            op = 'create' if oid in changes['created'] else ('rewrite' if rewrite else 'modify')
            roots = set(binding['turns'])
            if op == 'modify':
                roots.update(self.trace(oid))
            self._record(op, oid, roots, request=binding['request_id'], parents=binding['parents'],
                         observation=receipt['step'])
        self.meta['object_snapshot'] = after
        self.meta['mutation_diffs'].append(receipt)
        self._sync(after)
        return receipt

    def _source_pairs(self, date, allowed):
        rows = []
        for q, a in self.meta['groups'][date]:
            if q in allowed or a in allowed:
                rows.append({'query': self.meta['turns'][q]['text'] if q in allowed else '',
                             'response': self.meta['turns'][a]['text'] if a in allowed else ''})
        return rows

    def _generate(self, layer, ident, memory, allowed, completions):
        mod = self.summaries
        if layer == 'summary':
            prompt = mod.summarize_content_prompt(self._source_pairs(ident, allowed), self.user_id, 'AI', 'en')
        elif layer == 'personality':
            prompt = mod.summarize_person_prompt(self._source_pairs(ident, allowed), self.user_id, 'AI', 'en')
        elif layer == 'overall_history':
            prompt = mod.summarize_overall_prompt(list(memory['summary'].items()), 'en')
        elif layer == 'overall_personality':
            prompt = mod.summarize_overall_personality(list(memory['personality'].items()), 'en')
        else:
            raise ValueError('Only derived fields can be regenerated')
        response = completions.complete(native_summary_request(mod, prompt, self.model), allow_length=True)
        return response['text'], response['request_id']

    async def form(self, source, completions):
        if self.checkpoint.exists():
            raise FileExistsError('Full build cannot overwrite an existing bank')
        turns, groups, order = import_source(source)
        self.meta = {'protocol': copy.deepcopy(PROTOCOL), 'model': self.model, 'turns': turns,
                     'write_transport': {'endpoint': completions.endpoint, 'compatibility': completions.compatibility},
                     'groups': groups, 'turn_order': order, 'source_digest': digest(source),
                     'src': {}, 'der': {}, 'touched': {}, 'ever': {}, 'history': [],
                     'object_snapshot': {}, 'mutation_diffs': [],
                     'birth': {}, 'deleted_turns': [], 'frozen_updates': True}
        self.memory = {self.user_id: {'history': {}, 'summary': {}, 'personality': {},
                                     'overall_history': '', 'overall_personality': ''}}
        user, all_turns = self.memory[self.user_id], set(turns)
        for date, pairs in groups.items():
            user['history'][date], self.pair_ids[date] = [], []
            for pair, roots in zip(self._source_pairs(date, all_turns), pairs):
                before = self.object_snapshot()
                pair_id = str(uuid.uuid5(uuid.NAMESPACE_URL, digest([source['id'], date, roots])))
                user['history'][date].append(pair)
                self.pair_ids[date].append(pair_id)
                self._observe_write(before, {address('history', pair_id): self._write_sources(roots)})
            for layer in ('summary', 'personality'):
                before = self.object_snapshot()
                binding = self._write_sources(parents=[address('history', p) for p in self.pair_ids[date]])
                value, request = self._generate(layer, date, user, all_turns, completions)
                user[layer][date] = {'content': value} if layer == 'summary' else value
                binding['request_id'] = request
                self._observe_write(before, {address(layer, date): binding})
        for layer, parent in [('overall_history', 'summary'), ('overall_personality', 'personality')]:
            before = self.object_snapshot()
            binding = self._write_sources(parents=[address(parent, d) for d in user[parent]])
            user[layer], request = self._generate(layer, self.user_id, user, all_turns, completions)
            binding['request_id'] = request
            self._observe_write(before, {address(layer, self.user_id): binding})
        self._sync()
        self._build_index()
        self.meta['formation_identity'] = self.formation_identity()
        self.save()

    def _record(self, op, oid, roots, *, request=None, parents=(), observation):
        root_set = set(roots)
        roots = [t for t in self.meta['turn_order'] if t in root_set]
        before = self.meta['src'].get(oid, [])
        if op == 'delete':
            self.meta['src'].pop(oid, None)
        else:
            self.meta['src'][oid] = roots
            ever = set(self.meta['ever'].get(oid, [])) | root_set
            self.meta['ever'][oid] = [t for t in self.meta['turn_order'] if t in ever]
            self.meta['birth'].setdefault(oid, len(self.meta['birth']))
        self.meta['history'].append({'op': op, 'obj_id': oid, 'before_turns': before, 'turns': roots,
                                     'parents': list(parents), 'request_id': request,
                                     'step': len(self.meta['history']), 'observation': observation})

    def _refresh_index(self):
        """Native document order, reuse survivor vectors, embed only changed texts."""
        import faiss
        import numpy as np
        from langchain.vectorstores import FAISS
        from langchain.docstore.in_memory import InMemoryDocstore
        previous = {}
        if self.store is not None:
            for row, ident in self.store.index_to_docstore_id.items():
                previous[self.doc_addresses[ident]] = (ident, self.store.docstore.search(ident),
                                                       self.store.index.reconstruct(row).copy())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'memory.json'
            path.write_text(json.dumps(self.memory, ensure_ascii=False))
            docs = self.native.load_memory_file(str(path), self.user_id, 'en')
        addresses = []
        for date, pairs in self.memory[self.user_id]['history'].items():
            addresses.extend(('history', p) for p in self.pair_ids[date])
            if date in self.memory[self.user_id]['summary']:
                addresses.append(('summary', date))
        if len(addresses) != len(docs):
            raise RuntimeError('Native loader object mapping changed')
        changed = [i for i, (a, d) in enumerate(zip(addresses, docs))
                   if a not in previous or previous[a][1].page_content != d.page_content]
        embedded = self.qa.embeddings.embed_documents([docs[i].page_content for i in changed]) if changed else []
        replacements = dict(zip(changed, embedded))
        if not docs:
            self.store, self.doc_addresses = None, {}
            return {'embedded_objects': 0}
        vectors = [replacements[i] if i in replacements else previous[a][2] for i, a in enumerate(addresses)]
        index = faiss.clone_index(self.store.index) if self.store is not None else faiss.IndexFlatL2(len(vectors[0]))
        index.reset()
        index.add(np.asarray(vectors, dtype=np.float32))
        ids = [previous[a][0] if a in previous else str(uuid.uuid5(uuid.NAMESPACE_URL, address(*a))) for a in addresses]
        self.store = FAISS(self.qa.embeddings.embed_query, index, InMemoryDocstore(dict(zip(ids, docs))), dict(enumerate(ids)))
        self.store.chunk_size = self.qa.chunk_size
        self.doc_addresses = dict(zip(ids, addresses))
        return {'embedded_objects': len(changed), 'reused_vectors': len(docs)-len(changed)}

    async def delete_turn(self, turn, completions):
        return await self.delete_turns([turn], completions)

    async def delete(self, obj_id):
        """Whole-object primitive; turn policies use delete_turns for the closure."""
        return await self._edit_object(obj_id, None, None)

    async def regenerate(self, obj_id, src_turn_ids, completions):
        return await self._edit_object(obj_id, set(src_turn_ids), completions)

    async def _edit_object(self, obj_id, remaining, completions):
        if self.checkpoint.with_suffix('.frozen.json').exists():
            raise ValueError('Fork the frozen snapshot before editing')
        oid = obj_id.address if hasattr(obj_id, 'address') else obj_id
        obj = next((o for o in await self.list_all() if o.address == oid), None)
        if obj is None:
            raise ValueError('Object is not live')
        if hasattr(obj_id, 'version') and obj_id.version != obj.version:
            raise ValueError('Stale native object version')
        if remaining is not None and (obj.layer == 'history' or not remaining <= self.trace(oid)):
            raise ValueError('Regenerate requires a subset of a derived object source set')
        if remaining and obj.layer in {'overall_history', 'overall_personality'}:
            parent = 'summary' if obj.layer == 'overall_history' else 'personality'
            roots = {t for d in self.memory[self.user_id][parent] for t in self.trace(address(parent, d))}
            if roots != remaining:
                raise ValueError('Regenerate daily parents to the requested remaining sources first')
        old = self.memory, self.pair_ids, self.meta, self.store, self.doc_addresses
        self.memory, self.pair_ids, self.meta = copy.deepcopy(self.memory), copy.deepcopy(self.pair_ids), copy.deepcopy(self.meta)
        try:
            user = self.memory[self.user_id]
            before = self.object_snapshot()
            if remaining:
                parents = []
                if obj.layer in {'overall_history', 'overall_personality'}:
                    parent = 'summary' if obj.layer == 'overall_history' else 'personality'
                    parents = [address(parent, d) for d in user[parent]]
                binding = self._write_sources(remaining, parents=parents)
                text, request = self._generate(obj.layer, obj.native_id, user, remaining, completions)
                if obj.layer in {'summary', 'personality'}:
                    user[obj.layer][obj.native_id] = {'content': text} if obj.layer == 'summary' else text
                else:
                    user[obj.layer] = text
                binding['request_id'] = request
                observation = self._observe_write(before, {oid: binding}, rewrite=True)
            else:
                if obj.layer == 'history':
                    date = next(d for d, ids in self.pair_ids.items() if obj.native_id in ids)
                    index = self.pair_ids[date].index(obj.native_id)
                    del self.pair_ids[date][index]
                    del user['history'][date][index]
                elif obj.layer in {'summary', 'personality'}:
                    del user[obj.layer][obj.native_id]
                else:
                    user[obj.layer] = ''
                observation = self._observe_write(before, {})
            self._sync()
            maintenance = self._refresh_index()
            self.save()
        except BaseException:
            self.memory, self.pair_ids, self.meta, self.store, self.doc_addresses = old
            raise
        return {'obj_id': oid, 'operation': 'rewrite' if remaining else 'delete',
                'maintenance': maintenance, 'mutation_diff': observation}

    async def delete_turns(self, selected, completions):
        """One transaction for a prefix delta; all affected fields rewritten once."""
        if self.checkpoint.with_suffix('.frozen.json').exists():
            raise ValueError('Fork the frozen snapshot before deleting turns')
        selected = set(selected)
        if not selected <= set(self.meta['turns']):
            raise ValueError('Unknown selected turn')
        added = selected - set(self.meta['deleted_turns'])
        before_state = self.state_identity()
        if not added:
            return {'selected_turns': [], 'deleted_objs': [], 'rewritten_objs': [],
                    'before_state': before_state, 'after_state': before_state}
        old = self.memory, self.pair_ids, self.meta, self.store, self.doc_addresses
        self.memory, self.pair_ids, self.meta = copy.deepcopy(self.memory), copy.deepcopy(self.pair_ids), copy.deepcopy(self.meta)
        try:
            all_deleted = set(self.meta['deleted_turns']) | added
            user, removed, rewritten, collateral = self.memory[self.user_id], [], [], set()
            diff_start = len(self.meta['mutation_diffs'])
            affected = {oid for t in added for oid in self.meta['der'].get(t, [])}
            for date in self.meta['groups']:
                before = self.object_snapshot()
                original_pair_count = len(user['history'][date])
                pairs, ids = [], []
                for pair, ident in zip(user['history'][date], self.pair_ids[date]):
                    oid = address('history', ident)
                    if oid in affected:
                        collateral.update(self.trace(oid) - all_deleted)
                    else:
                        pairs.append(pair)
                        ids.append(ident)
                user['history'][date], self.pair_ids[date] = pairs, ids
                if len(pairs) != original_pair_count:
                    observation = self._observe_write(before, {})
                    removed.extend(observation['deleted'])
                for layer in ('summary', 'personality'):
                    oid = address(layer, date)
                    if oid not in affected:
                        continue
                    before = self.object_snapshot()
                    remaining = self.trace(oid) - all_deleted
                    if not remaining:
                        user[layer].pop(date, None)
                        observation = self._observe_write(before, {})
                        removed.extend(observation['deleted'])
                    else:
                        binding = self._write_sources(remaining)
                        value, request = self._generate(layer, date, user, remaining, completions)
                        user[layer][date] = {'content': value} if layer == 'summary' else value
                        binding['request_id'] = request
                        observation = self._observe_write(before, {oid: binding}, rewrite=True)
                        rewritten.extend(observation['modified'] + observation['unchanged_writes'])
            for layer, parent in [('overall_history', 'summary'), ('overall_personality', 'personality')]:
                oid = address(layer, self.user_id)
                if oid not in affected:
                    continue
                before = self.object_snapshot()
                remaining = self.trace(oid) - all_deleted
                if not remaining:
                    user[layer] = ''
                    observation = self._observe_write(before, {})
                    removed.extend(observation['deleted'])
                else:
                    binding = self._write_sources(parents=[address(parent, d) for d in user[parent]])
                    if set(binding['turns']) != remaining:
                        raise ValueError('Updated parent ancestry differs from remaining global sources')
                    value, request = self._generate(layer, self.user_id, user, remaining, completions)
                    user[layer] = value
                    binding['request_id'] = request
                    observation = self._observe_write(before, {oid: binding}, rewrite=True)
                    rewritten.extend(observation['modified'] + observation['unchanged_writes'])
            self.meta['deleted_turns'] = [t for t in self.meta['turn_order'] if t in all_deleted]
            self._sync()
            maintenance = self._refresh_index()
            self.save()
        except BaseException:
            self.memory, self.pair_ids, self.meta, self.store, self.doc_addresses = old
            raise
        return {'selected_turns': [t for t in self.meta['turn_order'] if t in added],
                'deleted_objs': removed, 'rewritten_objs': rewritten,
                'cumulative_deleted_objs': list(dict.fromkeys(e['obj_id'] for e in self.meta['history'] if e['op'] == 'delete')),
                'cumulative_rewritten_objs': list(dict.fromkeys(e['obj_id'] for e in self.meta['history'] if e['op'] == 'rewrite')),
                'collateral_raw_turns': sorted(collateral), 'maintenance': maintenance,
                'mutation_diffs': self.meta['mutation_diffs'][diff_start:],
                'before_state': before_state, 'after_state': self.state_identity()}

    async def residual_objects(self, roots):
        ids = {oid for t in roots for key in ('der', 'touched') for oid in self.meta[key].get(t, [])}
        return sorted([o for o in await self.list_all() if o.address in ids], key=lambda o: self.meta['birth'][o.address])

    async def native_context(self, question):
        before = self.state_identity()
        result = await self.answer_context(question)
        if before != self.state_identity():
            raise RuntimeError('Native reading mutated storage')
        return result

    @staticmethod
    def system_render(search_result):
        return search_result.native_context['prompt']


def open_memorybank(upstream, user_id, checkpoint, model):
    base, summaries = make_memorybank(upstream, user_id, checkpoint)
    bank = SourceMemoryBank(base.native, base.prompts, base.qa, user_id, checkpoint, summaries, model)
    if Path(checkpoint).exists():
        bank.restore()
    return bank
