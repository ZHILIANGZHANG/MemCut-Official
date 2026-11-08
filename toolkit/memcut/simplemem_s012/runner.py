"""SimpleMem entrypoint reusing current candidates, branches and 2/2 scoring."""
from contextlib import contextmanager
import copy
from datetime import datetime
from pathlib import Path
import pickle
import os
import shutil
import time
from unittest.mock import patch

from memcut.native_memory.contract import digest
from memcut.source_anchored_systems import runner as shared
from memcut.source_anchored_v2 import runner as core
from memcut.source_anchored_v2.candidates import candidate_protocol
from memcut.source_anchored_systems.formation_runtime.concurrency import ConcurrentCompletions, ordered_map
from memcut.raw_turn_rag import runner as inputs
from memcut.execution_optimization import receipts, scheduling
from memcut.model_adapters.glm import MODEL, ENDPOINT, PROFILE, Gateway
from . import VERSION, SOURCE_POLICY, RETRIEVAL_POLICY
from .upstream import ROOT, CONFIG, profile, runtime_identity, doctor
from .formation import windows, extract

read, sha, sealed_write = core.read, core.sha, core.sealed_write
import_source = shared.import_source
PROTOCOL = {**copy.deepcopy(shared.PROTOCOL), 'version': VERSION + '-whole-object-eval2of2',
    'candidate_search': 'native planned three-path ordered union on S0; final top20 objects, source cap20',
    'provenance': SOURCE_POLICY, 'retrieval_policy': RETRIEVAL_POLICY,
    'memory_generation_finish': 'one native bulk-window ingestion; final tail and full FTS index; no global rewrite',
    'attribution_extra_llm_calls': 0}


def code_identity():
    paths = list(Path(__file__).parent.glob('*.py')) + [CONFIG, ROOT/'tools/simplemem_s012.py',
        ROOT/'tools/setup_simplemem.py', ROOT/'configs/simplemem_runtime_requirements.txt']
    paths += [ROOT / p for p in shared.code_identity()]
    paths += [ROOT/'memcut/raw_turn_rag/runner.py', ROOT/'memcut/raw_turn_rag/embedding.py',
              ROOT/'memcut/model_adapters/glm.py']
    paths += list((ROOT/'memcut/execution_optimization').glob('*.py'))
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(paths))}


def formation_identity(plan):
    names = {'__init__.py', 'formation.py', 'client.py', 'native.py', 'bank.py', 'upstream.py'}
    paths = {p: h for p, h in plan['code'].items() if
        (Path(p).parent == Path('memcut/simplemem_s012') and Path(p).name in names) or p in {
            'memcut/source_anchored_systems/base.py', 'memcut/source_anchored_systems/lineage.py',
            'memcut/source_anchored_v2/provenance.py', 'memcut/source_anchored_v2/completions.py',
            'memcut/raw_turn_rag/embedding.py', 'memcut/native_memory/contract.py', 'memcut/model_adapters/glm.py'}}
    return digest({'settings': plan['settings'], 'code': paths, 'profile': plan['profile'],
        'runtime': plan['runtime'], 'write_request_profile': plan['request_profiles']['write'],
        'source': plan['inputs']['source']['sha256']})


def prepare(folder, *, source, contracts, annotations, write_model=MODEL, answer_model=None,
            selector_model=None, retrieval_model=None, endpoint=ENDPOINT, compatibility='openai',
            answer_endpoint=None, answer_compatibility=None, selector_endpoint=None,
            selector_compatibility=None, retrieval_endpoint=None, retrieval_compatibility=None,
            claim_ids=None, reuse_full=None, reuse_requests_from=(), reuse_deletions_from=None,
            build_workers=4, answer_workers=4, candidate_objects=20, candidate_turns=20,
            k_list=(10,), controls=False, residual_plus=False):
    if controls or residual_plus:
        raise ValueError('This entry freezes the main native/residual experiment; optional appendix controls require a separate plan')
    for workers in (build_workers, answer_workers):
        if type(workers) is not int or not 1 <= workers <= 16:
            raise ValueError('Use 1–16 workers')
    native = profile()
    runtime = doctor()['packages']
    if reuse_deletions_from and retrieval_model is None:
        # A changed answer model must not silently change the saved planner.
        retrieval_model = read(Path(reuse_deletions_from).resolve()/'PLAN.json')['models']['retrieval']
    answer_model = answer_model or write_model
    models = {'write': write_model, 'answer': answer_model, 'selector': selector_model or write_model,
              'retrieval': retrieval_model or answer_model}
    providers = {'write': {'endpoint': endpoint, 'compatibility': compatibility, 'key_env': 'MEMCUT_MODEL_API_KEY'}}
    for role, url, mode, env in (
            ('answer', answer_endpoint, answer_compatibility, 'MEMCUT_RECOVERY_API_KEY'),
            ('selector', selector_endpoint, selector_compatibility, 'MEMCUT_SELECTOR_API_KEY'),
            ('retrieval', retrieval_endpoint or answer_endpoint, retrieval_compatibility or answer_compatibility,
             'MEMCUT_RETRIEVAL_API_KEY')):
        providers[role] = {'endpoint': url or endpoint, 'compatibility': mode or compatibility,
                          'key_env': env if url and url != endpoint else 'MEMCUT_MODEL_API_KEY'}
    if reuse_deletions_from and retrieval_endpoint is None and retrieval_compatibility is None:
        providers['retrieval'] = copy.deepcopy(read(Path(reuse_deletions_from).resolve()/'PLAN.json')['providers']['retrieval'])
    for role, provider in providers.items():
        if provider['compatibility'] not in ('openai','deepseek','mimo'):
            raise ValueError('Unknown provider compatibility')
        if models[role] == MODEL and (provider['endpoint'].rstrip('/') != ENDPOINT or provider['compatibility'] != 'openai'):
            raise ValueError(f'GLM requires its official endpoint and compatibility=openai for {role}')
    sources = list(reuse_requests_from or [])
    if reuse_deletions_from:
        reuse_full = reuse_full or str(Path(reuse_deletions_from).resolve()/'full/full.pkl')
        sources.append(reuse_deletions_from)
    if reuse_full:
        sources.append(str(Path(reuse_full).resolve().parent.parent))
    catalog = receipts.catalog(sources)
    turns, _, order = import_source(read(source))
    settings = copy.deepcopy(native['settings'])
    settings.update(model=write_model, write_transport={
        'endpoint': endpoint, 'compatibility': compatibility, 'key_env': 'MEMCUT_MODEL_API_KEY'},
        retrieval_reference_time=max((t['timestamp'] for t in turns.values()), key=datetime.fromisoformat))
    # Reuse the existing source/contract/annotation and cohort validation. This
    # only prepares a new local plan; no RAG bank or embedding is constructed.
    normalizations = []
    def normalize_claim(row):
        normalized = copy.deepcopy(row)
        for rotation in normalized.get('rotations') or []:
            trial = rotation.get('trial')
            if isinstance(trial, str) and trial in ('trial1', 'trial2', 'trial3'):
                rotation['trial'] = int(trial[-1])
                normalizations.append({'claim_id': row['claim_id'], 'before': trial, 'after': rotation['trial']})
        return core.normalize_claim(normalized)
    with patch.object(inputs, 'normalize_claim', normalize_claim):
        plan = inputs.prepare(folder, source=source, contracts=contracts, annotations=annotations,
            answer_model=answer_model, selector_model=models['selector'], endpoint=endpoint,
            compatibility=compatibility, claim_ids=claim_ids, reuse_full=reuse_full,
            candidate_objects=candidate_objects, candidate_turns=candidate_turns,
            k_list=k_list, controls=controls, residual_plus=residual_plus)
    plan.update(system='simplemem', profile=native, runtime=runtime, settings=settings,
        protocol=candidate_protocol(PROTOCOL, candidate_objects, candidate_turns), code=code_identity(), models=models,
        providers=providers,
        request_profiles={role: copy.deepcopy(PROFILE) if model == MODEL else {'compatibility': providers[role]['compatibility'],
            'temperature': 0, 'automatic_retries': 0} for role, model in models.items()},
        answer_policy={'version': 'simplemem-uniform-2048-v1', 'max_tokens': 2048,
                       'readouts': ['native', 'residual'], 'all_conditions': plan['conditions']},
        execution={'build_workers': build_workers, 'answer_workers': answer_workers,
                   'commit_order': 'source-window-order', 'checkpoint': 'every-committed-window', 'paid_retries': 0})
    plan['input_normalization'] = {'policy': 'exact-trial-name-to-integer-v1', 'changes': normalizations,
        'source_files_modified': False, 'choice_mapping_changes': False}
    if reuse_full:
        old = read(Path(reuse_full).resolve().parent.parent/'PLAN.json')
        if old.get('system') != 'simplemem' or formation_identity(old) != formation_identity(plan):
            raise ValueError('Reusable SimpleMem Full has a different formation identity')
    sealed_write(Path(folder)/'RECEIPT_CATALOG.json', catalog)
    plan['receipt_catalog_sha256'] = sha(Path(folder)/'RECEIPT_CATALOG.json')
    if reuse_deletions_from:
        from .reuse import prepare_catalog
        sealed_write(Path(folder)/'BRANCH_CATALOG.json', prepare_catalog(reuse_deletions_from, plan))
        plan['branch_catalog_sha256'] = sha(Path(folder)/'BRANCH_CATALOG.json')
    core.write_json(Path(folder)/'PLAN.json', plan)
    return plan


def load_plan(folder):
    folder = Path(folder)
    plan = read(folder/'PLAN.json')
    if (plan['system'] != 'simplemem' or plan['code'] != code_identity() or plan['profile'] != profile()
            or plan['runtime'] != runtime_identity()
            or plan['protocol'] != candidate_protocol(PROTOCOL, plan['protocol']['n_obj'], plan['protocol']['n_turn'])):
        raise ValueError('Frozen SimpleMem implementation/profile/runtime changed; use a new run')
    if sha(folder/'PROTOCOL.md') != plan['protocol_document_sha256']:
        raise ValueError('Frozen protocol changed')
    for entry in plan['inputs'].values():
        if sha(entry['path']) != entry['sha256']:
            raise ValueError('Frozen input changed')
    if plan['receipt_catalog_sha256'] != sha(folder/'RECEIPT_CATALOG.json'):
        raise ValueError('Frozen receipt sources changed')
    if plan.get('branch_catalog_sha256') and plan['branch_catalog_sha256'] != sha(folder/'BRANCH_CATALOG.json'):
        raise ValueError('Frozen deletion sources changed')
    expected_settings = {**copy.deepcopy(plan['profile']['settings']), 'model': plan['models']['write'],
        'write_transport': plan['providers']['write'],
        'retrieval_reference_time': max((t['timestamp'] for t in import_source(read(plan['inputs']['source']['path']))[0].values()), key=datetime.fromisoformat)}
    if plan['settings'] != expected_settings:
        raise ValueError('SimpleMem settings differ from the accepted profile')
    if plan['answer_policy'] != {'version': 'simplemem-uniform-2048-v1', 'max_tokens': 2048,
                                'readouts': ['native', 'residual'], 'all_conditions': plan['conditions']}:
        raise ValueError('Answer budget/policy changed')
    execution = plan['execution']
    if (any(type(execution[k]) is not int or not 1 <= execution[k] <= 16 for k in ('build_workers','answer_workers'))
            or execution['paid_retries'] != 0 or execution['commit_order'] != 'source-window-order'
            or execution['checkpoint'] != 'every-committed-window'):
        raise ValueError('Unsupported frozen execution policy')
    for role, model in plan['models'].items():
        provider = plan['providers'][role]
        if model == MODEL and (provider['endpoint'].rstrip('/') != ENDPOINT or provider['compatibility'] != 'openai'):
            raise ValueError('Frozen GLM provider differs')
        expected = PROFILE if model == MODEL else {'compatibility': plan['providers'][role]['compatibility'],
                                                    'temperature': 0, 'automatic_retries': 0}
        if plan['request_profiles'][role] != expected:
            raise ValueError('Frozen request profile changed')
    return plan


def gateway(folder, plan, *, cached_only=False, create_factory=None):
    delegate = Gateway(folder, plan, create_factory=create_factory)
    return receipts.ReuseGateway(delegate, read(Path(folder)/'RECEIPT_CATALOG.json'), cached_only=cached_only)


class Factory:
    def __init__(self, plan, completions=None, *, embedder=None):
        from .native import LocalEmbedding
        self.plan = plan
        self.turns, _, _ = import_source(read(plan['inputs']['source']['path']))
        self.embedder = embedder or LocalEmbedding(plan['settings']['embedding'], ROOT/'runtime/simplemem/embeddings')
        self.cache = core.role_cache(completions, 'retrieval') if completions is not None else None

    def create(self, snapshot=None):
        from .bank import SimpleMemBank
        bank = SimpleMemBank(self.plan['settings'], self.turns, embedder=self.embedder,
            retrieval_cache=self.cache, retrieval_model=self.plan['models']['retrieval'])
        try:
            if snapshot:
                bank.restore(snapshot)
            else:
                bank.progress['formation_identity'] = formation_identity(self.plan)
            return bank
        except BaseException:
            bank.close()
            raise

    def close(self):
        pass


def validate_full(bank, plan):
    turns, _, _ = import_source(read(plan['inputs']['source']['path']))
    expected = windows(list(turns), plan['settings']['window_size'], plan['settings']['overlap_size'])
    if (not bank.frozen or bank.progress['stage'] != 'FULL' or bank.turns != turns
            or bank.progress['next_turn'] != len(turns) or bank.meta['deleted_turns']
            or bank.progress['formation_identity'] != formation_identity(plan)
            or bank.progress['next_window'] != len(expected)
            or [b['input_turn_ids'] for b in bank.progress['formation_batches']] != expected
            or not bank.store.backend._fts_initialized):
        raise ValueError('Require complete undeleted SimpleMem Full of the exact source')
    bank.lineage.validate(observed=bank.observation())


def build(folder, plan, completions, *, factory=None, stop_after_windows=None):
    if stop_after_windows is not None and (type(stop_after_windows) is not int or stop_after_windows < 1):
        raise ValueError('Stop limit must be a positive number of committed native windows')
    folder = Path(folder)
    factory = factory or Factory(plan)
    bank = factory.create()
    full, checkpoint = folder/'full/full.pkl', folder/'formation/checkpoint.pkl'
    started = time.perf_counter()
    try:
        if full.exists():
            bank.restore(full)
        elif plan.get('reuse_full'):
            reuse = plan['reuse_full']
            if sha(reuse['path']) != reuse['sha256']:
                raise ValueError('Reusable Full changed')
            bank.restore(reuse['path'])
        else:
            if checkpoint.exists():
                shared.restore_checkpoint(checkpoint, bank, plan)
            else:
                shared.checkpoint(checkpoint, bank, plan)
            batches = windows(list(bank.turns), plan['settings']['window_size'], plan['settings']['overlap_size'])
            end = len(batches) if stop_after_windows is None else min(stop_after_windows, len(batches))
            if type(end) is not int or end < bank.progress['next_window']:
                raise ValueError('Stop limit precedes the committed checkpoint')
            cache = ConcurrentCompletions(core.role_cache(completions, 'write'), plan['execution']['build_workers'])
            def generate(job):
                i, ids = job
                try:
                    return extract(bank.turns, ids, cache, plan['settings'], window_index=i)
                except BaseException:
                    cache.failed.set()
                    raise
            for batch in ordered_map(generate, enumerate(batches[bank.progress['next_window']:end],
                                        bank.progress['next_window']), plan['execution']['build_workers']):
                bank.commit_batch(batch)
                shared.checkpoint(checkpoint, bank, plan)
                core.write_json(folder/'PROGRESS.json', {'stage': 'formation',
                    'completed_windows': bank.progress['next_window'], 'planned_windows': len(batches),
                    'completed_turns': bank.progress['next_turn'], 'planned_turns': len(bank.turns),
                    'objects': len(bank.meta['src'])})
            if end != len(batches):
                return {'complete': False, 'checkpoint': str(checkpoint), 'completed_windows': end}
            bank.finish()
            shared.checkpoint(checkpoint, bank, plan)
        validate_full(bank, plan)
        if not full.exists():
            if plan.get('reuse_full'):
                # Keep the exact immutable Full bytes as well as its semantic
                # identity; this also permits repeated audited branch imports.
                original = Path(plan['reuse_full']['path'])
                full.parent.mkdir(parents=True, exist_ok=True)
                pending = full.with_suffix('.pending')
                shutil.copyfile(original, pending)
                if sha(pending) != plan['reuse_full']['sha256']:
                    raise ValueError('Reusable Full changed during copying')
                os.replace(pending, full)
                shutil.copyfile(original.with_suffix('.frozen.json'), full.with_suffix('.frozen.json'))
            else:
                bank.snapshot(full)
        sealed_write(folder/'full/IDENTITY.json', {'plan_sha256': digest(plan), 'state_identity': bank.state_identity()})
        core.write_json(folder/'PROGRESS.json', {'stage': 'full_complete', 'completed_turns': len(bank.turns),
            'objects': len(bank.meta['src']), 'full': str(full)})
        return str(full)
    finally:
        core.write_json(folder/'formation/LAST_STAGE.json', {'elapsed_seconds': time.perf_counter()-started,
            'completed_windows': bank.progress['next_window'], 'full': full.exists()})
        bank.close()


async def delete(folder, plan, completions, *, factory=None, claim_ids=None):
    if plan.get('branch_catalog_sha256'):
        from .reuse import import_claims
        return import_claims(folder, plan, read(Path(folder)/'BRANCH_CATALOG.json'), claim_ids=claim_ids)
    factory = factory or Factory(plan, completions)
    async def erase(bank, selected):
        return await bank.delete_turns(selected)
    # Only this process' invocation changes dispatch; existing adapters/files
    # and their frozen formation identities are untouched.
    with patch.object(shared, 'validate_full', validate_full), patch.object(shared, 'delete_source_turns', erase):
        await shared.delete(folder, plan, completions, factory=factory, claim_ids=claim_ids)


@contextmanager
def answer_runtime(plan):
    from memcut.source_anchored_v2 import evaluation
    original = evaluation.answer_request
    def request(*args, **kwargs):
        return {**original(*args, **kwargs), 'max_tokens': plan['answer_policy']['max_tokens']}
    with patch.object(evaluation, 'answer_request', request):
        yield


def answer(folder, plan, completions, *, claim_ids=None):
    with answer_runtime(plan):
        return scheduling.answer(folder, plan, completions, workers=plan['execution']['answer_workers'],
                                 claim_ids=claim_ids, source_import=import_source)


def report(folder, plan):
    return shared.report(folder, plan)
