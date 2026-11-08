"""Independent whole-session baseline; local formation and frozen S012 evaluation."""
import copy
import hashlib
import os
from pathlib import Path
import shutil

from memcut.native_memory.contract import digest
from memcut.native_memory.persistence import write_json
from memcut.source_anchored_systems import runner as shared
from memcut.source_anchored_v2.runner import (read, sha, sealed_write, normalize_claim,
    stage_claims, validate_state, role_cache, gateway)
from memcut.source_anchored_v2.candidates import build_pool, rank_leakcut, verify_pool, candidate_protocol
from . import VERSION
from .bank import SessionBank, DELETION_POLICY
from memcut.raw_turn_rag.embedding import LocalEmbeddings

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'configs/session_rag_v1.json'
PROTOCOL_DOC = ROOT / 'docs/integrations/SESSION_RAG_S012.md'
PROTOCOL = {**copy.deepcopy(shared.PROTOCOL), 'version': VERSION + '-precise-turn-delete-eval2of2',
    'candidate_search': 'top20 live sessions by max chunk cosine on S0, then raw-turn cosine cap20',
    'deletion_policy': DELETION_POLICY,
    'closure': 'remove selected complete turn text from live sessions; locally rechunk and encode affected sessions',
    'residual': 'only live E+D turn text extracted from current stored session members',
    'provenance': 'each live session maps exactly to its live original single-role turns',
    'memory_generation_finish': 'no generation; store original text and local embedding',
    'formation_llm_calls': 0}


def code_identity():
    paths = list(Path(__file__).parent.glob('*.py')) + [CONFIG, ROOT/'tools/session_rag.py']
    paths += [ROOT/p for p in shared.code_identity()]
    paths += [ROOT/p for p in ('memcut/raw_turn_rag/embedding.py', 'memcut/raw_turn_rag/bank.py',
        'memcut/answer_execution/early_stop.py', 'memcut/efficient_execution/recovery.py',
        'memcut/model_adapters/glm.py', 'memcut/model_adapters/qwen.py',
        'memcut/execution_optimization/receipts.py')]
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(paths))}


def runtime_identity():
    from importlib.metadata import version
    return {n: version(n) for n in ('numpy','torch','sentence-transformers','transformers','tokenizers','huggingface-hub','tiktoken','openai','httpx')}


def formation_identity(plan):
    paths = ['memcut/session_rag/bank.py', 'memcut/session_rag/text.py',
             'memcut/session_rag/__init__.py', 'memcut/raw_turn_rag/embedding.py',
             'memcut/source_anchored_systems/base.py','memcut/source_anchored_systems/lineage.py',
             'memcut/native_memory/contract.py','memcut/source_anchored_v2/provenance.py']
    return digest({'format': VERSION, 'settings': plan['settings'],
                   'source': plan['inputs']['source']['sha256'], 'runtime': plan['runtime'],
                   'code': {p: plan['code'][p] for p in paths}})


def load_plan(folder):
    folder = Path(folder)
    plan = read(folder/'PLAN.json')
    if read(folder/'PLAN_IDENTITY.json') != {'plan_sha256': digest(plan)}:
        raise ValueError('Frozen plan changed')
    if plan['runtime'] != runtime_identity():
        raise ValueError('Frozen runtime package versions changed')
    expected = candidate_protocol(PROTOCOL, plan['protocol']['n_obj'], plan['protocol']['n_turn'])
    if plan['system'] != 'session_rag' or plan['protocol'] != expected or plan['code'] != code_identity():
        raise ValueError('Frozen implementation changed; use a new RAG run version')
    expected_settings = {**read(CONFIG), 'native_eval_k': plan['settings']['native_eval_k']}
    if plan['settings'] != expected_settings:
        raise ValueError('Frozen RAG settings differ')
    if sha(folder/'PROTOCOL.md') != plan['protocol_document_sha256']:
        raise ValueError('Frozen protocol changed')
    for entry in plan['inputs'].values():
        if sha(entry['path']) != entry['sha256']:
            raise ValueError('Frozen source or annotations changed')
    return plan


def prepare(folder, *, source, contracts, annotations, answer_model, endpoint,
            selector_model=None, compatibility='openai', claim_ids=None, k_list=(10,),
            candidate_objects=20, candidate_turns=20, native_eval_k=16,
            controls=False, residual_plus=False, reuse_full=None,
            selector_endpoint=None, selector_compatibility=None, qa_max_tokens=1024, early_stop=True):
    if controls or residual_plus:
        raise ValueError('Session RAG v1 covers native/residual main readouts only')
    if type(qa_max_tokens) is not int or qa_max_tokens <= 0 or type(early_stop) is not bool:
        raise ValueError('Invalid answer execution settings')
    if type(native_eval_k) is not int or native_eval_k <= 0:
        raise ValueError('native_eval_k must be positive')
    protocol = candidate_protocol(PROTOCOL, candidate_objects, candidate_turns)
    inputs = {k: {'path': str(Path(p).resolve()), 'sha256': sha(p)} for k, p in
              [('source', source), ('contracts', contracts), ('annotations', annotations)]}
    turns, _, order = shared.import_source(read(source))
    contract, annotation = read(contracts), read(annotations)
    if any(x.get('source_sha256') != inputs['source']['sha256'] for x in (contract, annotation)):
        raise ValueError('Contracts/annotations do not match source')
    all_ids = [c['claim_id'] for c in contract['claims']]
    selected = set(all_ids if claim_ids is None else claim_ids)
    if len(all_ids) != len(set(all_ids)) or not selected or not selected <= set(all_ids):
        raise ValueError('Unknown, duplicate or empty claim cohort')
    ks = sorted(set(k_list))
    if not ks or any(type(k) is not int or k not in PROTOCOL['k_list'] for k in ks):
        raise ValueError('Use documented deletion budgets')
    claims = []
    for row in contract['claims']:
        if row['claim_id'] not in selected:
            continue
        claim = {**normalize_claim(row), 'deletion_request': row['deletion_request']}
        for field, key in [('evidence_turns', 'carriers'), ('disclosure_turns', 'direct'), ('distractor_turns', 'distractors')]:
            ids = set()
            for a in annotation[key]:
                if a['claim_id'] != claim['claim_id']:
                    continue
                tid = a['record_id']
                if tid not in turns or a['role'] != turns[tid]['role']:
                    raise ValueError('Annotation turn or role differs')
                text = turns[tid]['text']
                if a.get('text_sha256') != hashlib.sha256(text.encode()).hexdigest() or ('text' in a and a['text'] != text):
                    raise ValueError('Annotation text differs')
                ids.add(tid)
            claim[field] = [t for t in order if t in ids]
        if not 0 < len(claim['evidence_turns']) <= 10 or not claim['disclosure_turns']:
            raise ValueError('Require E<=10 and nonempty E/D')
        if 'direct_record_ids' in row and set(row['direct_record_ids']) != set(claim['disclosure_turns']):
            raise ValueError('Direct fields disagree')
        claims.append(claim)
    if compatibility not in {'openai', 'deepseek', 'mimo'} or (selector_compatibility or compatibility) not in {'openai', 'deepseek', 'mimo'}:
        raise ValueError('Unknown provider compatibility')
    reuse = None
    if reuse_full:
        p = Path(reuse_full).resolve()
        if sha(p) != read(p.with_suffix('.frozen.json'))['sha256']:
            raise ValueError('Reusable Full changed')
        reuse = {'path': str(p), 'sha256': sha(p)}
    settings = {**read(CONFIG), 'native_eval_k': native_eval_k}
    plan = {'system': 'session_rag', 'protocol': protocol, 'settings': settings,
            'code': code_identity(), 'inputs': inputs, 'claims': claims, 'k_list': ks,
            'models': {'answer': answer_model, 'selector': selector_model or answer_model},
            'providers': {
                'answer': {'endpoint': endpoint, 'compatibility': compatibility, 'key_env': 'MEMCUT_MODEL_API_KEY'},
                'selector': {'endpoint': selector_endpoint or endpoint,
                             'compatibility': selector_compatibility or compatibility,
                             'key_env': 'MEMCUT_SELECTOR_API_KEY' if selector_endpoint and selector_endpoint != endpoint else 'MEMCUT_MODEL_API_KEY'}},
            'endpoint': endpoint, 'compatibility': compatibility, 'reuse_full': reuse,
            'controls': controls, 'residual_plus': residual_plus, 'retention': [],
            'protocol_document_sha256': sha(PROTOCOL_DOC),
            'conditions': ['Full', 'S0'] + [f'{s}@{k}' for s in ('S1', 'S2') for k in ks]}
    from .transport import provider_profile
    plan['provider_profiles'] = {role: provider_profile(model, plan['providers'][role]['endpoint'], plan['providers'][role]['compatibility']) for role, model in plan['models'].items()}
    plan['runtime'] = runtime_identity()
    plan['qa_max_tokens'] = qa_max_tokens
    plan['early_stop'] = early_stop
    plan['answer_execution'] = 'valid-incorrect-trial1-stop' if early_stop else 'always-two-fixed-trials'
    folder = Path(folder)
    if folder.exists():
        raise FileExistsError('Use a new versioned run directory')
    folder.mkdir(parents=True)
    shutil.copyfile(PROTOCOL_DOC, folder/'PROTOCOL.md')
    sealed_write(folder/'PLAN.json', plan)
    sealed_write(folder/'PLAN_IDENTITY.json', {'plan_sha256': digest(plan)})
    return plan


class Factory:
    def __init__(self, plan, *, embedder=None, cache_dir=None, tokenizer=None):
        self.plan = plan
        self.tokenizer = tokenizer
        self.embedder = embedder or LocalEmbeddings(plan['settings']['embedding'],
            cache_dir or ROOT/'runtime/raw-turn-rag/embeddings')
        self.turns, _, _ = shared.import_source(read(plan['inputs']['source']['path']))

    def create(self, snapshot=None):
        bank = SessionBank(self.plan['settings'], self.turns, self.embedder, tokenizer=self.tokenizer)
        bank.progress['formation_identity'] = formation_identity(self.plan)
        if snapshot:
            bank.restore(snapshot)
        return bank


def validate_full(bank, plan):
    expected, _, _ = shared.import_source(read(plan['inputs']['source']['path']))
    live = [t for row in bank.rows.values() for t in row['turn_ids']]
    if (bank.turns != expected or not bank.frozen or bank.progress['stage'] != 'FULL'
            or bank.meta['deleted_turns'] or bank.progress['formation_identity'] != formation_identity(plan)
            or len(live) != len(set(live)) or set(live) != set(expected)
            or any(bank.trace(oid) != set(row['turn_ids']) for oid,row in bank.rows.items())):
        raise ValueError('Require complete undeleted exact-source Session Full')


def snapshot(bank, path, *, reference=None):
    """Finish an interrupted local write only when its bytes match reconstruction.

    No unsealed pickle is loaded. Differing or partial bytes remain untouched as
    failure evidence; recovery cannot bless an arbitrary stored state.
    """
    import pickle
    from memcut.source_anchored_systems import VERSION as snapshot_format
    path = Path(path)
    seal = path.with_suffix('.frozen.json')
    if seal.exists() and not path.exists():
        raise ValueError('Sealed snapshot is missing; preserve evidence instead of replacing: '+str(path))
    if path.exists() and seal.exists():
        bank.restore(path)
        return str(path)
    expected = Path(reference).read_bytes() if reference else pickle.dumps(bank.export(), protocol=5)
    pending = path.with_suffix('.pending')
    if path.exists() or pending.exists():
        for candidate in (path, pending):
            if candidate.exists() and candidate.read_bytes() != expected:
                raise ValueError('Unsealed snapshot differs from deterministic reconstruction: '+str(candidate))
        if not path.exists():
            os.replace(pending, path)
        sealed_write(seal, {'sha256':hashlib.sha256(expected).hexdigest(),
            'state_identity':bank.state_identity(),'system':bank.system,'format':snapshot_format})
        bank.restore(path)
        return str(path)
    if reference:
        path.parent.mkdir(parents=True, exist_ok=True)
        os.link(reference, path)
        shutil.copyfile(Path(reference).with_suffix('.frozen.json'), seal)
        return str(path)
    return bank.snapshot(path)


def build(folder, plan, *, factory=None):
    """No completion transport is accepted or constructed during formation."""
    folder = Path(folder)
    factory = factory or Factory(plan)
    bank = factory.create()
    full = folder/'full/full.pkl'
    if full.exists() and full.with_suffix('.frozen.json').exists():
        bank.restore(full)
    elif plan['reuse_full']:
        reuse = plan['reuse_full']
        if sha(reuse['path']) != reuse['sha256']:
            raise ValueError('Reusable Full changed')
        bank.restore(reuse['path'])
    else:
        bank.form()
    validate_full(bank, plan)
    snapshot(bank, full)
    sealed_write(folder/'full/IDENTITY.json', {'plan_sha256': digest(plan), 'state_identity': bank.state_identity()})
    write_json(folder/'PROGRESS.json', {'stage': 'full_complete', 'objects': len(bank.rows),
        'completed_turns': len(bank.turns), 'full': str(full), 'formation_llm_calls': 0})
    return str(full)


async def delete(folder, plan, completions=None, *, factory=None, claim_ids=None, strategies=('S0', 'S1', 'S2')):
    if not strategies or not set(strategies) <= {'S0', 'S1', 'S2'}:
        raise ValueError('Choose S0/S1/S2')
    folder = Path(folder)
    factory = factory or Factory(plan)
    full = folder/'full/full.pkl'
    if read(folder/'full/IDENTITY.json')['plan_sha256'] != digest(plan):
        raise ValueError('Full belongs to another plan')

    async def save(claim, condition, bank, receipt, pool_sha=None):
        target = folder/'claims'/claim['claim_id']/condition
        path = target/'state.pkl'
        if path.exists() and path.with_suffix('.frozen.json').exists():
            bank.restore(path)
            receipt = bank.progress.get('last_receipt', {})
        else:
            target.mkdir(parents=True, exist_ok=True)
            if condition == 'Full':
                snapshot(bank, path, reference=full)
            else:
                bank.progress['last_receipt'] = receipt
                snapshot(bank, path)
        if condition != 'Full':
            sealed_write(target/'RECEIPT.json', receipt)
        if (target/'STATE.json').exists():
            state = validate_state(folder, plan, claim, condition)
            if state['state_identity'] != bank.state_identity() or state['pool_sha256'] != pool_sha:
                raise ValueError('Saved branch identity differs')
        else:
            ctx = await bank.contexts(claim)
            sealed_write(target/'STATE.json', {'claim_id': claim['claim_id'], 'condition': condition,
                'plan_sha256': digest(plan), 'snapshot_sha256': sha(path),
                'state_identity': bank.state_identity(), 'pool_sha256': pool_sha,
                'deletion': receipt, 'contexts': ctx})
        return path

    for claim in stage_claims(plan, claim_ids):
        base = folder/'claims'/claim['claim_id']
        bank = factory.create(full)
        validate_full(bank, plan)
        await save(claim, 'Full', bank, {})
        bank = factory.create(full)
        receipt = await bank.delete_turns(claim['disclosure_turns'])
        s0 = await save(claim, 'S0', bank, receipt)
        if set(bank.meta['deleted_turns']) != set(claim['disclosure_turns']):
            raise ValueError('S0 did not delete exactly Direct')
        if set(strategies) == {'S0'}:
            continue
        pool_path = base/'POOL.json'
        pool = read(pool_path) if pool_path.exists() else await bank.build_pool(claim['deletion_request'],
            claim['disclosure_turns'], n_obj=plan['protocol']['n_obj'], n_turn=plan['protocol']['n_turn'])
        verify_pool(pool)
        if (pool['state_identity'] != bank.state_identity() or pool['requested_objects'] != plan['protocol']['n_obj']
                or pool['requested_candidate_turns'] != plan['protocol']['n_turn']):
            raise ValueError('Candidate pool differs from frozen shared S0')
        sealed_write(pool_path, pool)
        orders = {'S1': [r['turn_id'] for r in pool['turns']]}
        if 'S2' in strategies:
            path = base/'S2_ORDER.json'
            if path.exists():
                ranking = read(path)
            else:
                if completions is None and pool['turns']:
                    raise ValueError('S2 needs a selector completion transport')
                ranking = {**rank_leakcut(pool, role_cache(completions, 'selector'), plan['models']['selector']),
                    'pool_sha256': pool['pool_sha256'], 'plan_sha256': digest(plan)}
            if ranking['pool_sha256'] != pool['pool_sha256'] or ranking['plan_sha256'] != digest(plan):
                raise ValueError('S2 ranking belongs to another pool or plan')
            sealed_write(path, ranking)
            orders['S2'] = ranking['order']
        for strategy in ('S1', 'S2'):
            if strategy not in strategies:
                continue
            order = orders[strategy]
            if len(order) != len(set(order)) or set(order) != set(orders['S1']):
                raise ValueError('Ordering must cover the pool exactly once')
            for k in plan['k_list']:
                bank = factory.create(s0)
                receipt = await bank.delete_turns(order[:k])
                receipt.update(requested_k=k, actual_k=min(k, len(order)), budget_complete=len(order)>=k,
                    extra_deleted_turns=order[:k], direct_turns=claim['disclosure_turns'])
                await save(claim, f'{strategy}@{k}', bank, receipt, pool['pool_sha256'])
                expected = set(claim['disclosure_turns']) | set(order[:k])
                if set(bank.meta['deleted_turns']) != expected or any(bank.reverse_trace(t) for t in expected):
                    raise ValueError('Saved condition did not erase the selected turns')
        write_json(folder/'PROGRESS.json', {'stage': 'claim_deleted', 'claim_id': claim['claim_id']})



class AnswerBudget:
    def __init__(self, delegate, tokens):
        self.delegate, self.tokens = delegate, tokens

    def complete(self, request):
        return self.delegate.complete({**request, 'max_tokens': self.tokens})


def answer_request(plan, claim, context, rotation):
    from memcut.source_anchored_v2.evaluation import answer_request as original
    return {**original(claim, context, rotation, plan['models']['answer']), 'max_tokens': plan['qa_max_tokens']}


def _validate_result(plan, claim, family, row, state):
    from memcut.answer_execution import early_stop
    from memcut.source_anchored_v2 import evaluation
    result, context = row['result'], state['contexts'][family]
    if (row['plan_sha256'] != digest(plan) or row['claim_id'] != claim['claim_id']
            or row['condition'] != state['condition'] or row['family'] != family
            or result['context_sha256'] != context['context_sha256']
            or result['state_identity'] != state['state_identity']
            or bool(result.get('evidence_derived_empty')) != bool(context['empty'])):
        raise ValueError('Result does not match its frozen context/plan')
    if plan['early_stop']:
        early_stop.validate_result(result, claim, family)
    else:
        evaluation.summarize([row], [claim], [state['condition']])
        for a in result['answers']:
            rotation = next(r for r in claim['rotations'] if r['trial'] == a['trial'])
            checked = evaluation.score_response({'text': __import__('json').dumps(a['parsed']),
                'request_id': a['request_id'], 'finish_reason': 'stop'}, rotation,
                claim['target_choice_id'], claim['choices'])
            if checked != a:
                raise ValueError('Recorded answer does not match fixed scoring')
    return result


def answer(folder, plan, completions, *, claim_ids=None, conditions=None):
    from memcut.answer_execution import early_stop
    from memcut.source_anchored_v2 import evaluation
    selected = plan['conditions'] if conditions is None else conditions
    if not selected or not set(selected) <= set(plan['conditions']):
        raise ValueError('Unknown/empty answer conditions')
    recovery = AnswerBudget(role_cache(completions, 'answer'), plan['qa_max_tokens'])
    evaluator = early_stop.evaluate if plan['early_stop'] else evaluation.evaluate
    for claim in stage_claims(plan, claim_ids):
        for condition in plan['conditions']:
            if condition not in selected:
                continue
            state = validate_state(folder, plan, claim, condition)
            for family, context in state['contexts'].items():
                path = Path(folder)/'claims'/claim['claim_id']/condition/(family+'.json')
                if path.exists():
                    _validate_result(plan, claim, family, read(path), state)
                    continue
                result = evaluator(claim, context, recovery, plan['models']['answer'])
                row = {'claim_id': claim['claim_id'], 'condition': condition, 'family': family,
                       'plan_sha256': digest(plan), 'result': result}
                _validate_result(plan, claim, family, row, state)
                sealed_write(path, row)
        write_json(Path(folder)/'PROGRESS.json', {'stage':'claim_answered','claim_id':claim['claim_id']})


def evidence_manifest(folder):
    """Bind audits to physical states, all responses, usage and retry evidence."""
    folder = Path(folder)
    paths = [folder/name for name in ('PLAN.json','PLAN_IDENTITY.json','PROTOCOL.md','TURN_RAG_REFERENCE.json')]
    for name in ('full','claims','requests','SESSION_RETRY_JOURNAL'):
        root = folder/name
        if root.is_symlink():
            raise ValueError('Audit evidence must not contain symlinks: '+str(root))
        paths.extend(root.rglob('*'))
    # rglob does not descend into symlink directories. Reject them (and linked
    # files) rather than silently issuing a proof that omits external evidence.
    if any(p.is_symlink() for p in paths):
        raise ValueError('Audit evidence must not contain symlinks')
    return {str(p.relative_to(folder)):sha(p) for p in sorted(set(paths))
            if p.is_file() and not any(part.startswith('.') for part in p.relative_to(folder).parts)}


def report(folder, plan):
    from memcut.answer_execution import early_stop
    from memcut.source_anchored_v2 import evaluation
    folder = Path(folder)
    rows, files = [], {}
    for claim in plan['claims']:
        for condition in plan['conditions']:
            for family in ('native','residual'):
                path = folder/'claims'/claim['claim_id']/condition/(family+'.json')
                if not path.exists():
                    continue
                state = validate_state(folder, plan, claim, condition)
                row = read(path)
                _validate_result(plan, claim, family, row, state)
                rows.append(row); files[str(path.relative_to(folder))] = sha(path)
    summarize = early_stop.summarize if plan['early_stop'] else evaluation.summarize
    result = {'system': 'session_rag', 'plan_sha256':digest(plan),
              **summarize(rows, plan['claims'], plan['conditions'])}
    result['main_complete'] = all(m['completed_claims'] == len(plan['claims']) for m in result['metrics'].values())
    result['budgets'] = {}
    for condition in plan['conditions']:
        if '@' not in condition:
            continue
        receipts = []
        for claim in plan['claims']:
            path = folder/'claims'/claim['claim_id']/condition/'RECEIPT.json'
            if path.exists():
                r = read(path)
                receipts.append({'claim_id':claim['claim_id'], **{k:r[k] for k in ('requested_k','actual_k','budget_complete')}})
        result['budgets'][condition] = receipts
        for family in ('native','residual'):
            result['metrics'][family+'-'+condition]['all_claims_reached_requested_budget'] = (
                len(receipts)==len(plan['claims']) and all(r['budget_complete'] for r in receipts))
    if 'S2@10' in plan['conditions']:
        result['named_metrics'] = {f+'-leakcut':result['metrics'][f+'-S2@10'] for f in ('native','residual')}
    # Category counts come only from pre-frozen labels, never model outcomes.
    categories = {}
    for category in sorted({str(c['category']) for c in plan['claims']}):
        cohort = [c for c in plan['claims'] if str(c['category']) == category]
        ids = {c['claim_id'] for c in cohort}
        categories[category] = summarize([r for r in rows if r['claim_id'] in ids], cohort, plan['conditions'])
    result['result_files_sha256'] = digest(files)
    audit_path = folder/'RESULT_AUDIT.json'
    result['audited_complete'] = bool(result['main_complete'] and audit_path.exists()
        and read(audit_path).get('status')=='PASS' and read(audit_path).get('result_files_sha256')==digest(files)
        and read(audit_path).get('plan_sha256')==digest(plan)
        and read(audit_path).get('evidence_files')==evidence_manifest(folder))
    write_json(folder/'SUMMARY.json', result)
    write_json(folder/'CATEGORY_RESULTS.json', {'plan_sha256':digest(plan),'categories':categories})
    from memcut.execution_optimization.receipts import usage
    write_json(folder/'USAGE.json', usage(folder))
    return result


async def coverage(folder, plan, *, factory=None, k_values=(1,3,16)):
    """Measure actual retrieved texts locally; never calls an answer model."""
    import statistics
    factory = factory or Factory(plan)
    records = []
    for claim in plan['claims']:
        for condition in plan['conditions']:
            path = Path(folder)/'claims'/claim['claim_id']/condition/'state.pkl'
            if condition == 'Full':
                path = Path(folder)/'full/full.pkl'
            if not path.exists():
                continue
            for k in k_values:
                if type(k) is not int or k <= 0:
                    raise ValueError('Coverage k must be positive')
                bank = factory.create(path)
                # Read-only counterfactual retrieval, not a change to frozen stored state.
                old = bank.settings['native_eval_k']
                try:
                    bank.settings['native_eval_k'] = k
                    native = (await bank.contexts(claim))['native']
                finally:
                    bank.settings['native_eval_k'] = old
                fields = {key:value for key,value in native.items() if key not in {'context','retrieval'}}
                records.append({'claim_id':claim['claim_id'],'condition':condition,'k':k,**fields})
    groups = []
    for condition,k in sorted({(r['condition'],r['k']) for r in records}):
        rr = [r for r in records if r['condition']==condition and r['k']==k]
        vv = [r['episode_coverage'] for r in rr]
        groups.append({'condition':condition,'k':k,'claims':len(rr),'mean_episode_coverage':statistics.mean(vv),
            'min_episode_coverage':min(vv),'max_episode_coverage':max(vv),'above_25_percent':statistics.mean(vv)>.25})
    result = {'plan_sha256':digest(plan),'api_calls':0,'token_basis':'cl100k_base shared numerator/denominator',
        'not_model_performance':True,'summary':groups,'records':records}
    write_json(Path(folder)/'COVERAGE.json', result)
    return result


async def audit(folder, plan, *, factory=None, require_complete=True, verify_api=True):
    """Rebuild contexts from physical states and rescore original provider receipts."""
    from .transport import verify_receipt
    from memcut.source_anchored_v2 import evaluation
    from memcut.source_anchored_v2.candidates import selector_input
    from memcut.efficient_execution.recovery import validate as validate_schema
    import json
    folder = Path(folder); factory = factory or Factory(plan)
    full = factory.create(folder/'full/full.pkl');validate_full(full,plan)
    full_sha = sha(folder/'full/full.pkl')
    checked_states = checked_answers = 0
    for claim in plan['claims']:
        base = folder/'claims'/claim['claim_id']
        pool = read(base/'POOL.json') if (base/'POOL.json').exists() else None
        if pool:
            s0 = factory.create(base/'S0/state.pkl')
            rebuilt = await s0.build_pool(claim['deletion_request'],claim['disclosure_turns'],
                n_obj=plan['protocol']['n_obj'],n_turn=plan['protocol']['n_turn'])
            if rebuilt != pool:
                raise ValueError('Saved candidate pool differs from actual S0 retrieval')
        rank = read(base/'S2_ORDER.json') if (base/'S2_ORDER.json').exists() else None
        if rank:
            # Capture exact selector request using the production prompt builder.
            captured=[]
            class Reader:
                def complete(self, request):
                    captured.append(request)
                    if verify_api:
                        return verify_receipt(folder,plan,'selector',request,rank['request_id'])
                    payload, groups=selector_input(pool)
                    scores={r['turn_id']:r for r in rank['scores']}
                    return {'text':json.dumps({'scores':[{'id':g['group'],'score':scores[g['turn_ids'][0]]['llm_score'],
                        'reason':scores[g['turn_ids'][0]]['llm_reason']} for g in groups]}),'request_id':rank['request_id']}
            rerank=rank_leakcut(pool,Reader(),plan['models']['selector'])
            if any(rank[k]!=v for k,v in rerank.items()):
                raise ValueError('S2 ranking disagrees with provider receipt')
        for condition in plan['conditions']:
            path=base/condition
            if not (path/'STATE.json').exists():
                if require_complete:raise ValueError('Missing condition state: '+str(path))
                continue
            state=validate_state(folder,plan,claim,condition);bank=factory.create(path/'state.pkl')
            expected=set() if condition=='Full' else set(claim['disclosure_turns'])
            if '@' in condition:
                strategy,k=condition.split('@');order=[r['turn_id'] for r in pool['turns']] if strategy=='S1' else rank['order']
                expected.update(order[:int(k)])
                receipt=read(path/'RECEIPT.json')
                if (receipt['extra_deleted_turns']!=order[:int(k)] or receipt['actual_k']!=min(int(k),len(order))
                    or receipt['requested_k']!=int(k) or receipt['budget_complete']!=(len(order)>=int(k))):
                    raise ValueError('Deletion receipt budget differs')
            if set(bank.meta['deleted_turns'])!=expected:
                raise ValueError('Saved state did not delete exact selected source turns')
            live={t['turn_id']:t['text'] for row in bank.rows.values() for t in row['turns']}
            original={t:row['text'] for t,row in factory.turns.items() if t not in expected}
            if live!=original:raise ValueError('Deleted/mutated/unselected live turn body')
            contexts=await bank.contexts(claim)
            if contexts!=state['contexts'] or bank.state_identity()!=state['state_identity']:
                raise ValueError('Saved contexts differ from actual live retrieval/state')
            wanted=set(claim['evidence_turns']+claim['disclosure_turns'])-expected
            reference='\n'.join(factory.turns[t]['text'] for t in factory.turns if t in wanted)
            if contexts['residual']['context']!=reference:
                raise ValueError('Residual is not exactly the live evidence/Direct raw text')
            checked_states+=1
            for family,context in contexts.items():
                answer_path=path/(family+'.json')
                if not answer_path.exists():
                    if require_complete:raise ValueError('Missing answer '+str(answer_path))
                    continue
                row=read(answer_path);res=_validate_result(plan,claim,family,row,state)
                for saved in res['answers']:
                    rotation=next(r for r in claim['rotations'] if r['trial']==saved['trial'])
                    request=answer_request(plan,claim,context,rotation)
                    if verify_api:
                        response=verify_receipt(folder,plan,'answer',request,saved['request_id'])
                        if evaluation.score_response(response,rotation,claim['target_choice_id'],claim['choices'])!=saved:
                            raise ValueError('Answer score disagrees with original provider response')
                    checked_answers+=1
    if sha(folder/'full/full.pkl')!=full_sha:raise ValueError('Full changed during audit')
    summary=report(folder,plan)
    result={'status':'PASS' if summary['main_complete'] and verify_api else 'PARTIAL_LOCAL_PASS',
        'plan_sha256':digest(plan),'result_files_sha256':summary['result_files_sha256'],
        'states_checked':checked_states,'provider_answers_checked':checked_answers if verify_api else 0,
        'main_complete':summary['main_complete'],'provider_receipts_verified':verify_api,'api_calls':0,
        'full_sha256':full_sha,'evidence_files':evidence_manifest(folder)}
    write_json(folder/'RESULT_AUDIT.json',result)
    report(folder,plan)
    return result


def gateway(folder,plan):
    from .transport import Gateway
    client=Gateway(folder,plan)
    return client,client
