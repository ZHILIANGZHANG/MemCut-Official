"""Independent storage/lineage/receipt validation; never dispatch a model API."""
import asyncio
from dataclasses import asdict
from pathlib import Path

from memcut.native_memory.contract import digest
from memcut.source_anchored_v2 import evaluation
from memcut.source_anchored_v2.candidates import build_pool, rank_leakcut, verify_pool
from memcut.execution_optimization.receipts import request_identity
from . import runner
from .formation import extraction_request, parse_entries
from .native import MemoryEntry, native_render
from .bank import address


def require(ok, message):
    if not ok:
        raise ValueError(message)


def _retrieval(bank, saved, expected_query, cache):
    require(saved['query'] == expected_query, 'Retrieval query differs from the frozen claim')
    live = {o['native_id']: o for o in bank.native_objects()}
    costs = saved['costs']
    ids = [o['native_id'] for o in saved['ranked']['retrieved_document']]
    union = costs['union_before_cap']
    cap = costs['requested_object_cap']
    require(len(set(union)) == len(union) and set(union) <= live.keys(), 'Native union contains missing/deleted entries')
    require(ids == (union if cap is None else union[:cap]), 'Final native object cap changed')
    entries = [MemoryEntry.model_validate(live[i]['value']) for i in ids]
    require(saved['native_context']['prompt'] == native_render(entries), 'Native renderer differs from actual live storage')
    for row, entry in zip(saved['ranked']['retrieved_document'], entries):
        require(row['text'] == native_render([entry]).split('\n', 1)[1]
                and row['version'] == digest(live[entry.entry_id]['value']), 'Native visible object text/version differs')
    seeds = costs['native_seed_hits']
    require(seeds == [{'address': address(i), 'rank': n+1, 'distance': float(n)} for n, i in enumerate(ids)],
            'Candidate rank is not the frozen native ordered-union position')
    for call in costs['native_calls']:
        require(all(h['entry_id'] in live for h in call['hits']), 'A native search path returned a deleted entry')
        expected_k = (cap or bank.settings['retrieval']['semantic_top_k']) if call['path'] == 'semantic' else bank.settings['retrieval'][call['path']+'_top_k']
        require(call['top_k'] == expected_k, 'Native search actually used a different k')
    for request in costs['planning_requests']:
        identity = request_identity(cache, request['request'])
        require(digest(identity) == request['request_id'], 'Planning request identity differs')
        response = cache.complete(request['request'])
        require(response['request_id'] == request['request_id'] and response['cached'], 'Planning response unavailable locally')
    require(set(union) <= {h['entry_id'] for call in costs['native_calls'] for h in call['hits']},
            'Native union contains an object absent from all actual search calls')


async def run(folder, plan, *, factory=None, completions=None):
    folder = Path(folder)
    cache_gateway = completions or runner.gateway(folder, plan, cached_only=True)
    factory = factory or runner.Factory(plan)
    full_path = folder/'full/full.pkl'
    full_hash = runner.sha(full_path)
    full = factory.create(full_path)
    own = completions is None
    state_count = answer_count = 0
    try:
        runner.validate_full(full, plan)
        writer = cache_gateway.for_role('write')
        live_full = {o['id']: o for o in full.native_objects()}
        attributed = set()
        for batch in full.progress['formation_batches']:
            messages = extraction_request(full.turns, batch['input_turn_ids'], batch['input_parents'])
            response = writer.complete({'model': plan['models']['write'], 'messages': messages,
                'temperature': 0, 'max_tokens': plan['settings']['write_max_tokens']})
            require(response['cached'] and response['request_id'] == batch['request_id'], 'Formation request differs')
            entries = parse_entries(response['text'], batch['input_turn_ids'], batch['input_parents'],
                                    window_identity=batch['window_identity'])
            require(entries == batch['entries'], 'Inline source binding differs from the actual generation response')
            for row in entries:
                entry = row['entry']
                oid = address(entry['entry_id'])
                require(oid not in attributed and oid in live_full, 'Missing/duplicate derived fact')
                require(full.trace(oid) == set(row['turns']), 'Fact roots differ from its declared contributors')
                actual = live_full[oid]['value']
                expected = {k: (v or '' if k in ('timestamp','location','topic') else v) for k, v in entry.items()}
                require(actual == expected, 'Stored fact differs from its generation')
                attributed.add(oid)
        require(attributed == set(live_full), 'Some native objects have no generation/source binding')
        runner.sealed_write(folder/'PROVENANCE_AUDIT.json', {'status': 'PASS', 'objects': len(attributed),
            'batches': len(full.progress['formation_batches']), 'semantic_accuracy_certified': False,
            'plan_sha256': digest(plan)})
        for claim in plan['claims']:
            root = folder/'claims'/claim['claim_id']
            pool = runner.read(root/'POOL.json')
            order = runner.read(root/'S2_ORDER.json')
            verify_pool(pool)
            expected_order = rank_leakcut(pool, cache_gateway.for_role('selector'), plan['models']['selector'])
            require(expected_order == {k: v for k, v in order.items() if k not in ('pool_sha256','plan_sha256')},
                    'S2 ordering differs from the original response')
            require(order['pool_sha256'] == pool['pool_sha256'] and order['plan_sha256'] == digest(plan), 'S2 binding changed')
            state_rows = []
            for condition in plan['conditions']:
                state = runner.core.validate_state(folder, plan, claim, condition)
                bank = factory.create(root/condition/'state.pkl')
                try:
                    selected = set() if condition == 'Full' else set(claim['disclosure_turns'])
                    if '@' in condition:
                        strategy, k = condition.split('@')
                        candidates = [r['turn_id'] for r in pool['turns']] if strategy == 'S1' else order['order']
                        selected.update(candidates[:int(k)])
                        receipt = state['deletion']
                        require(receipt['actual_k'] == min(int(k), len(candidates)) and
                            receipt['extra_deleted_turns'] == candidates[:int(k)] and
                            receipt['budget_complete'] == (len(candidates) >= int(k)), 'Source-turn budget differs')
                    affected = {o for t in selected for o in full.reverse_trace(t)}
                    observed = bank.observation()
                    require(observed == {o: v for o, v in full.observation().items() if o not in affected},
                            'Branch is not Full minus the exact selected source closure')
                    full_vectors = {r['entry_id']: r for r in full.dump_native()['rows']}
                    require(all(r == full_vectors[r['entry_id']] for r in bank.dump_native()['rows']), 'Survivor vector/storage changed')
                    require(set(bank.meta['deleted_turns']) == selected and bank.state_identity() == state['state_identity'], 'Branch identity differs')
                    require(all(bank.trace(o) == full.trace(o) for o in observed), 'Survivor sources changed')
                    if condition != 'Full':
                        receipt = runner.read(root/condition/'RECEIPT.json')
                        require(receipt == state['deletion'] and receipt['deletion_llm_calls'] == 0
                                and not receipt['created_objs'] and not receipt['rewritten_objs'], 'Not a pure deletion receipt')
                    _retrieval(bank, state['contexts']['native']['retrieval'], claim['stem'], cache_gateway.for_role('retrieval'))
                    require(state['contexts']['native']['context'] == state['contexts']['native']['retrieval']['native_context']['prompt']
                            and state['contexts']['native']['empty'] is False, 'Native answer input differs from the actual renderer')
                    require(state['contexts']['native']['obj_ids'] == [address(i) for i in state['contexts']['native']['retrieval']['native_context']['entry_ids']], 'Native object IDs differ')
                    residual = await bank.residual_objects(claim['evidence_turns'] + claim['disclosure_turns'])
                    ctx = state['contexts']['residual']
                    require(ctx['obj_ids'] == [o.address for o in residual] and ctx['context'] == '\n'.join(o.text for o in residual)
                            and ctx['empty'] == (not residual), 'Residual is not all and only live evidence-derived text')
                    if condition == 'S0':
                        _retrieval(bank, pool['native_retrieval'], claim['deletion_request'], cache_gateway.for_role('retrieval'))
                        # Reuse exactly the first observed native retrieval; independently
                        # rerun the public source-turn reducer with no planner/search API.
                        from memcut.native_memory.contract import SearchResult, MemoryObject
                        value = pool['native_retrieval']
                        result = SearchResult(value['query'], {k: [MemoryObject(**o) for o in v] for k, v in value['ranked'].items()},
                            [MemoryObject(**o) for o in value['resident']], value['native_context'], value['costs'])
                        async def frozen_search(*args, **kwargs):
                            return result
                        from unittest.mock import patch
                        with patch.object(bank, 'search', frozen_search):
                            reconstructed = await build_pool(bank, claim['deletion_request'], claim['disclosure_turns'],
                                n_obj=plan['protocol']['n_obj'], n_turn=plan['protocol']['n_turn'])
                        require(reconstructed == pool, 'Candidate turn mapping/order differs from frozen native exposure')
                    with runner.answer_runtime(plan):
                        for family, context in state['contexts'].items():
                            scored = evaluation.evaluate(claim, context, cache_gateway.for_role('answer'), plan['models']['answer'])
                            saved = runner.read(root/condition/(family+'.json'))
                            require(saved['result'] == scored and saved['plan_sha256'] == digest(plan), 'Answer/2-of-2 score differs')
                            answer_count += len(scored['answers'])
                    state_rows.append({'condition': condition, 'status': 'PASS', 'objects': len(observed)})
                    state_count += 1
                finally:
                    bank.close()
            runner.sealed_write(root/'STATE_AUDIT.json', {'status': 'PASS', 'claim_id': claim['claim_id'], 'states': state_rows})
        summary = runner.report(folder, plan)
        require(summary['main_complete'], 'Experiment has incomplete recovery results')
        require(runner.sha(full_path) == full_hash, 'Protected Full changed')
        result = {'status': 'PASS', 'plan_sha256': digest(plan), 'claims': len(plan['claims']),
            'state_audits': state_count, 'logical_answers': answer_count, 'full_unchanged': True,
            'native_storage': 'actual LanceDB snapshots', 'api_calls': 0,
            'semantic_source_accuracy': 'model-declared sources; not a causal attribution accuracy claim'}
        runner.sealed_write(folder/'RESULT_AUDIT.json', result)
        runner.sealed_write(folder/'COMPLETE.json', {'plan_sha256': digest(plan), 'claims': [c['claim_id'] for c in plan['claims']],
            'full_unchanged': True, 'result_audit_sha256': runner.sha(folder/'RESULT_AUDIT.json')})
        return result
    finally:
        full.close()
        if own:
            cache_gateway.close()
