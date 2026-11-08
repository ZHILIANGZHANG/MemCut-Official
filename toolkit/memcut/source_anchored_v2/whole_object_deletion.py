"""Current S012 source closure: erase every affected object, with no LLM.

Kept outside formation adapters so an undeleted native Full remains reusable.
The adapters' earlier delete_turns methods retain historical regeneration
semantics; current runners explicitly dispatch through this module.
"""
import copy

POLICY = 'whole-object-source-closure-v1'


def deletion_protocol(protocol):
    return {**copy.deepcopy(protocol),
            'version': protocol['version'] + '-whole-object-v1',
            'deletion_policy': POLICY,
            'closure': 'erase every current derived object, atomic and aggregate; no regeneration',
            'deletion_llm_calls': 0}


async def delete_source_turns(bank, selected):
    """No completion client is accepted: S0/S1/selected-S2 deletion is local."""
    if bank.system == 'memorybank':
        return _delete_memorybank(bank, selected)
    if bank.system not in {'nemori', 'lightmem'}:
        raise ValueError('Unsupported native source-closure system')
    if not bank.frozen:
        raise ValueError('Deletion requires a frozen Full-derived branch')
    added = set(bank.lineage.ordered(selected)) - set(bank.meta['deleted_turns'])
    all_selected = added | set(bank.meta['deleted_turns'])
    affected = {oid for t in added for oid in bank.reverse_trace(t)}
    backup = copy.deepcopy(bank.export())
    before = bank.observation()
    collateral = {t for oid in affected for t in bank.trace(oid)} - all_selected
    diff_start = len(bank.lineage.data['mutation_diffs'])
    try:
        if affected:
            def erase():
                for oid in sorted(affected):
                    bank.erase_native(oid)
            bank.observe_write(erase, {}, label=POLICY)
        bank.lineage.mark_deleted(all_selected)
        after = bank.observation()
        bank.lineage.validate(observed=after)
        if set(before) - set(after) != affected or any(after[o] != before[o] for o in after):
            raise ValueError('Pure deletion changed an unselected object or left an affected object')
    except BaseException:
        bank.restore_value(backup)
        raise
    return {'deletion_policy': POLICY, 'deletion_llm_calls': 0,
            'deleted_turns': bank.lineage.ordered(added),
            'all_deleted_turns': bank.lineage.ordered(all_selected),
            'deleted_objs': [o for o in before if o in affected],
            'rewritten_objs': [], 'created_objs': [],
            'collateral_raw_turns': bank.lineage.ordered(
                {t for o in affected if before[o]['kind']=='atomic' for t in backup['lineage']['src'][o]} - all_selected),
            'collateral_source_turns': bank.lineage.ordered(collateral),
            'selected_turns_have_empty_der': all(not bank.reverse_trace(t) for t in all_selected),
            'mutation_diffs': bank.lineage.data['mutation_diffs'][diff_start:]}


def _delete_memorybank(bank, selected):
    from .memorybank import address
    if bank.checkpoint.with_suffix('.frozen.json').exists():
        raise ValueError('Fork the frozen snapshot before deleting turns')
    selected = set(selected)
    if not selected <= set(bank.meta['turns']):
        raise ValueError('Unknown selected turn')
    added = selected - set(bank.meta['deleted_turns'])
    all_selected = added | set(bank.meta['deleted_turns'])
    affected = {oid for t in added for oid in bank.reverse_trace(t)}
    before_state = bank.state_identity()
    before = bank.object_snapshot()
    collateral = {t for oid in affected for t in bank.trace(oid)} - all_selected
    raw_collateral = set()
    old = bank.memory, bank.pair_ids, bank.meta, bank.store, bank.doc_addresses
    bank.memory, bank.pair_ids, bank.meta = copy.deepcopy(bank.memory), copy.deepcopy(bank.pair_ids), copy.deepcopy(bank.meta)
    diff_start = len(bank.meta['mutation_diffs'])
    try:
        user = bank.memory[bank.user_id]
        for date in bank.meta['groups']:
            kept = []
            for pair, ident in zip(user['history'][date], bank.pair_ids[date]):
                oid = address('history', ident)
                if oid in affected:
                    raw_collateral.update(bank.trace(oid) - all_selected)
                else:
                    kept.append((pair, ident))
            user['history'][date] = [pair for pair, _ in kept]
            bank.pair_ids[date] = [ident for _, ident in kept]
            for layer in ('summary', 'personality'):
                if address(layer, date) in affected:
                    user[layer].pop(date, None)
        for layer in ('overall_history', 'overall_personality'):
            if address(layer, bank.user_id) in affected:
                user[layer] = ''
        if affected:
            bank._observe_write(before, {})
        bank.meta['deleted_turns'] = [t for t in bank.meta['turn_order'] if t in all_selected]
        bank._sync()
        after = bank.object_snapshot()
        if set(before) - set(after) != affected or any(after[o] != before[o] for o in after):
            raise ValueError('Pure deletion changed an unselected field or left an affected field')
        maintenance = bank._refresh_index()
        bank.save()
    except BaseException:
        bank.memory, bank.pair_ids, bank.meta, bank.store, bank.doc_addresses = old
        raise
    return {'deletion_policy': POLICY, 'deletion_llm_calls': 0,
            'selected_turns': [t for t in bank.meta['turn_order'] if t in added],
            'all_deleted_turns': bank.meta['deleted_turns'],
            'deleted_objs': [o for o in before if o in affected],
            'rewritten_objs': [], 'created_objs': [],
            'collateral_raw_turns': sorted(raw_collateral),
            'collateral_source_turns': sorted(collateral), 'maintenance': maintenance,
            'selected_turns_have_empty_der': all(not bank.reverse_trace(t) for t in all_selected),
            'mutation_diffs': bank.meta['mutation_diffs'][diff_start:],
            'before_state': before_state, 'after_state': bank.state_identity()}
