"""Turn-budget S0/S1/S2; independent of frozen object-budget experiments."""

PROTOCOL = {
    'version': 'memorybank-source-anchored-s012-v2',
    'strategies': ['S0', 'S1', 'S2'],
    'n_obj': 20, 'n_turn': 20, 'k': 10,
    'k_list': [1, 2, 3, 5, 7, 10, 15, 20],
    'budget_unit': 'distinct source turn; Direct excluded',
    'candidate_search': 'native FAISS seed20; score=-distance; native adjacency retained in QA',
    'ties': 'best native seed rank, then original turn order',
    'closure': 'whole raw pair; remaining-source daily then overall regeneration',
    'provenance': 'native storage snapshot diff plus actual input ancestry; no semantic attribution judge',
    'residual': 'live objects in der(E+D) union touched(E+D); current stored text',
    'native_eval_k': 2, 'temperature': 0,
    'memory_generation_finish': 'native nonempty stop/length at max_tokens400; QA and selector require stop',
    'raw_pair': 'atomic; either source turn selects the entire stored pair',
    'grouping': 'original session timestamp, matching existing MemoryBank importer',
    'score': 'three frozen permutations all correct; missing/invalid is unavailable',
    'conditional_cohort': 'Full residual 3/3, shared across both readouts',
}
