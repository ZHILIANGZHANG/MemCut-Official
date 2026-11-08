"""Nemori and LightMem adapters for the source-turn S0/S1/S2 protocol.

MemoryBank's frozen implementation remains in source_anchored_v2. Candidate
selection and recovery scoring are shared with it; native storage is not.
"""

VERSION = 'source-anchored-nemori-lightmem-v1'

PROTOCOL = {
    'version': VERSION,
    'strategies': ['S0', 'S1', 'S2'],
    'n_obj': 20, 'n_turn': 20, 'k': 10,
    'k_list': [1, 2, 3, 5, 7, 10, 15, 20],
    'budget_unit': 'distinct original single-role source turn; Direct excluded',
    'candidate_search': 'pinned system native object search; shared S0 top20 to source-turn top20',
    'ties': 'best native seed rank, then original turn order',
    'closure': 'native atomic objects erased; aggregates replayed from remaining actual input ancestry',
    'provenance': 'independent native storage diffs plus actual input versions; no semantic attribution judge',
    'residual': 'live der(E+D) union touched(E+D); current stored text in original birth order',
    'native_evaluation': 'stem-only native query and renderer; per-system pinned defaults in settings',
    'temperature': 0,
    'memory_generation_finish': 'complete native JSON, stop only; no silent truncation/fallback',
    'score': 'three frozen permutations all correct; missing/invalid is unavailable',
    'conditional_cohort': 'Full residual 3/3, shared across both readouts',
}
