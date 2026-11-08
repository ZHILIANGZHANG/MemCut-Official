"""Scoped overrides for new plans; historical runners and profiles stay intact."""
import copy
from contextlib import ExitStack, contextmanager
from functools import lru_cache, partial
import importlib
from pathlib import Path
from unittest.mock import patch

from . import RAG, SYSTEMS, VERSION
from memcut.source_anchored_v2 import runner as core

ROOT = Path(__file__).resolve().parents[2]
# One answer cap for the four systems: the reply is a reason plus a label, and an answer cut off at a cap is
# unavailable, not wrong. LightMem's formally accepted QA budget is 2048,
# 400 (Nemori, LightMem) and 1024 (MemoryBank) were inherited defaults.
ANSWER_POLICY = {'version': 'aligned-uniform-qa2048-v2', 'max_tokens': 2048,
                 'all_systems': list(SYSTEMS), 'unusable_answer': 'unavailable readout; the stage continues'}
TRIAL_NAMES = {'trial1': 1, 'trial2': 2, 'trial3': 3}


def runner(system):
    modules = {'memorybank': 'memcut.source_anchored_v2.runner',
               'nemori': 'memcut.source_anchored_systems.runner',
               'lightmem': 'memcut.source_anchored_systems.runner',
               'turnrag': 'memcut.raw_turn_rag.runner',
               'sessionrag': 'memcut.session_rag.runner'}
    return importlib.import_module(modules[system])


def identity(system):
    if system not in SYSTEMS:
        raise ValueError('Unknown MemCut memory system')
    files = list(Path(__file__).parent.glob('*.py')) + [ROOT / 'tools/memcut_s012.py',
        ROOT / 'memcut/model_adapters/qwen.py']
    return dict(version=VERSION, system=system,
                code={str(p.relative_to(ROOT)): core.sha(p) for p in sorted(files)})


def native_profile(original, system):
    value = copy.deepcopy(original(system))
    value['settings']['memcut_alignment'] = VERSION
    if system == 'nemori':
        value['settings'].update(enable_episode_merging=True, enable_prediction_correction=True,
                                 extract_semantic_per_episode=False)
        value['settings']['native_generation_caps'].update(merge_decision=4096,
            merge_content=8192, semantic_prediction=8192, semantic_coldstart=8192,
            semantic_correction=8192)
        value['reference'] = ('v0.1.1 native merging and prediction/correction, LoCoMo flush of the final '
                              'partial batch; local snapshot stores')
    elif system == 'lightmem':
        value['settings']['compression_rate'] = .6
        value['reference'] = 'LoCoMo add_locomo.py LightMem(512,0.6); MemCut inline source attribution'
    return value


@contextmanager
def trial_names(system, log):
    """Contracts that name their fixed rotations "trial1"/"trial2" prepare for every system; each rename is
    logged in the plan's input_normalization."""
    module = runner(system)
    original = module.normalize_claim

    def normalize_claim(row):
        normalized = copy.deepcopy(row)
        for rotation in normalized.get('rotations') or []:
            trial = rotation.get('trial')
            if isinstance(trial, str) and trial in TRIAL_NAMES:
                rotation['trial'] = TRIAL_NAMES[trial]
                log.append({'claim_id': row['claim_id'], 'before': trial, 'after': rotation['trial']})
        return original(normalized)

    with patch.object(module, 'normalize_claim', normalize_claim):
        yield


@contextmanager
def runtime(system, *, native=False):
    base = runner(system)
    from . import answers
    from memcut.source_anchored_v2 import evaluation
    with ExitStack() as stack:
        if system in ('nemori', 'lightmem'):
            original_profile = base.load_profile
            # Pin/source checks once per invocation, not once per claim branch.
            resolved = lru_cache(None)(lambda name: native_profile(original_profile, name))
            stack.enter_context(patch.object(base, 'load_profile', lambda name: copy.deepcopy(resolved(name))))
        evaluate = evaluation.evaluate
        if system == 'memorybank':
            from memcut.source_anchored_v2.memorybank import SourceMemoryBank
            from memcut.answer_execution import early_stop
            stack.enter_context(patch.object(SourceMemoryBank, 'native_answer_top_k', 10))
            stack.enter_context(patch.object(core, 'evaluate', early_stop.evaluate))
            stack.enter_context(patch.object(core, 'summarize', early_stop.summarize))
            evaluate = early_stop.evaluate
        elif system in RAG:
            from .rag import stages
            stack.enter_context(stages(base, system))
        # Every system: an unusable answer is an unavailable readout, and the report adds the available cohort.
        stack.enter_context(patch.object(base, 'answer', partial(answers.answer, evaluate=evaluate)))
        stack.enter_context(patch.object(base, 'report', answers.report(base.report)))
        if native and system == 'nemori':
            from memcut.source_anchored_systems.formation_runtime import nemori
            from .nemori import AlignedNemoriBank
            stack.enter_context(patch.object(nemori, 'FastNemoriBank', AlignedNemoriBank))
        elif native and system == 'lightmem':
            from memcut.source_anchored_systems import lightmem
            from memcut.source_anchored_systems.formation_runtime import lightmem as fast
            from .lightmem import AlignedLightMemBank
            original_settings = lightmem.native_settings
            def settings(*args, **kwargs):
                value = original_settings(*args, **kwargs)
                value['pre_compressor']['configs']['compress_config']['rate'] = .6
                return value
            stack.enter_context(patch.object(lightmem, 'native_settings', settings))
            stack.enter_context(patch.object(fast, 'FastLightMemBank', AlignedLightMemBank))
        yield base
