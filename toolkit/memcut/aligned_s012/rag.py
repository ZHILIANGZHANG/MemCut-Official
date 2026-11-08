"""The two retrieval controls, Raw-turn RAG and Session RAG, on the aligned entry.

Their frozen lifecycles stay as they are (memcut/raw_turn_rag, memcut/session_rag, tools/raw_turn_rag.py,
tools/session_rag.py). The aligned entry runs them like the four memory systems, with its transport policy
(transport.py), unavailable readouts and the uniform answer cap (answers.py, runtime.ANSWER_POLICY):
- build takes no completion transport (formation is local encoding), so the stage drops the one it is given;
- Session RAG keeps the gate of its frozen `run` command: before any paid request it measures how much of the
  episode its native context shows. S0 and S1 are local, so they run first; COVERAGE.json records the
  measurement, and a mean coverage above a quarter of the episode stops the run unless the plan accepted it
  at prepare (`--accept-coverage`), or a smaller `--native-eval-k` is planned instead.
"""
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

from memcut.native_memory.contract import digest
from memcut.source_anchored_v2.runner import read


class CoverageReview(ValueError):
    """Session RAG's native context shows more than a quarter of the episode and the plan did not accept it."""


async def measured_coverage(base, folder, plan):
    """COVERAGE.json of this plan at its native k: measured once, locally, after S0 and S1."""
    path = Path(folder) / 'COVERAGE.json'
    k = plan['settings']['native_eval_k']
    if path.exists():
        measured = read(path)
        if measured.get('plan_sha256') == digest(plan) and {g['k'] for g in measured['summary']} == {k}:
            return measured
    return await base.coverage(folder, plan, k_values=(k,))


@contextmanager
def stages(base, system):
    original_build, original_delete = base.build, base.delete

    def build(folder, plan, completions=None):
        return original_build(folder, plan)

    async def gated_delete(folder, plan, completions=None, **kwargs):
        await original_delete(folder, plan, None, strategies=('S0', 'S1'))
        over = [g for g in (await measured_coverage(base, folder, plan))['summary'] if g['above_25_percent']]
        if over and not plan['aligned_options']['accept_coverage']:
            worst = max(over, key=lambda g: g['mean_episode_coverage'])
            raise CoverageReview(
                f"Session RAG's native context (k={worst['k']}) shows {worst['mean_episode_coverage']:.0%} of the "
                f"episode on average in {worst['condition']} (limit 25%; COVERAGE.json). No paid request was sent. "
                'Prepare a new run with a smaller --native-eval-k, or with --accept-coverage to keep this k.')
        return await original_delete(folder, plan, completions, **kwargs)

    with ExitStack() as stack:
        stack.enter_context(patch.object(base, 'build', build))
        if system == 'sessionrag':
            stack.enter_context(patch.object(base, 'delete', gated_delete))
        yield base
