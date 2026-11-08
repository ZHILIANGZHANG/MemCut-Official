"""Parallel API waits, original request bodies, serial native storage commits."""
from pathlib import Path

from memcut.native_memory.contract import digest
from memcut.source_anchored_v2 import runner as core
from memcut.source_anchored_v2.completions import native_summary_request
from memcut.source_anchored_v2.memorybank import import_source
from memcut.source_anchored_v2.evaluation import evaluate
from memcut.source_anchored_systems.formation_runtime.concurrency import ConcurrentCompletions, ordered_map
from .receipts import request_identity


def session_requests(seed, source):
    """Capture upstream request builders on the main thread (they patch globals)."""
    turns, groups, _ = import_source(source)
    requests = []
    for date, pairs in groups.items():
        rows = [{'query': turns[q]['text'], 'response': turns[a]['text']} for q, a in pairs]
        for prompt in (seed.summaries.summarize_content_prompt(rows, seed.user_id, 'AI', 'en'),
                       seed.summaries.summarize_person_prompt(rows, seed.user_id, 'AI', 'en')):
            requests.append(native_summary_request(seed.summaries, prompt, seed.model))
    return requests


class PrefetchedFormation:
    """Refuse mismatches instead of silently paying for a different request."""
    def __init__(self, cache, requests):
        self.cache, self.requests, self.position = cache, requests, 0

    def __getattr__(self, name):
        return getattr(self.cache, name)

    def complete(self, request, **kwargs):
        if self.position < len(self.requests):
            if request != self.requests[self.position]:
                raise ValueError('Native formation request differs from its prefetched request')
            key = digest(request_identity(self.cache, request))
            if not (Path(self.cache.folder) / key / 'response.json').exists():
                raise RuntimeError('Prefetched receipt missing; do not dispatch again')
            response = self.cache.complete(request, **kwargs)
            if not response['cached']:
                raise RuntimeError('Prefetched native request was unexpectedly dispatched again')
        else:
            if self.position >= len(self.requests) + 2:
                raise ValueError('Unexpected extra MemoryBank generation')
            # The two original global summaries depend on committed daily fields.
            response = self.cache.complete(request, **kwargs)
        self.position += 1
        return response


async def build_memorybank(folder, plan, gateway, *, workers=4, seed=None):
    folder = Path(folder)
    if plan.get('reuse_full') or (folder / 'full/full.pkl').exists() or (folder / 'formation/state.pkl').exists():
        return await core.build(folder, plan, gateway, seed=seed)
    seed = seed or core.template(folder, plan)
    requests = session_requests(seed, core.read(plan['inputs']['source']['path']))
    writer = gateway.for_role('write')
    parallel = ConcurrentCompletions(writer, workers)
    def generate(request):
        return parallel.complete(request, allow_length=True)
    for _ in ordered_map(generate, requests, workers):
        pass
    prefetched = PrefetchedFormation(writer, requests)
    # The original form/build implementation still owns all ingestion, diffs,
    # parent bindings, global summaries, index construction and freezing.
    result = await core.build(folder, plan, prefetched, seed=seed)
    if prefetched.position != len(requests) + 2:
        raise ValueError('Incomplete native formation replay')
    return result


def answer(folder, plan, gateway, *, workers=4, claim_ids=None, source_import=import_source):
    folder = Path(folder)
    selected = core.stage_claims(plan, claim_ids)
    jobs = []
    # Resolve and validate every snapshot before dispatch. Workers never access
    # a mutable bank, retrieval engine, or another answer's conversation.
    for claim in selected:
        for condition in plan['conditions']:
            state = core.validate_state(folder, plan, claim, condition)
            for family, context in state['contexts'].items():
                target = folder / 'claims' / claim['claim_id'] / condition / f'{family}.json'
                envelope = {'claim_id': claim['claim_id'], 'condition': condition, 'family': family,
                            'plan_sha256': digest(plan)}
                jobs.append((claim, context, target, envelope))
    if plan['controls']:
        turns, _, _ = source_import(core.read(plan['inputs']['source']['path']))
        for claim in selected:
            for name, text in [('no_memory', ''), ('complete_evidence', '\n'.join(
                    turns[t]['text'] for t in dict.fromkeys(claim['evidence_turns'] + claim['disclosure_turns'])))]:
                context = {'context': text, 'context_sha256': digest(text), 'state_identity': name, 'empty': False}
                jobs.append((claim, context, folder / 'controls' / claim['claim_id'] / f'{name}.json', None))
    cache = ConcurrentCompletions(gateway.for_role('answer'), workers)
    def run(job):
        claim, context, target, envelope = job
        # Original evaluate performs both fixed permutations, even after a miss.
        try:
            result = evaluate(claim, context, cache, plan['models']['answer'])
            return target, result if envelope is None else {**envelope, 'result': result}
        except BaseException:
            cache.failed.set()
            raise
    for target, result in ordered_map(run, jobs, workers):
        core.sealed_write(target, result)
