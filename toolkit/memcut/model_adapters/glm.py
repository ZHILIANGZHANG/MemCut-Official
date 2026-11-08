"""Official GLM-5.3-Flash transport with explicit low reasoning and exact cache."""
import copy
import fcntl
import os
from pathlib import Path
import threading

from memcut.native_memory.contract import digest
from memcut.source_anchored_v2.completions import CachedCompletions
from memcut.source_anchored_v2.runner import gateway as legacy_gateway

MODEL = 'glm-5.3-flash'
ENDPOINT = 'https://open.bigmodel.cn/api/paas/v4'
PROFILE = {
    'version': 'glm53-flash-low-v1', 'model': MODEL, 'endpoint': ENDPOINT,
    'thinking': {'type': 'enabled'}, 'reasoning_effort': 'low',
    'output_budget': 'preserve each native/selector/answer request max_tokens',
    'content': 'message.content only; reasoning_content is receipt-only',
    'automatic_retries': 0,
}


def normalize(request):
    sent = copy.deepcopy(request)
    if sent.get('model') != MODEL:
        raise ValueError('GLM transport requires the frozen exact model ID')
    extra = sent.setdefault('extra_body', {})
    # Refuse conflicting caller settings rather than silently hiding a change.
    for key in ('thinking', 'reasoning_effort'):
        expected = PROFILE[key]
        for value in (sent.pop(key, expected), extra.get(key, expected)):
            if value != expected:
                raise ValueError('Request conflicts with frozen GLM reasoning policy')
        extra[key] = copy.deepcopy(expected)
    return sent


class GLMCompletions(CachedCompletions):
    def __init__(self, folder, create, *, before_dispatch=None):
        super().__init__(folder, create, endpoint=ENDPOINT, compatibility='openai',
                         before_dispatch=before_dispatch)

    def complete(self, request, *, allow_length=False):
        sent = normalize(request)
        key = digest({'endpoint': self.endpoint, 'compatibility': self.compatibility, 'request': sent})
        # The native formation concurrency wrapper locks its unnormalized input.
        # Lock the actual request too, so two spellings of the same request share
        # one dispatch. A separate directory avoids reentrant file-lock deadlock.
        locks = self.folder / '.glm_singleflight'
        locks.mkdir(parents=True, exist_ok=True)
        with (locks / key).open('a+b') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            return super().complete(sent, allow_length=allow_length)


class Gateway:
    """Same role-cache interface used by all four accepted system runners."""
    def __init__(self, folder, plan, *, create_factory=None):
        self.folder, self.plan = Path(folder), plan
        self.clients, self.caches = {}, {}
        self.lock = threading.RLock()
        self.create_factory = create_factory
        self.legacy, _ = legacy_gateway(folder, plan)

    def for_role(self, role):
        with self.lock:
            if role not in self.caches:
                provider = self.plan['providers'][role]
                if self.plan['models'][role] != MODEL:
                    self.caches[role] = self.legacy.for_role(role)
                    return self.caches[role]
                if provider['endpoint'].rstrip('/') != ENDPOINT or provider['compatibility'] != 'openai':
                    raise ValueError('GLM requires its official endpoint and OpenAI-compatible transport')
                def ensure_client():
                    with self.lock:
                        if role in self.clients or self.create_factory:
                            return
                        from openai import OpenAI
                        key = os.environ.get('MEMCUT_GLM_API_KEY') or os.environ.get(provider['key_env'])
                        if not key:
                            raise ValueError('MEMCUT_GLM_API_KEY is required; never put the key in a CLI argument')
                        self.clients[role] = OpenAI(api_key=key, base_url=ENDPOINT,
                                                   max_retries=0, timeout=180)
                def create(**kwargs):
                    if self.create_factory:
                        return self.create_factory(role, kwargs)
                    return self.clients[role].chat.completions.create(**kwargs)
                self.caches[role] = GLMCompletions(self.folder/'requests', create, before_dispatch=ensure_client)
            return self.caches[role]

    @property
    def calls(self):
        return [call for cache in self.caches.values() for call in cache.calls]

    def close(self):
        for client in self.clients.values():
            client.close()
        self.legacy.close()
