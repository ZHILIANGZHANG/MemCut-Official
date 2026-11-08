"""A model served on this machine (vLLM, SGLang, llama.cpp, Ollama) through an OpenAI-compatible endpoint.

Installed by `run.py aligned` next to the Qwen profile. It applies when the plan's endpoint is a local address:
- no key is needed (MEMCUT_MODEL_API_KEY defaults to "EMPTY");
- a Qwen3 model gets its thinking switched off through the chat template (`chat_template_kwargs.enable_thinking`,
  vLLM / SGLang) unless the request already says so, as the DashScope profile does for qwen3.8-flash with
  `enable_thinking`; a <think> block a server leaves in the content is stripped before the reply is read.
Every other request field is sent as the systems make it.
"""
import copy
import os
import re

LOCAL_HOSTS = ('127.0.0.1', 'localhost', '0.0.0.0', '[::1]', 'host.docker.internal')
_THINK = re.compile(r'^\s*<think>.*?</think>\s*', re.S)


def is_local(endpoint):
    host = endpoint.split('://', 1)[-1].split('/', 1)[0].rsplit(':', 1)[0]
    return host in LOCAL_HOSTS


def normalize(q, endpoint):
    if not is_local(endpoint):
        return q
    q = copy.deepcopy(q)
    if 'qwen3' in str(q.get('model', '')).lower():
        extra = q.setdefault('extra_body', {})
        kw = extra.setdefault('chat_template_kwargs', {})
        kw.setdefault('enable_thinking', False)
    return q


def strip_thinking(text):
    if not isinstance(text, str) or '<think>' not in text:
        return text
    out = _THINK.sub('', text, count=1)
    return '' if out == text and text.lstrip().startswith('<think>') else out


def install():
    from memcut.source_anchored_v2.completions import CachedCompletions
    from memcut.execution_optimization import receipts
    from memcut.efficient_execution import replay
    old = CachedCompletions.complete
    old_identity = receipts.request_identity

    def complete(self, q, **kw):
        res = old(self, normalize(q, self.endpoint), **kw)
        if is_local(self.endpoint) and isinstance(res, dict):
            res['text'] = strip_thinking(res.get('text'))
        return res

    def identity(cache, q):
        return old_identity(cache, normalize(q, cache.endpoint))
    CachedCompletions.complete = complete
    receipts.request_identity = identity
    replay.base_request_identity = identity
