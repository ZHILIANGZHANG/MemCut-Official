"""Exact request reuse and explicit failure, without native retry/truncation fallbacks."""
import copy
import json
import time
from pathlib import Path
from types import SimpleNamespace

from memcut.native_memory.contract import digest
from memcut.native_memory.persistence import write_json


def native_summary_request(module, prompt, model):
    """Ask the upstream request builder to render its own messages and settings.

    The capture transport cannot send requests. Only model and protocol temp0
    override upstream settings; its max_tokens400/penalties/stops are retained.
    """
    captured = []
    def capture(**kwargs):
        captured.append(copy.deepcopy(kwargs))
        return {'choices': [{'message': {'content': 'captured'}}]}
    original = module.openai
    module.openai = SimpleNamespace(api_key=None, ChatCompletion=SimpleNamespace(create=capture))
    try:
        config = {**module.chatgpt_config, 'model': model, 'temperature': 0}
        client = module.LLMClientSimple(config)
        client.generate_text_simple(prompt, 1, language='en')
    finally:
        module.openai = original
    if len(captured) != 1:
        raise RuntimeError('Native request capture was not exactly one request')
    return captured[0]


class CachedCompletions:
    def __init__(self, folder, create, *, endpoint, compatibility='openai', before_dispatch=None):
        if compatibility not in {'openai', 'deepseek', 'mimo'}:
            raise ValueError('Unknown provider compatibility')
        self.folder, self.create = Path(folder), create
        self.endpoint, self.compatibility = endpoint, compatibility
        self.before_dispatch = before_dispatch
        self.calls = []

    def complete(self, request, *, allow_length=False):
        sent = copy.deepcopy(request)
        if self.compatibility in {'deepseek', 'mimo'}:
            sent['extra_body'] = {**sent.get('extra_body', {}), 'thinking': {'type': 'disabled'}}
        if self.compatibility == 'mimo' and 'max_tokens' in sent:
            sent['max_completion_tokens'] = sent.pop('max_tokens')
        identity = {'endpoint': self.endpoint, 'compatibility': self.compatibility, 'request': sent}
        key = digest(identity)
        path = self.folder / key
        response_file = path / 'response.json'
        request_file = path / 'request.json'
        if response_file.exists():
            if json.loads(request_file.read_text()) != identity:
                raise ValueError('Cached request identity changed')
            raw = json.loads(response_file.read_text())
            cached = True
        else:
            if request_file.exists():
                raise RuntimeError('Unresolved model dispatch; inspect request before any retry')
            if self.before_dispatch is not None:
                self.before_dispatch()
            write_json(request_file, identity)
            started = time.monotonic()
            try:
                raw = self.create(**sent)
                if hasattr(raw, 'model_dump'):
                    raw = raw.model_dump(mode='json')
                write_json(response_file, raw)
                write_json(path / 'usage.json', {'usage': raw.get('usage'), 'seconds': time.monotonic()-started})
            except BaseException as exc:
                write_json(path / 'error.json', {'type': type(exc).__name__})
                raise
            cached = False
        choice = raw['choices'][0]
        text = choice['message'].get('content')
        finish = choice.get('finish_reason')
        self.calls.append({'request_id': key, 'cached': cached, 'usage': raw.get('usage'), 'finish_reason': finish})
        # Native MemoryBank stores nonempty summaries at its 400-token cap.
        # This explicit opt-in applies to memory generation only; QA and S2
        # still require a complete response, even when reusing the same cache.
        if finish not in ({'stop', 'length'} if allow_length else {'stop'}) or not isinstance(text, str) or not text.strip():
            raise RuntimeError('Incomplete/empty model response; no deletion success or zero score inferred')
        return {'text': text, 'request_id': key, 'finish_reason': finish, 'cached': cached}


def parse_json(text):
    text = text.strip()
    if text.startswith('```json\n') and text.endswith('```'):
        text = text[8:-3].strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError('Expected a JSON object')
    return value
