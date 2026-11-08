"""Frozen provider normalization and received-only SessionRAG recovery.

No model request is retried after HTTP/network/unknown dispatch. Only a durable
received unusable QA/selector reply permits the next of three exact-body slots.
"""
from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import threading

from memcut.efficient_execution.recovery import validate as validate_schema
from memcut.execution_optimization.receipts import ReuseCache
from memcut.model_adapters import glm, qwen
from memcut.native_memory.contract import digest
from memcut.source_anchored_v2.completions import CachedCompletions
from memcut.source_anchored_v2.candidates import S2_PROMPT_PREFIX


RETRY_HEADER = 'X-MemCut-Session-Retry'
POLICY = {'version': 'session-rag-received-only-recovery-v1', 'maximum_total_attempts': 3,
          'same_body_budget': True, 'unknown_dispatch_retry': False,
          'http_error_retry': False, 'valid_wrong_answer_retry': False,
          'first_valid_response_wins': True}


class TransportBlocked(RuntimeError):
    """A request has no usable received response; never infer a zero score."""


class FormatExhausted(RuntimeError):
    """All three allowed received-format samples were unusable."""


class ReceiptError(RuntimeError):
    """Saved evidence does not match its frozen request/proof."""


def _read(path):
    return json.loads(Path(path).read_text())


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def provider_profile(model, endpoint, compatibility):
    """Return frozen settings, rejecting unofficial endpoints for known models."""
    if not isinstance(model, str) or not isinstance(endpoint, str):
        raise ValueError('Model and endpoint must be strings')
    endpoint = endpoint.rstrip('/')
    if endpoint == 'offline-fixture' or endpoint.startswith('offline://'):
        if compatibility not in {'openai', 'deepseek', 'mimo'}:
            raise ValueError('Unsupported fixture compatibility')
        return {'version': 'session-rag-provider-v1', 'model': model, 'endpoint': endpoint,
                'compatibility': compatibility, 'offline_fixture': True,
                'automatic_retries': 0, 'settings': {}}
    known = {
        'deepseek-flash': ({'https://api.deepseek.com', 'https://api.deepseek.com/v1'}, 'deepseek',
                           {'thinking': {'type': 'disabled'}}),
        'deepseek-v4-flash': ({'https://api.deepseek.com', 'https://api.deepseek.com/v1'}, 'deepseek',
                              {'thinking': {'type': 'disabled'}}),
        qwen.MODEL: ({qwen.ENDPOINT}, 'openai', {'enable_thinking': False, 'preserve_thinking': False}),
        glm.MODEL: ({glm.ENDPOINT}, 'openai', {'thinking': {'type': 'enabled'}, 'reasoning_effort': 'low'}),
        'mimo-v2.5': ({'https://api.xiaomimimo.com/v1'}, 'mimo', {'thinking': {'type': 'disabled'}}),
        'mimo-v2.6-flash': ({'https://api.xiaomimimo.com/v1'}, 'mimo', {'thinking': {'type': 'disabled'}}),
    }
    if model not in known:
        raise ValueError('Unsupported SessionRAG model; use an explicit supported profile')
    endpoints, expected, settings = known[model]
    if endpoint not in endpoints or compatibility != expected:
        raise ValueError('Model requires its frozen official endpoint and compatibility')
    return {'version': 'session-rag-provider-v1', 'model': model, 'endpoint': endpoint,
            'compatibility': compatibility, 'offline_fixture': False,
            'automatic_retries': 0, 'settings': copy.deepcopy(settings)}


def request_identity(plan, role, request):
    """Pure identity function used both before dispatch and by offline audits."""
    if role not in plan['models'] or role not in plan['providers']:
        raise ValueError('Unplanned model role')
    provider = plan['providers'][role]
    profile = provider_profile(plan['models'][role], provider['endpoint'], provider['compatibility'])
    sent = copy.deepcopy(request)
    if sent.get('model') != profile['model']:
        raise ValueError('Request model differs from frozen role model')
    if profile['model'] == glm.MODEL:
        sent = glm.normalize(sent)
    elif profile['model'] == qwen.MODEL:
        sent = qwen.normalize(sent)
    if profile['compatibility'] in {'deepseek', 'mimo'}:
        extra = sent.setdefault('extra_body', {})
        disabled = {'type': 'disabled'}
        if sent.pop('thinking', disabled) != disabled or extra.get('thinking', disabled) != disabled:
            raise ValueError('Request conflicts with disabled thinking profile')
        if any(k in sent or k in extra for k in ('reasoning_effort', 'thinking_budget')):
            raise ValueError('Reasoning parameters conflict with disabled thinking')
        extra['thinking'] = disabled
    if profile['compatibility'] == 'mimo' and 'max_tokens' in sent:
        if 'max_completion_tokens' in sent and sent['max_completion_tokens'] != sent['max_tokens']:
            raise ValueError('Conflicting output budgets')
        sent['max_completion_tokens'] = sent.pop('max_tokens')
    return {'endpoint': profile['endpoint'], 'compatibility': profile['compatibility'], 'request': sent}


def _base_request(request):
    if any(k.lower().startswith('x-memcut-') for k in request.get('extra_headers', {})):
        raise ValueError('Caller must supply the frozen request without execution retry headers')
    return request


def _attempt_request(request, attempt):
    sent = copy.deepcopy(request)
    if attempt:
        sent.setdefault('extra_headers', {})[RETRY_HEADER] = str(attempt)
    return sent


def _validate_prompt(role, request):
    messages = request.get('messages')
    if not isinstance(messages, list) or not messages:
        raise ValueError('Missing request messages')
    prompt = messages[0].get('content', '')
    prefix = {'answer': 'Answer the multiple-choice question', 'selector': S2_PROMPT_PREFIX}
    if role in prefix and not prompt.startswith(prefix[role]):
        raise ValueError('Unrecognized frozen QA/selector schema')
    # Validate local schema metadata before a dispatch: a broken prompt must
    # never be mistaken for three invalid model samples.
    if role == 'answer':
        content = messages[-1].get('content', '')
        if '\nOptions:\n' not in content:
            raise ValueError('QA request lacks displayed options')
        options = content.rsplit('\nOptions:\n', 1)[1].split('\n\n', 1)[0]
        labels = re.findall(r'(?m)^([a-z])\. ', options)
        if not labels or len(set(labels)) != len(labels):
            raise ValueError('Invalid displayed QA option labels')
    elif role == 'selector':
        payload = json.loads(messages[-1].get('content', ''))
        candidates = payload.get('candidates')
        if not isinstance(candidates, list) or any(not isinstance(c, dict) for c in candidates):
            raise ValueError('Selector request lacks candidate records')
        ids = [c.get('id') for c in candidates]
        if any(not isinstance(i, str) for i in ids) or len(set(ids)) != len(ids):
            raise ValueError('Selector candidate IDs must be unique strings')


def _classify(folder, identity, role, request, attempt):
    """Read an attempt; never parse corrupted local evidence as model failure."""
    rid = digest(identity)
    path = Path(folder) / 'requests' / rid
    if not (path / 'request.json').exists():
        if path.exists() and any(path.iterdir()):
            raise ReceiptError('Receipt directory exists without request identity')
        return None, None
    if _read(path / 'request.json') != identity or digest(identity) != path.name:
        raise ReceiptError('Saved request differs from normalized frozen body')
    row = {'attempt': attempt, 'request_id': rid, 'request_sha256': _sha(path / 'request.json')}
    if not (path / 'response.json').exists():
        row.update(classification='no_received_response', charge='unknown')
        if (path / 'error.json').exists():
            row['error_sha256'] = _sha(path / 'error.json')
            row['error_type'] = _read(path / 'error.json').get('type')
        return row, None
    try:
        raw = _read(path / 'response.json')
        choice = raw['choices'][0]
        text, finish = choice['message'].get('content'), choice.get('finish_reason')
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise ReceiptError('Malformed provider receipt envelope') from exc
    row.update(response_sha256=_sha(path / 'response.json'), finish_reason=finish)
    if finish not in {'stop', 'length'}:
        row['classification'] = 'unsupported_finish_reason'
        return row, None
    result = {'text': text, 'request_id': rid, 'finish_reason': finish,
              'raw': raw, 'usage': raw.get('usage')}
    try:
        if finish != 'stop' or not isinstance(text, str) or not text.strip():
            raise ValueError('Incomplete or empty model content')
        if role in {'answer', 'selector'}:
            result = validate_schema(request, result)
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        row.update(classification='received_invalid_format', invalid_reason=str(exc))
        return row, None
    row['classification'] = 'valid'
    return row, result


def _journal(origin, role, attempts, *, selected=None, exhausted=False, blocked=False):
    return {'policy': POLICY, 'origin': origin, 'role': role, 'attempts': attempts,
            'selected_request_id': selected, 'exhausted': exhausted, 'blocked': blocked}


def _verify_journal_prefix(folder, plan, role, request, origin, saved):
    """A restart may extend an unfinished proof but never rewrite its evidence."""
    limit = 3 if role in {'answer', 'selector'} else 1
    recorded = saved.get('attempts') if isinstance(saved, dict) else None
    if not isinstance(recorded, list) or not 1 <= len(recorded) <= limit:
        raise ReceiptError('Invalid recovery journal attempt count')
    rows, selected, blocked = [], None, False
    for n in range(len(recorded)):
        actual = request_identity(plan, role, _attempt_request(request, n))
        row, result = _classify(folder, actual, role, request, n)
        if row != recorded[n]:
            raise ReceiptError('Previously journaled receipt evidence changed')
        rows.append(row)
        if result is not None:
            selected = row['request_id']
        elif row['classification'] in {'no_received_response', 'unsupported_finish_reason'}:
            blocked = True
        if (selected or blocked) and n != len(recorded) - 1:
            raise ReceiptError('Journal continued after a valid or non-retryable attempt')
    expected = _journal(origin, role, rows, selected=selected, blocked=blocked,
                        exhausted=len(rows) == limit and not selected and not blocked)
    if saved != expected:
        raise ReceiptError('Recovery journal policy or selection changed')


def verify_receipt(folder, plan, role, request, request_id):
    """Verify identity, first-valid selection, and the complete durable proof.

    Returns the validated response (``text`` may contain numeric-score
    normalization), its untouched provider envelope in ``raw``, and the ID.
    This function never constructs a client or calls a transport.
    """
    _base_request(request)
    _validate_prompt(role, request)
    if not isinstance(request_id, str) or not re.fullmatch('[0-9a-f]{64}', request_id):
        raise ReceiptError('Invalid selected request ID')
    origin = digest(request_identity(plan, role, request))
    attempts = []
    for n in range(3 if role in {'answer', 'selector'} else 1):
        actual = request_identity(plan, role, _attempt_request(request, n))
        row, result = _classify(folder, actual, role, request, n)
        if row is None or row['classification'] in {'no_received_response', 'unsupported_finish_reason'}:
            raise ReceiptError('Selected response has a missing or non-retryable predecessor')
        attempts.append(row)
        if result is not None:
            if row['request_id'] != request_id:
                raise ReceiptError('Selected response is not the first valid attempt')
            expected = _journal(origin, role, attempts, selected=request_id)
            if _read(Path(folder) / 'SESSION_RETRY_JOURNAL' / (origin + '.json')) != expected:
                raise ReceiptError('Recovery journal differs from durable receipts')
            return {**result, 'cached': True}
    raise ReceiptError('Selected request exceeds the three-attempt format limit')


class RoleCache:
    def __init__(self, gateway, role):
        self.gateway, self.role = gateway, role
        self.folder = gateway.folder / 'requests'
        profile = gateway.profiles[role]
        self.endpoint, self.compatibility = profile['endpoint'], profile['compatibility']
        self.raw_cache = CachedCompletions(self.folder, self._create, endpoint=self.endpoint,
                                           compatibility=self.compatibility,
                                           before_dispatch=lambda: gateway.ensure_client(role))
        catalog = gateway.plan.get('receipt_catalog')
        self.delegate = ReuseCache(self.raw_cache, catalog) if catalog else self.raw_cache

    def _check_import(self, identity, request, attempt):
        """Import complete usable receipts only, with frozen safe file names."""
        catalog = self.gateway.plan.get('receipt_catalog')
        if not catalog:
            return
        if catalog.get('policy') != 'exact-complete-receipts-v1':
            raise ReceiptError('Unsupported external receipt catalog policy')
        key = digest(identity)
        entry = catalog['entries'].get(key)
        if entry is None:
            return
        files = entry.get('files', {})
        if not {'request.json', 'response.json'} <= set(files) or not set(files) <= {
                'request.json', 'response.json', 'usage.json'}:
            raise ReceiptError('Receipt import requires exact complete safe file names')
        source = Path(entry['directory'])
        if source.name != key or source.parent.name != 'requests':
            raise ReceiptError('Catalog receipt directory differs from request identity')
        for name, expected in files.items():
            if not isinstance(expected, str) or _sha(source / name) != expected:
                raise ReceiptError('Reusable receipt changed after catalog was frozen')
        row, result = _classify(source.parent.parent, identity, self.role, request, attempt)
        if row is None or result is None:
            raise ReceiptError('External receipt is not a complete valid response')

    @property
    def calls(self):
        return self.raw_cache.calls

    def _create(self, **sent):
        try:
            return self.gateway.create(self.role, sent)
        except Exception as exc:
            identity = {'endpoint': self.endpoint, 'compatibility': self.compatibility, 'request': sent}
            rid = digest(identity)
            status = getattr(exc, 'status_code', None)
            _write(self.folder / rid / 'TRANSPORT_FAILURE.json',
                   {'exception_type': type(exc).__name__, 'status_code': status,
                    'classification': 'http_error' if isinstance(status, int) else 'transport_or_unknown',
                    'automatic_retry': False, 'charge': 'unknown'})
            raise

    def complete(self, request, **kwargs):
        if kwargs:
            raise ValueError('SessionRAG does not allow length/validation bypass options')
        _base_request(request)
        _validate_prompt(self.role, request)
        identity = request_identity(self.gateway.plan, self.role, request)
        origin = digest(identity)
        locks = self.folder / '.session_request_locks'
        locks.mkdir(parents=True, exist_ok=True)
        attempts = []
        journal = self.gateway.folder / 'SESSION_RETRY_JOURNAL' / (origin + '.json')
        with (locks / origin).open('a+b') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if journal.exists():
                _verify_journal_prefix(self.gateway.folder, self.gateway.plan, self.role,
                                       request, origin, _read(journal))
            limit = 3 if self.role in {'answer', 'selector'} else 1
            for n in range(limit):
                actual = request_identity(self.gateway.plan, self.role, _attempt_request(request, n))
                row, result = _classify(self.gateway.folder, actual, self.role, request, n)
                was_cached = row is not None and (self.folder / row['request_id'] / 'response.json').exists()
                loaded_local = was_cached
                if row is None:
                    self._check_import(actual, request, n)
                    caught = None
                    try:
                        delivered = self.delegate.complete(actual['request'])
                        was_cached = bool(delivered.get('cached'))
                    except Exception as exc:
                        caught = exc
                    row, result = _classify(self.gateway.folder, actual, self.role, request, n)
                    if row is None:
                        if caught is not None:
                            raise caught  # local pre-dispatch cancellation, no unknown receipt
                        raise ReceiptError('Transport returned without a durable request')
                    if caught is not None and row['classification'] not in {
                            'received_invalid_format', 'no_received_response', 'unsupported_finish_reason'}:
                        raise caught  # programming failure after a valid receipt is not recoverable
                attempts.append(row)
                if row['classification'] in {'no_received_response', 'unsupported_finish_reason'}:
                    _write(journal, _journal(origin, self.role, attempts, blocked=True))
                    raise TransportBlocked('No automatic retry for HTTP/network/unknown response; request=' + row['request_id'])
                if result is not None:
                    _write(journal, _journal(origin, self.role, attempts, selected=row['request_id']))
                    if loaded_local:
                        self.raw_cache.calls.append({'request_id': row['request_id'], 'cached': True,
                                                     'usage': result['usage'], 'finish_reason': 'stop'})
                    return {**result, 'cached': was_cached}
                # Only this explicit received-invalid classification advances.
                _write(journal, _journal(origin, self.role, attempts, exhausted=n == limit - 1))
            raise FormatExhausted('Received invalid-format recovery exhausted; no score inferred')


class Gateway:
    def __init__(self, folder, plan, create_factory=None):
        self.folder, self.plan = Path(folder), copy.deepcopy(plan)
        self.create_factory = create_factory
        self.clients, self.caches = {}, {}
        self.lock = threading.RLock()
        self.profiles = {role: provider_profile(model, plan['providers'][role]['endpoint'],
                                               plan['providers'][role]['compatibility'])
                         for role, model in plan['models'].items()}

    def ensure_client(self, role):
        with self.lock:
            if self.create_factory is not None or role in self.clients:
                return
            if self.profiles[role]['offline_fixture']:
                raise RuntimeError('Offline fixture requires an injected transport')
            provider = self.plan['providers'][role]
            env_name = provider.get('key_env', 'MEMCUT_MODEL_API_KEY')
            key = os.environ.get(env_name)
            if not key:
                raise ValueError(env_name + ' is required; never store keys in the plan')
            from openai import OpenAI
            self.clients[role] = OpenAI(api_key=key, base_url=self.profiles[role]['endpoint'],
                                        max_retries=0, timeout=180)

    def create(self, role, request):
        if self.create_factory is not None:
            return self.create_factory(role, request)
        return self.clients[role].chat.completions.create(**request)

    def for_role(self, role):
        if role not in self.profiles:
            raise ValueError('Unplanned model role')
        with self.lock:
            if role not in self.caches:
                self.caches[role] = RoleCache(self, role)
            return self.caches[role]

    @property
    def calls(self):
        return [call for cache in self.caches.values() for call in cache.calls]

    def close(self):
        for client in self.clients.values():
            client.close()
