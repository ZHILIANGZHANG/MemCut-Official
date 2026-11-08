"""What the aligned entry does when the provider does not answer a paid request.

The frozen exact-request cache and its receipt files stay as they are (request.json, response.json,
usage.json, error.json); only the transport callable inside it is wrapped:
- HTTP 408/409/429 and 5xx, and a request that never reached the provider (refused connection, DNS): retried
  inside the one dispatch with a 2, 4, 8, 16, 32 s backoff. None of these is billed. Each retry is a line of
  requests/TRANSPORT_RETRIES.jsonl.
- A content refusal (HTTP 400/413/422 content inspection, an input too long): saved as the request's response,
  an empty reply with finish_reason "content_filter" and the provider's error. The frozen cache rejects it like
  any unusable reply: an answer readout becomes unavailable (answers.py), a formation or selector request stops
  the run, and a rerun meets the same refusal without sending it again.
- Any other 4xx (a key, a quota, a parameter), or a retry that keeps failing: the run stops; nothing was billed,
  so the next run archives the receipt (requests/.archived/, ARCHIVED.jsonl) and sends it again.
- A timeout or a connection dropped after sending: the outcome and its charge are unknown. The receipt stays
  pending and blocks the run until `python run.py aligned resolve --run DIR --retry` archives it.
"""
import json
import os
import re
import time
from pathlib import Path

BACKOFF = (2.0, 4.0, 8.0, 16.0, 32.0)
RETRY_STATUS = (408, 409, 429)            # and every 5xx
REFUSAL = re.compile(r'data_inspection_failed|inappropriate|content[_ ]?filter|content[_ ]?policy|'
                     r'range of input length|input length|maximum context|context[_ ]length|too long|too large', re.I)
OUT_OF_CREDIT = re.compile(r'insufficient[_ ]?(balance|quota)|exceeded your current quota|arrearage|'
                           r'allocation[_ .]?quota|free (allocated )?quota|欠费|余额不足|额度不足', re.I)
UNSENT_CAUSES = ('ConnectError', 'ConnectTimeout')     # httpx: the request never left this machine


class NotBilled(RuntimeError):
    """The provider answered with an error, or never got the request: nothing to pay, safe to send again."""


def _body(exc):
    body = getattr(exc, 'body', None)
    if body is None:
        response = getattr(exc, 'response', None)
        try:
            body = response.text if response is not None else ''
        except Exception:  # noqa: BLE001 - only the provider's reason is lost
            body = ''
    return body if isinstance(body, str) else json.dumps(body, ensure_ascii=False, default=str)


def outcome(exc):
    """'refused' | 'retry' | 'rejected' | 'unknown' for one failed provider call (an OpenAI SDK exception)."""
    status = getattr(exc, 'status_code', None)
    if isinstance(status, int):
        body = _body(exc)
        if status in (400, 413, 422) and REFUSAL.search(body):
            return 'refused'
        if (status in RETRY_STATUS or status >= 500) and not OUT_OF_CREDIT.search(body):
            return 'retry'
        return 'rejected'
    if type(exc).__name__ == 'APIConnectionError' and type(exc.__cause__).__name__ in UNSENT_CAUSES:
        return 'retry'
    return 'unknown'


def refusal_response(exc):
    return {'id': 'memcut-refusal', 'object': 'chat.completion', 'usage': None,
            'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': ''},
                         'finish_reason': 'content_filter'}],
            'memcut_refusal': {'status': getattr(exc, 'status_code', None), 'body': _body(exc)[:4000],
                               'billed': False}}


def policy(create, folder, *, sleep=time.sleep):
    """`create(**request)` with the retries and refusals above."""
    folder = Path(folder)

    def send(**request):
        for attempt in range(len(BACKOFF) + 1):
            try:
                return create(**request)
            except Exception as exc:  # noqa: BLE001 - classified below; anything else is re-raised as it was
                kind = outcome(exc)
                if kind == 'refused':
                    return refusal_response(exc)
                if kind == 'retry' and attempt < len(BACKOFF):
                    folder.mkdir(parents=True, exist_ok=True)
                    with (folder / 'TRANSPORT_RETRIES.jsonl').open('a') as log:
                        log.write(json.dumps({'time': time.time(), 'attempt': attempt + 1,
                                              'error': type(exc).__name__,
                                              'status': getattr(exc, 'status_code', None)}) + '\n')
                    sleep(BACKOFF[attempt])
                    continue
                if kind in ('retry', 'rejected'):
                    raise NotBilled(f'{type(exc).__name__}: {str(exc)[:500]}') from exc
                raise
    return send


def innermost(cache):
    seen = set()
    while hasattr(cache, 'delegate') and id(cache) not in seen:
        seen.add(id(cache))
        cache = cache.delegate
    return cache


class TransportGateway:
    """The stage gateway (role caches) with the policy installed on each cache's transport, once."""

    def __init__(self, inner, *, sleep=time.sleep):
        self.inner, self.sleep = inner, sleep

    def for_role(self, role):
        cache = self.inner.for_role(role)
        base = innermost(cache)
        if not getattr(base, '_aligned_transport', False) and hasattr(base, 'create'):
            base.create = policy(base.create, base.folder, sleep=self.sleep)
            base._aligned_transport = True
        return cache

    @property
    def calls(self):
        return self.inner.calls

    def close(self):
        self.inner.close()

    def __getattr__(self, name):
        return getattr(self.inner, name)


def pending(folder):
    """Receipts sent without a usable response (request.json, no response.json), with what stopped them."""
    rows = []
    for request in sorted((Path(folder) / 'requests').glob('*/request.json')):
        if request.with_name('response.json').exists():
            continue
        error = request.with_name('error.json')
        rows.append({'request_id': request.parent.name,
                     'error': json.loads(error.read_text()).get('type') if error.exists() else None})
    return rows


def archive(folder, request_ids, reason):
    """Move pending receipts out of the cache (requests/.archived/) so the next run sends them again."""
    root = Path(folder) / 'requests'
    stamp = time.time_ns()
    moved = []
    for rid in request_ids:
        source = root / rid
        if not (source / 'request.json').exists() or (source / 'response.json').exists():
            continue
        target = root / '.archived' / f'{rid}-{stamp}'
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, target)
        with (root / 'ARCHIVED.jsonl').open('a') as log:
            log.write(json.dumps({'request_id': rid, 'archived': str(target), 'reason': reason,
                                  'time': time.time()}) + '\n')
        moved.append(rid)
    return moved


def archive_not_billed(folder):
    """Pending receipts whose stop was a provider error (NotBilled): nothing was billed, send them again."""
    return archive(folder, [r['request_id'] for r in pending(folder) if r['error'] == 'NotBilled'],
                   'provider error or unsent request, nothing billed')
