"""Explicit, hash-bound reuse of complete requests; never repair unknown dispatches."""
import copy
import fcntl
from pathlib import Path
import shutil
import tempfile

from memcut.native_memory.contract import digest
from memcut.native_memory.persistence import write_json
from memcut.source_anchored_v2.runner import read, sha


def request_identity(cache, request):
    sent = copy.deepcopy(request)
    from memcut.model_adapters.glm import normalize
    # Only GLMCompletions applies GLM normalization. An ordinary OpenAI cache
    # may legitimately contain a different request for the same model name.
    from memcut.model_adapters.glm import GLMCompletions
    delegate = getattr(cache, 'delegate', cache)
    if isinstance(delegate, GLMCompletions):
        sent = normalize(sent)
    if cache.compatibility in {'deepseek', 'mimo'}:
        sent['extra_body'] = {**sent.get('extra_body', {}), 'thinking': {'type': 'disabled'}}
    if cache.compatibility == 'mimo' and 'max_tokens' in sent:
        sent['max_completion_tokens'] = sent.pop('max_tokens')
    return {'endpoint': cache.endpoint, 'compatibility': cache.compatibility, 'request': sent}


def catalog(sources):
    """Freeze existing receipts only. Conflicting samples cannot be cherry-picked."""
    entries = {}
    for source in dict.fromkeys(str(Path(s).resolve()) for s in sources):
        root = Path(source) / 'requests'
        if not root.is_dir():
            raise ValueError(f'No request directory in {source}')
        for response in sorted(root.glob('*/response.json')):
            request = response.with_name('request.json')
            identity = read(request)
            key = digest(identity)
            if key != response.parent.name:
                raise ValueError('Receipt directory does not match actual request identity')
            item = {'directory': str(response.parent), 'files': {
                name: sha(response.with_name(name)) for name in ('request.json', 'response.json', 'usage.json')
                if response.with_name(name).exists()}}
            previous = entries.get(key)
            if previous and previous['files']['response.json'] != item['files']['response.json']:
                raise ValueError('Different received samples for the same request; select one explicit source')
            entries.setdefault(key, item)
    return {'policy': 'exact-complete-receipts-v1', 'entries': entries}


class ReuseCache:
    def __init__(self, delegate, receipt_catalog):
        self.delegate, self.receipt_catalog = delegate, receipt_catalog

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    @property
    def before_dispatch(self):
        return self.delegate.before_dispatch

    @before_dispatch.setter
    def before_dispatch(self, value):
        self.delegate.before_dispatch = value

    def complete(self, request, **kwargs):
        identity = request_identity(self.delegate, request)
        key = digest(identity)
        folder = Path(self.folder)
        locks = folder / '.receipt_reuse_locks'
        locks.mkdir(parents=True, exist_ok=True)
        with (locks / key).open('a+b') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            target = folder / key
            # A local request without a response remains unresolved, even if
            # another run has a matching receipt. Do not hide a possible charge.
            if not target.exists() and key in self.receipt_catalog['entries']:
                entry = self.receipt_catalog['entries'][key]
                source = Path(entry['directory'])
                if read(source / 'request.json') != identity:
                    raise ValueError('Reusable request identity changed')
                for name, expected in entry['files'].items():
                    if sha(source / name) != expected:
                        raise ValueError('Reusable receipt changed after the catalog was frozen')
                temporary = Path(tempfile.mkdtemp(prefix='.import-', dir=folder))
                try:
                    for name, expected in entry['files'].items():
                        shutil.copyfile(source / name, temporary / name)
                        if sha(temporary / name) != expected:
                            raise ValueError('Receipt changed during import')
                    write_json(temporary / 'REUSED_RECEIPT.json', entry)
                    temporary.rename(target)
                finally:
                    if temporary.exists():
                        shutil.rmtree(temporary)
            # Original parser/finish checks still apply, including allow_length
            # only for native MemoryBank formation. No retry or fallback here.
            return self.delegate.complete(request, **kwargs)


class ReuseGateway:
    def __init__(self, delegate, receipt_catalog, *, cached_only=False):
        self.delegate, self.receipt_catalog = delegate, receipt_catalog
        self.cached_only, self.caches = cached_only, {}

    def for_role(self, role):
        if role not in self.caches:
            cache = self.delegate.for_role(role)
            if self.cached_only:
                def forbid():
                    raise RuntimeError('Offline replay has no matching receipt; API dispatch forbidden')
                cache.before_dispatch = forbid
            self.caches[role] = ReuseCache(cache, self.receipt_catalog)
        return self.caches[role]

    @property
    def calls(self):
        return self.delegate.calls

    def close(self):
        self.delegate.close()


def usage(folder):
    """Imported receipts stay auditable but are not newly paid requests."""
    totals = {kind: dict(requests=0, prompt_tokens=0, completion_tokens=0,
                        cached_prompt_tokens=0, missing_usage=0)
              for kind in ('new', 'imported')}
    unresolved = []
    for request in (Path(folder) / 'requests').glob('*/request.json'):
        response = request.with_name('response.json')
        if not response.exists():
            unresolved.append(request.parent.name)
            continue
        kind = 'imported' if request.with_name('REUSED_RECEIPT.json').exists() else 'new'
        row = totals[kind]
        value = read(response).get('usage') or {}
        row['requests'] += 1
        if not all(type(value.get(k)) is int for k in ('prompt_tokens', 'completion_tokens')):
            row['missing_usage'] += 1
        for name in ('prompt_tokens', 'completion_tokens'):
            row[name] += value.get(name) or 0
        row['cached_prompt_tokens'] += (value.get('prompt_tokens_details') or {}).get('cached_tokens') or 0
    return {**totals, 'unresolved_request_ids': sorted(unresolved),
            'note': 'New means dispatched in this run across all resumptions; reasoning is included in completion.'}
