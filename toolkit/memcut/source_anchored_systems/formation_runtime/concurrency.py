"""Bounded requests, exact-request single flight, and ordered result delivery."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from collections import deque
import copy
import fcntl
from pathlib import Path
import threading
import time

from memcut.native_memory.contract import digest


class Timings:
    def __init__(self):
        self.values = {}
        self.lock = threading.Lock()

    @contextmanager
    def measure(self, name):
        start = time.perf_counter()
        try:
            yield
        finally:
            with self.lock:
                row = self.values.setdefault(name, {'calls': 0, 'seconds': 0.0})
                row['calls'] += 1
                row['seconds'] += time.perf_counter() - start

    def report(self):
        with self.lock:
            return copy.deepcopy(self.values)


class ConcurrentCompletions:
    """Keep the original cache/receipts; serialize only identical requests.

    The dispatch guard (including lazy client initialization) runs under a
    short lock. Network waits for unrelated keys never hold that lock.
    Failed requests are still unresolved in the original cache, never retried.
    """
    def __init__(self, cache, max_workers=4):
        self.cache = cache
        self.slots = threading.BoundedSemaphore(max_workers)
        self.failed = threading.Event()
        self.lock = threading.Lock()
        guard = getattr(cache, '_formation_original_guard', getattr(cache, 'before_dispatch', None))
        if hasattr(cache, 'before_dispatch'):
            cache._formation_original_guard = guard
            def guarded():
                with self.lock:
                    if self.failed.is_set():
                        raise RuntimeError('Formation dispatch stopped after a worker failure')
                    if guard:
                        guard()
            cache.before_dispatch = guarded

    def complete(self, request, **kwargs):
        if self.failed.is_set():
            raise RuntimeError('Formation dispatch stopped after a worker failure')
        # Same normalization as the underlying cache, including provider mode.
        sent = copy.deepcopy(request)
        compatibility = self.cache.compatibility
        if compatibility in {'deepseek', 'mimo'}:
            sent['extra_body'] = {**sent.get('extra_body', {}), 'thinking': {'type': 'disabled'}}
        if compatibility == 'mimo' and 'max_tokens' in sent:
            sent['max_completion_tokens'] = sent.pop('max_tokens')
        key = digest({'endpoint': self.cache.endpoint, 'compatibility': compatibility, 'request': sent})
        folder = Path(self.cache.folder) / '.singleflight'
        folder.mkdir(parents=True, exist_ok=True)
        try:
            with (folder / key).open('a+b') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                if self.failed.is_set():
                    raise RuntimeError('Formation dispatch stopped after a worker failure')
                with self.slots:
                    if self.failed.is_set():
                        raise RuntimeError('Formation dispatch stopped after a worker failure')
                    return self.cache.complete(request, **kwargs)
        except BaseException:
            self.failed.set()
            raise


def ordered_map(fn, items, workers):
    """At most workers jobs in flight; commit in input order, cancel on error."""
    iterator = iter(items)
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='memcut-formation')
    pending = deque()
    try:
        for _ in range(workers):
            item = next(iterator, None)
            if item is None:
                break
            pending.append(pool.submit(fn, item))
        while pending:
            yield pending.popleft().result()
            item = next(iterator, None)
            if item is not None:
                pending.append(pool.submit(fn, item))
    finally:
        for future in pending:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
