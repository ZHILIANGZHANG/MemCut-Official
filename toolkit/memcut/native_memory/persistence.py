"""Closed-bank snapshots and durable mutation journals; no model requests."""
import hashlib
import json
import os
from pathlib import Path
import shutil
from contextlib import contextmanager


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def file_hashes(folder):
    folder = Path(folder).resolve()
    hashes = {}
    for path in sorted(folder.rglob("*")):
        if path.is_symlink():
            raise ValueError("Bank snapshots must not contain symlinks")
        if path.is_file() and path.name not in {"FROZEN.json", ".bank.lock"}:
            h = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(block)
            hashes[str(path.relative_to(folder))] = h.hexdigest()
    return hashes


@contextmanager
def bank_lock(folder, *, writable=True):
    import fcntl
    folder = Path(folder)
    if writable and (folder / "FROZEN.json").exists():
        raise ValueError("Frozen Full state is read-only; fork it first")
    with (folder / ".bank.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def freeze_file_bank(folder, close):
    """Caller holds bank_lock and supplies actual native close, not a live DB copy."""
    folder = Path(folder).resolve()
    if (folder / "INFLIGHT.json").exists():
        raise RuntimeError("Partial mutation requires audit before freezing")
    close()
    manifest = {"format": "memcut-native-file-snapshot-v1", "files": file_hashes(folder),
                "scope": "entire closed bank including derived state and native indexes"}
    write_json(folder / "FROZEN.json", manifest)
    return manifest


def fork_file_bank(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Snapshot and fork paths must be disjoint")
    if destination.exists():
        raise FileExistsError("Each claim/condition gets a fresh destination")
    manifest = json.loads((source / "FROZEN.json").read_text())
    expected = manifest["files"]
    if file_hashes(source) != expected:
        raise ValueError("Frozen bank changed; cannot reproduce an independent Full branch")
    # copy2 gives independent inodes, unlike hard links; Qdrant and SQLite included.
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns(".bank.lock", "FROZEN.json"))
    if file_hashes(destination) != expected or file_hashes(source) != expected:
        raise RuntimeError("Snapshot copy verification failed")
    write_json(destination / "FORK.json", {"source": str(source),
               "source_manifest_sha256": hashlib.sha256((source / "FROZEN.json").read_bytes()).hexdigest()})
    return destination


class OperationJournal:
    """Failures retain INFLIGHT; callers inspect partial native state before resume."""
    def __init__(self, folder):
        self.folder = Path(folder)
        self.inflight = self.folder / "INFLIGHT.json"

    async def run(self, ident, request, operation):
        import re
        if not isinstance(ident, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", ident):
            raise ValueError("Operation ID must be a safe single filename component")
        if self.inflight.exists():
            raise RuntimeError("An incomplete operation needs state audit; do not replay blindly")
        receipt = self.folder / "operations" / (ident + ".json")
        if receipt.exists():
            old = json.loads(receipt.read_text())
            if old["request"] != request:
                raise ValueError("Operation identity reused with different input")
            return old["result"]
        write_json(self.inflight, {"id": ident, "request": request})
        result = await operation()
        write_json(receipt, {"id": ident, "request": request, "result": result})
        self.inflight.unlink()
        return result
