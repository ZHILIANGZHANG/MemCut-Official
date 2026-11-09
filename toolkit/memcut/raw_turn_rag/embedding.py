"""Pinned local embeddings with durable text-keyed reuse, never an API client."""
import json
from pathlib import Path
import os
import tempfile

import numpy as np

from memcut.native_memory.contract import digest

REGISTRY = Path(__file__).resolve().parents[2] / 'configs/embeddings.json'
DEFAULT_EMBEDDING = 'e5-small-v2'      # all-MiniLM-L6-v2 is the equal alternative


def embedding_choices():
    return sorted(json.loads(REGISTRY.read_text()))


def embedding_settings(name):
    """The frozen settings of one registered encoder, as a PLAN records them."""
    entries = json.loads(REGISTRY.read_text())
    if name not in entries:
        raise ValueError(f'Unknown embedding {name!r}; choose one of {sorted(entries)}')
    return {'name': name, **entries[name]}


def with_prefix(settings, texts, kind):
    """The encoder's own input convention: e5 needs 'query: ' / 'passage: '; MiniLM takes plain text."""
    prefix = settings.get(f'{kind}_prefix', '')
    return [prefix + t for t in texts]


def resolved_revision(settings):
    """The pinned commit, or for an unpinned entry the commit in the local Hugging Face cache."""
    if settings.get('revision'):
        return settings['revision']
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(settings['model'], local_files_only=True)).name


def validate_vectors(value, count, dimensions):
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (count, dimensions) or not np.isfinite(array).all():
        raise ValueError('Invalid embedding shape or non-finite vector')
    if count and not np.allclose(np.linalg.norm(array, axis=1), 1, atol=2e-5):
        raise ValueError('Require nonzero normalized embeddings')
    return array


class LocalEmbeddings:
    def __init__(self, settings, cache_dir):
        self.settings = settings
        self.cache_dir = Path(cache_dir)
        self.model = None
        self.revision = None
        self.computed = self.cache_hits = 0

    def identity(self):
        """Settings with the actual model commit: cached vectors never mix two revisions of one encoder."""
        if self.revision is None:
            self.revision = resolved_revision(self.settings)
        return {**self.settings, 'revision': self.revision}

    def load(self):
        if self.model is None:
            from huggingface_hub import snapshot_download
            from sentence_transformers import SentenceTransformer
            path = snapshot_download(self.settings['model'], revision=self.identity()['revision'],
                                     local_files_only=True)
            self.model = SentenceTransformer(path, device=self.settings['device'], local_files_only=True)
            self.model.max_seq_length = self.settings['max_seq_length']
            self.model.eval()
        return self.model

    def encode(self, texts):
        texts = list(texts)
        if any(not isinstance(t, str) for t in texts):
            raise ValueError('Embedding input must be text')
        dimension = self.settings['dimensions']
        if not texts:
            return np.empty((0, dimension), dtype=np.float32)
        values, missing = {}, {}
        keys = [digest({'embedding': self.identity(), 'text': t}) for t in texts]
        for key, text in zip(keys, texts):
            if key in values or key in missing:
                continue
            path = self.cache_dir / (key + '.npy')
            if path.exists():
                values[key] = validate_vectors(np.load(path, allow_pickle=False).reshape(1, -1), 1, dimension)[0]
                self.cache_hits += 1
            else:
                missing[key] = text
        items = list(missing.items())
        size = self.settings['batch_size']
        for start in range(0, len(items), size):
            batch = items[start:start + size]
            encoded = self.load().encode([text for _, text in batch], batch_size=size,
                normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
            encoded = validate_vectors(encoded, len(batch), dimension)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            for (key, _), vector in zip(batch, encoded):
                # Commit each completed batch: interrupted formation reuses it.
                with tempfile.NamedTemporaryFile(dir=self.cache_dir, suffix='.pending', delete=False) as handle:
                    temporary = Path(handle.name)
                    np.save(handle, vector, allow_pickle=False)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, self.cache_dir / (key + '.npy'))
                values[key] = vector
                self.computed += 1
        return np.stack([values[key] for key in keys])
