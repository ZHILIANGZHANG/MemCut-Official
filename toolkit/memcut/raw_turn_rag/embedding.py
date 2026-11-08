"""Pinned local embeddings with durable text-keyed reuse, never an API client."""
from pathlib import Path
import os
import tempfile

import numpy as np

from memcut.native_memory.contract import digest


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
        self.computed = self.cache_hits = 0

    def load(self):
        if self.model is None:
            from huggingface_hub import snapshot_download
            from sentence_transformers import SentenceTransformer
            path = snapshot_download(self.settings['model'], revision=self.settings['revision'],
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
        keys = [digest({'embedding': self.settings, 'text': t}) for t in texts]
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
