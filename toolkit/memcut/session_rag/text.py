"""Lossless source text segmentation and separately pinned coverage counting.

Embedding chunks are an index detail, never the stored or returned unit. Every
chunk is bounded *including* tokenizer special tokens. No token decoding is used
to reconstruct source text: overlong turns retain exact character spans.
"""
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import tempfile


COVERAGE_ENCODING = 'cl100k_base'
COVERAGE_VERSION = '0.12.0'
COVERAGE_URL = 'https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken'
COVERAGE_SHA256 = '223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7'


def token_count(text, tokenizer):
    """Count the exact encoder input, with its native special tokens."""
    if not isinstance(text, str):
        raise ValueError('Embedding input must be text')
    return len(tokenizer.encode(text, add_special_tokens=True))


def load_tokenizer(settings):
    """Resolve only the same cached pinned model used by LocalEmbeddings."""
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from memcut.raw_turn_rag.embedding import resolved_revision
    path = snapshot_download(settings['model'], revision=resolved_revision(settings), local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)
    # Counting must not implicitly truncate or warn at a model default; limits
    # are explicitly checked before the actual encoder receives each string.
    tokenizer.model_max_length = 1_000_000_000
    return tokenizer


def split_turn(text, tokenizer, max_length):
    """Return lossless [start,end) spans of one turn within the encoder budget.

    Long turns split at tokenizer offsets, without overlap. Whitespace and
    punctuation between tokens are retained. Every candidate is re-tokenized:
    a substring can acquire a different wordpiece split at its left boundary.
    """
    if type(max_length) is not int or max_length <= token_count('', tokenizer):
        raise ValueError('Embedding limit must leave room for content tokens')
    if token_count(text, tokenizer) <= max_length:
        return [(0, len(text))]
    try:
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = encoded['offset_mapping']
    except (TypeError, KeyError, NotImplementedError) as exc:
        raise ValueError('Overlong turns require a tokenizer with exact character offsets') from exc
    boundaries = sorted({int(end) for start, end in offsets if 0 < end <= len(text)})
    if not boundaries:
        raise ValueError('Tokenizer gave no boundaries for an overlong turn')
    boundaries = sorted(set(boundaries + [len(text)]))
    spans, start = [], 0
    while start < len(text):
        choices = [end for end in boundaries if end > start]
        # A prefix with N tokens may yield >N tokens after slicing. Therefore
        # test exact strings and retain the longest fitting token boundary.
        end = None
        for candidate in choices:
            if token_count(text[start:candidate], tokenizer) > max_length:
                break
            end = candidate
        if end is None:
            raise ValueError('A tokenizer unit cannot fit the encoder limit; no truncation allowed')
        spans.append((start, end))
        start = end
    if ''.join(text[a:b] for a, b in spans) != text:
        raise ValueError('Lost characters during turn segmentation')
    return spans


def chunk_session(turn_ids, turns, tokenizer, max_length):
    """Greedy nonoverlapping complete-turn chunks; split only long single turns."""
    chunks, pending = [], []

    def emit(spans):
        text = '\n'.join(turns[s['turn_id']]['text'][s['start']:s['end']] for s in spans)
        count = token_count(text, tokenizer)
        if count > max_length:
            raise ValueError('Embedding chunk exceeds max_seq_length')
        chunks.append({'text': text, 'turn_ids': list(dict.fromkeys(s['turn_id'] for s in spans)),
                       'spans': spans, 'token_count': count})

    for tid in turn_ids:
        text = turns[tid]['text']
        parts = split_turn(text, tokenizer, max_length)
        if len(parts) > 1:
            if pending:
                emit(pending)
                pending = []
            for start, end in parts:
                emit([{'turn_id': tid, 'start': start, 'end': end}])
            continue
        span = {'turn_id': tid, 'start': 0, 'end': len(text)}
        trial = pending + [span]
        joined = '\n'.join(turns[s['turn_id']]['text'] for s in trial)
        if pending and token_count(joined, tokenizer) > max_length:
            emit(pending)
            pending = []
        pending.append(span)
    if pending:
        emit(pending)
    return chunks


def coverage_identity():
    return {'encoding': COVERAGE_ENCODING, 'tiktoken_version': COVERAGE_VERSION,
            'encoding_asset_sha256': COVERAGE_SHA256, 'special_tokens': 'treated as ordinary text',
            'usage': 'diagnostic token count; not provider billing usage'}


@lru_cache(maxsize=1)
def _coverage_encoder():
    import tiktoken
    if tiktoken.__version__ != COVERAGE_VERSION:
        raise ValueError('Coverage tokenizer package differs from frozen version')
    cache = Path(os.environ.get('TIKTOKEN_CACHE_DIR',
                 os.environ.get('DATA_GYM_CACHE_DIR', str(Path(tempfile.gettempdir())/'data-gym-cache'))))
    asset = cache / hashlib.sha1(COVERAGE_URL.encode()).hexdigest()
    if not asset.is_file() or hashlib.sha256(asset.read_bytes()).hexdigest() != COVERAGE_SHA256:
        raise ValueError('Pinned cl100k_base cache is absent or changed; install it before the offline run')
    # The verified cache entry ensures this call never needs a remote fetch.
    return tiktoken.get_encoding(COVERAGE_ENCODING)


def coverage_tokens(text):
    return len(_coverage_encoder().encode(text, disallowed_special=()))
