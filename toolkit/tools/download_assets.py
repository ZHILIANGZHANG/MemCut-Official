"""Explicit download of pinned public assets; never loads a model or calls an LLM."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from memcut.session_rag.text import COVERAGE_URL, COVERAGE_SHA256


def coverage(cache):
    cache = Path(cache)
    path = cache / hashlib.sha1(COVERAGE_URL.encode()).hexdigest()
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest() != COVERAGE_SHA256:
            raise ValueError('Existing tokenizer asset differs; not overwriting')
        return path
    import ssl
    context = ssl.create_default_context()   # the system's CAs, plus certifi's (python.org's Python on macOS has none)
    try:
        import certifi
        context.load_verify_locations(certifi.where())
    except ImportError:
        pass
    with urllib.request.urlopen(COVERAGE_URL, timeout=60, context=context) as response:
        raw = response.read()
    if hashlib.sha256(raw).hexdigest() != COVERAGE_SHA256:
        raise ValueError('Downloaded tokenizer hash differs')
    cache.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('asset', choices=['coverage', 'encoder', 'lightmem-compressor'])
    parser.add_argument('--cache', type=Path,
                        default=Path(os.environ.get('TIKTOKEN_CACHE_DIR', ROOT/'runtime/tiktoken')),
                        help='Coverage tokenizer cache only; models use standard HF_HOME')
    parser.add_argument('--embedding', choices=['e5-small-v2', 'all-MiniLM-L6-v2'],
                        help='encoder only: download this retrieval encoder (default: both)')
    args = parser.parse_args()
    if args.asset == 'coverage':
        print(coverage(args.cache))
    else:
        from huggingface_hub import snapshot_download
        if args.asset == 'encoder':      # every registered retrieval encoder, or the one --embedding names
            registry = json.loads((ROOT/'configs/embeddings.json').read_text())
            for name in ([args.embedding] if args.embedding else sorted(registry)):
                entry = registry[name]
                path = snapshot_download(entry['model'], revision=entry['revision'])
                print(json.dumps({'embedding': name, 'model': entry['model'], 'revision': Path(path).name,
                                  'path': path}))
            return
        print(snapshot_download('microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank',
                                revision='5f0c82792b7ea14c6484e015b6a072009496b7f2'))


if __name__ == '__main__':
    main()
