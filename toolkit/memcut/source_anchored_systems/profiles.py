"""Pinned lightweight native configurations; no model/runtime imports."""
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def profile(system):
    value = json.loads((ROOT / 'configs/source_anchored_systems_v1.json').read_text())[system]
    upstream = (ROOT / value['repository']).resolve()
    commit = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != value['commit']:
        raise ValueError(f'{system} upstream commit differs')
    for name, expected in value['files'].items():
        if hashlib.sha256((upstream / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'{system} upstream file differs: {name}')
    return value


def nemori_modules():
    upstream = str((ROOT / profile('nemori')['repository']).resolve())
    if 'src' in sys.modules and not str(getattr(sys.modules['src'], '__file__', '')).startswith(upstream + '/'):
        raise ValueError('Another src package is loaded; use the dedicated native runtime')
    if upstream not in sys.path:
        sys.path.insert(0, upstream)
    return importlib.import_module('src')


def load_assets(system=None):
    from huggingface_hub import snapshot_download
    result = {
        'embedder': snapshot_download('sentence-transformers/all-MiniLM-L6-v2',
            revision='1110a243fdf4706b3f48f1d95db1a4f5529b4d41', local_files_only=True)}
    if system in {None, 'lightmem'}:
        result['compressor'] = snapshot_download('microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank',
            revision='5f0c82792b7ea14c6484e015b6a072009496b7f2', local_files_only=True)
    return result


def doctor(system):
    """Inspect native source pins, local assets and importability; no generation."""
    import importlib.metadata
    settings = profile(system)
    dependencies = ['sentence-transformers', 'numpy', 'openai']
    dependencies += ['chromadb', 'rank-bm25', 'spacy'] if system=='nemori' else ['qdrant-client','llmlingua','tiktoken']
    if system=='nemori':
        nemori_modules()
    else:
        from memcut.native_memory.lightmem_v1 import native_modules
        native_modules()
    return {'system':system, 'upstream_commit':settings['commit'], 'assets':load_assets(system),
            'packages':{name:importlib.metadata.version(name) for name in dependencies}, 'api_calls':0}
