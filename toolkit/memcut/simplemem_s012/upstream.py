"""Load verified text-core sources without modifying other systems' runtimes."""
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'configs/simplemem_s012_v1.json'


def profile():
    value = json.loads(CONFIG.read_text())
    path = ROOT / value['repository']
    actual = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != value['commit']:
        raise ValueError('SimpleMem upstream commit differs from the pinned text baseline')
    for name, expected in value['files'].items():
        if hashlib.sha256((path / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'SimpleMem upstream source changed: {name}')
    return value


def modules():
    value = profile()
    path = str((ROOT / value['repository']).resolve())
    if 'simplemem' in sys.modules:
        loaded = str(getattr(sys.modules['simplemem'], '__file__', ''))
        if not loaded.startswith(path + '/'):
            raise ValueError('Another SimpleMem package is loaded')
    if path not in sys.path:
        sys.path.insert(0, path)
    # Use explicit text components, never the auto-router's Omni/Evolve path.
    for name in ('models.memory_entry', 'memory_builder', 'hybrid_retriever',
                 'database.vector_store', 'database.vector_store_backend', 'answer_generator'):
        importlib.import_module('simplemem.core.' + name)
    # Prevent an unrelated config.py or shell setting changing native requests.
    importlib.import_module('simplemem.core.settings').settings.USE_JSON_FORMAT = False
    return value


def runtime_identity():
    names = ('lancedb', 'pyarrow', 'tantivy', 'dateparser', 'numpy', 'pydantic',
             'sentence-transformers', 'transformers', 'torch', 'openai', 'pylance',
             'lance-namespace', 'lance-namespace-urllib3-client')
    return {name: importlib.metadata.version(name) for name in names}


def doctor():
    value = modules()
    versions = runtime_identity()
    for name, version in value['native_packages'].items():
        if versions[name] != version:
            raise ValueError(f'Unsupported SimpleMem native dependency: {name}')
    from memcut.source_anchored_systems.profiles import load_assets
    return {'system': 'simplemem', 'mode': 'text', 'commit': value['commit'],
            'packages': versions, 'assets': load_assets('simplemem'), 'api_calls': 0}
