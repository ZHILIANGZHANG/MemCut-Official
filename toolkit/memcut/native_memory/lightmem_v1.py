"""Pinned LightMem native store and retrieval, without importing old runners."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

from .contract import MemoryObject, NativeAdapter, SearchResult, digest
from .persistence import write_json

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / json.loads((ROOT / 'configs/source_anchored_systems_v1.json').read_text())['lightmem']['repository']
COMMIT = '8449d574df6bae1bdf3314a1564da65e2f37e046'
PROFILE_PATH = ROOT / 'configs/lightmem_native_v1.json'
TEXT_FIELDS = ('memory', 'original_memory', 'compressed_memory', 'topic_summary')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_upstream():
    profile = read(PROFILE_PATH)
    if subprocess.check_output(['git', '-C', str(UPSTREAM), 'rev-parse', 'HEAD'], text=True).strip() != COMMIT:
        raise ValueError('LightMem checkout differs from the pinned native reference')
    for path, expected in profile['upstream_files'].items():
        if sha(UPSTREAM / path) != expected:
            raise ValueError(f'Pinned LightMem file changed: {path}')
    return profile


def native_modules():
    verify_upstream()
    src = str(UPSTREAM / 'src')
    if src not in sys.path:
        sys.path.insert(0, src)
    from lightmem.factory.retriever.embeddingretriever.qdrant import Qdrant, QdrantConfig
    name = 'memcut_lightmem_v1_locomo_retrievers'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, UPSTREAM / 'experiments/locomo/retrievers.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return Qdrant, QdrantConfig, sys.modules[name].VectorRetriever


def payload_record(payload):
    """All substantive stored text; no vectors, queues or repeated JSON shells."""
    texts = {}
    for key in TEXT_FIELDS:
        value = payload.get(key)
        if value is not None and not isinstance(value, str):
            raise ValueError(f'Unknown LightMem text representation: {key}')
        if value and value not in texts.values():
            texts[key] = value
    result = {key: payload[key] for key in ('speaker_name', 'speaker_id', 'time_stamp', 'weekday')
              if payload.get(key) is not None}
    result['text'] = texts
    if result.get('speaker_id') == result.get('speaker_name'):
        result.pop('speaker_id', None)
    return result


def object_from_row(row):
    return MemoryObject('lightmem', 'long_term', str(row['id']),
        json.dumps(payload_record(row['payload']), ensure_ascii=False, separators=(',', ':')),
        digest({'payload': row['payload'], 'vector': row.get('vector')}))


class LightMemAdapter(NativeAdapter):
    system = 'lightmem'

    def __init__(self, bank, *, embedder=None, _create_fixture=False):
        self.bank, self.embedder = Path(bank), embedder
        manifest = read(self.bank / 'BANK.json')
        if manifest['format'] != 'lightmem-native-bank-v1':
            raise ValueError('Not a current LightMem bank; old snapshots require explicit migration')
        if (self.bank / 'FROZEN.json').exists():
            raise ValueError('Fork a frozen bank before opening its native database')
        if not _create_fixture and not (self.bank / 'details').is_dir():
            raise ValueError('Missing native database; reopening must never create a new empty bank')
        Qdrant, Config, self.retriever_class = native_modules()
        self.store = Qdrant(Config(collection_name='details', path=str(self.bank / 'details'),
                                  embedding_model_dims=384, on_disk=True))

    def rows(self):
        return self.store.get_all(with_vectors=True)

    async def audit_objects(self):
        return [object_from_row(row) for row in self.rows()]

    async def search(self, query, *, top_k=20):
        if type(top_k) is not int or top_k < 1 or self.embedder is None:
            raise ValueError('Native LightMem search needs a positive k and the pinned local encoder')
        rows = self.rows()
        by_id = {str(r['id']): object_from_row(r) for r in rows}
        results = self.retriever_class(self.embedder).retrieve(rows, query, limit=top_k)
        if len(results) != min(top_k, len(rows)):
            raise ValueError('Native retrieval failed or skipped stored vectors; do not score as empty evidence')
        objects = [by_id[str(row['id'])] for row in results]
        return SearchResult(query, {'long_term': objects}, native_context={'results': results},
            costs={'requested_top_k': top_k, 'native_operation': 'locomo.VectorRetriever.retrieve',
                   'retrieval_backend': 'native_cosine_over_stored_vectors', 'embedding_calls': 1,
                   'returned_count': len(objects), 'query_rewritten': False})

    async def _erase(self, obj):
        self.store.delete(obj.native_id)
        return {'native_operation': 'Qdrant.delete', 'removed_point_and_vector': obj.native_id,
                'semantic_descendants_deleted': False, 'reembedding_calls': 0,
                'update_queues': 'unchanged native ID-only metadata; no post-formation consolidation during evaluation'}

    def close(self):
        self.store.client.close()


def local_embedder(assets):
    native_modules()
    from lightmem.factory.text_embedder.huggingface import TextEmbedderHuggingface
    from lightmem.configs.text_embedder.base_config import BaseTextEmbedderConfig
    return TextEmbedderHuggingface(BaseTextEmbedderConfig(
        model=assets['embedder'], model_kwargs={'device': 'cpu'}))


def create_fixture_bank(bank, rows, lineage, *, source_sha256='fixture', model='fixture'):
    """Test-only input injection. Production banks are made by native formation."""
    bank = Path(bank)
    if bank.exists():
        raise FileExistsError(bank)
    bank.mkdir(parents=True)
    write_json(bank / 'BANK.json', {'format': 'lightmem-native-bank-v1',
        'source_sha256': source_sha256, 'model': model, 'test_fixture': True})
    adapter = LightMemAdapter(bank, _create_fixture=True)
    if rows:
        adapter.store.insert(ids=[r['id'] for r in rows], vectors=[r['vector'] for r in rows],
                             payloads=[r['payload'] for r in rows])
    objects = {o.native_id: o for o in __import__('asyncio').run(adapter.audit_objects())}
    write_json(bank / 'LINEAGE.json', {rid: {**entry, 'version': objects[rid].version,
                                           'content_sha256': digest(objects[rid].text)}
                                      for rid, entry in lineage.items()})
    adapter.close()
