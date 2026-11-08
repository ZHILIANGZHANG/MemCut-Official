"""Source-release boundaries; no native models, network, or paid calls."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest
from tools.import_dataset import import_bundle
from tools.setup_upstreams import recipe

ROOT = Path(__file__).resolve().parents[1]


def bundle(path, change=None):
    entries, payloads = [], {}
    for index in range(15):
        files = {}
        for role in ('source', 'contracts', 'annotations', 'tokens'):
            name = f'data/EP{index:02}/{role}.json'
            raw = json.dumps({'ep': index, 'role': role}).encode()
            files[role] = {'path': name, 'bytes': len(raw),
                           'sha256': hashlib.sha256(raw).hexdigest()}
            payloads[name] = raw
        entries.append({'id': f'EP{index:02}', 'files': files})
    manifest = {'format': 'memcut-15ep-data-bundle-v1', 'episodes': entries}
    if change:
        change(manifest)
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('MANIFEST.json', json.dumps(manifest))
        archive.writestr('unlisted.txt', 'Do not import')
        for name, raw in payloads.items():
            archive.writestr(name, raw)


def test_dataset_import_and_no_overwrite(tmp_path):
    archive, output = tmp_path/'data.zip', tmp_path/'import'
    bundle(archive)
    assert import_bundle(archive, output)['files'] == 60
    assert len(list((output/'data').rglob('*.json'))) == 60
    assert not (output/'unlisted.txt').exists()
    with pytest.raises(FileExistsError):
        import_bundle(archive, output)


@pytest.mark.parametrize('kind', ['sha', 'traversal', 'duplicate_episode'])
def test_invalid_dataset_leaves_destination_absent(tmp_path, kind):
    def change(manifest):
        row = manifest['episodes'][0]['files']['source']
        if kind == 'sha':
            row['sha256'] = '0'*64
        elif kind == 'traversal':
            row['path'] = 'data/../../escape.json'
        else:
            manifest['episodes'][1]['id'] = manifest['episodes'][0]['id']
    archive, output = tmp_path/'data.zip', tmp_path/'import'
    bundle(archive, change)
    with pytest.raises(ValueError):
        import_bundle(archive, output)
    assert not output.exists()


def case_study(manifest, base='EP03'):
    raw = json.dumps({'ep': 'case'}).encode()
    files = {r: {'path': f'data/EP03A/{r}.json', 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
             for r in ('source', 'contracts', 'annotations', 'tokens')}
    manifest['episodes'].append({'id': 'EP03A', 'set': 'case_study', 'base_episode': base, 'files': files})
    return raw


def test_case_study_episodes_import_and_are_listed_apart(tmp_path):
    archive, output = tmp_path/'data.zip', tmp_path/'import'
    extra = {}
    bundle(archive, lambda m: extra.setdefault('raw', case_study(m)))
    with zipfile.ZipFile(archive, 'a') as z:
        for r in ('source', 'contracts', 'annotations', 'tokens'):
            z.writestr(f'data/EP03A/{r}.json', extra['raw'])
    summary = import_bundle(archive, output)
    assert summary['episodes'] == 15 and summary['case_study_episodes'] == ['EP03A']
    assert (output/'data/EP03A/source.json').exists()


def test_a_case_study_of_an_unknown_episode_is_refused(tmp_path):
    archive, output = tmp_path/'data.zip', tmp_path/'import'
    bundle(archive, lambda m: case_study(m, base='EP99'))
    with pytest.raises(ValueError):
        import_bundle(archive, output)
    assert not output.exists()


@pytest.mark.parametrize('system', ['aligned'])
def test_cli_help_from_other_directory(tmp_path, system):
    for extra in ([], ['--help']):
        result = subprocess.run([sys.executable, str(ROOT/'run.py'), system, *extra],
                                cwd=tmp_path, capture_output=True, text=True, timeout=20)
        assert result.returncode == 0, result.stderr
        assert 'usage:' in result.stdout




@pytest.mark.parametrize('system', ['memorybank', 'nemori', 'lightmem'])
def test_upstream_recipes_are_pinned_inside_release(system):
    config = recipe(system)
    assert len(config['commit']) == 40
    assert config['repository'] == 'runtime/upstream/'+system
    assert config['url'].startswith('https://github.com/')
