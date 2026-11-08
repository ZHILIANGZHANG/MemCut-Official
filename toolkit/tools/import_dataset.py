"""Import only manifest-listed data files from a MemCut bundle: the 15 main-set episodes and any case-study episodes.

Case-study episodes (manifest `set: "case_study"`, e.g. EP08A) extend a main-set episode with one designed fact; the
summary lists them apart so a main-set run never picks them up by accident.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

ROLES = {'source', 'contracts', 'annotations', 'tokens'}


def import_bundle(bundle, destination):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('Choose a new destination; existing data is never overwritten')
    payloads = {}
    with zipfile.ZipFile(bundle) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate ZIP members')
        manifest = json.loads(archive.read('MANIFEST.json'))
        if manifest.get('format') != 'memcut-15ep-data-bundle-v1':
            raise ValueError('Unsupported bundle')
        episodes = manifest['episodes']
        main = [e for e in episodes if e.get('set', 'main') == 'main']
        cases = [e for e in episodes if e.get('set', 'main') != 'main']
        if len({episode['id'] for episode in episodes}) != len(episodes):
            raise ValueError('Duplicate episode ids')
        if len(main) != 15:
            raise ValueError('Expected 15 main-set episodes')
        if any(e.get('set') != 'case_study' or e.get('base_episode') not in {m['id'] for m in main} for e in cases):
            raise ValueError('A non-main episode must be a case study of a main-set episode')
        for episode in episodes:
            if not ROLES <= set(episode['files']):
                raise ValueError('Missing file role')
            for entry in episode['files'].values():
                path = PurePosixPath(entry['path'])
                if (path.is_absolute() or '..' in path.parts or not path.parts
                        or path.parts[0] != 'data' or '\\' in entry['path']):
                    raise ValueError('Unsafe dataset path')
                if str(path) in payloads:
                    raise ValueError('Duplicate dataset path')
                info = archive.getinfo(str(path))
                if info.file_size > 32 * 1024 * 1024:
                    raise ValueError('EP file exceeds 32 MiB')
                raw = archive.read(info)
                if (len(raw) != entry['bytes']
                        or hashlib.sha256(raw).hexdigest() != entry['sha256']):
                    raise ValueError('Dataset payload differs: ' + str(path))
                payloads[str(path)] = raw
    # Validate the whole bundle before creating any destination files.
    destination.mkdir(parents=True)
    for name, raw in payloads.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    (destination / 'MANIFEST.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return {'episodes': len(main), 'case_study_episodes': [e['id'] for e in cases], 'files': len(payloads),
            'destination': str(destination)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(import_bundle(args.bundle, args.output), ensure_ascii=False))


if __name__ == '__main__':
    main()
