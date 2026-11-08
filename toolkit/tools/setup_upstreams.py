"""Clone one pinned source explicitly; no package installation, models, or paid calls."""
import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def recipe(system):
    if system == 'memorybank':
        config = json.loads((ROOT/'configs/native_memory_profiles.json').read_text())
        return {**config['profiles']['memorybank_faiss_en']['upstream'],
                'url': 'https://github.com/zhongwanjun/MemoryBank-SiliconFriend.git'}
    if system == 'simplemem':
        return json.loads((ROOT/'configs/simplemem_s012_v1.json').read_text())
    return json.loads((ROOT/'configs/source_anchored_systems_v1.json').read_text())[system]


def setup(system):
    config = recipe(system)
    destination = ROOT / config['repository']
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', '--no-checkout', config['url'], str(destination)], check=True)
        subprocess.run(['git', '-C', str(destination), 'checkout', '--detach', config['commit']], check=True)
    actual = subprocess.check_output(
        ['git', '-C', str(destination), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != config['commit']:
        raise RuntimeError('Existing checkout differs; not resetting it: ' + str(destination))
    print(json.dumps({'system': system, 'repository': str(destination),
                      'commit': actual, 'api_calls': 0}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('system', choices=['memorybank', 'nemori', 'lightmem', 'simplemem'])
    setup(parser.parse_args().system)


if __name__ == '__main__':
    main()
