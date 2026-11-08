"""Explicit MemCut CLI; selecting a system without a command prints help."""
import argparse
import os
from pathlib import Path
import runpy
import sys

if sys.platform == 'darwin':     # faiss and torch each bundle libomp on macOS; MemoryBank loads both (OMP Error #15)
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

ROOT = Path(__file__).resolve().parent
ENTRIES = {
    'aligned': 'memcut_s012.py',
    'aligned-retention': 'aligned_retention.py',
    'efficient': 'efficient_s012.py',
    'memorybank': 'memorybank_s012.py',
    'memorybank-efficient': 'memorybank_efficient_s012.py',
    'source': 'source_anchored_s012.py',
    'simplemem': 'simplemem_s012.py',
    'turnrag': 'raw_turn_rag.py',
    'sessionrag': 'session_rag.py',
    'glm': 'glm_s012.py',
    'qwen': 'qwen_s012.py',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('system', choices=ENTRIES)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    target = ROOT / 'tools' / ENTRIES[args.system]
    sys.path.insert(0, str(ROOT))
    sys.argv = [str(target), *(args.arguments or ['--help'])]
    runpy.run_path(str(target), run_name='__main__')


if __name__ == '__main__':
    main()
