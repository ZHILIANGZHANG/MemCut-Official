"""Compatibility entry for explicit pinned SimpleMem setup."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.setup_upstreams import setup
if __name__ == "__main__":
    setup("simplemem")
