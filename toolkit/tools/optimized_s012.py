"""Compatibility CLI; implementation lives in memcut.execution_optimization.cli_core."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memcut.execution_optimization.cli_core import *
if __name__ == "__main__":
    main()
