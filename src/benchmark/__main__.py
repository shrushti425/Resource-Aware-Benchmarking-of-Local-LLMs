"""Entry point for python -m benchmark."""

import sys
from pathlib import Path

# Ensure src in sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from benchmark.cli import main

if __name__ == "__main__":
    main()
