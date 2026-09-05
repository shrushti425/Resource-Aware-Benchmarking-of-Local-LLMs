"""Diagnostics script checking macOS environment and dependencies."""

import sys
import platform
from pathlib import Path

# Add src to path
root_dir = Path(__file__).resolve().parent.parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from benchmark.hardware import HardwareMonitor


def check_env():
    print("=== Apple Silicon Environment Diagnostics ===")
    hw = HardwareMonitor.detect_hardware()
    for k, v in hw.items():
        print(f"  {k:30s}: {v}")

    print("\n--- Package Dependencies ---")
    packages = ["yaml", "pydantic", "click", "numpy", "pandas", "matplotlib", "seaborn", "streamlit", "pytest"]
    for pkg in packages:
        try:
            __import__(pkg)
            print(f"  [✓] {pkg:25s}: Installed")
        except ImportError:
            print(f"  [✗] {pkg:25s}: NOT installed")

    # llama-cpp-python check
    try:
        import llama_cpp
        print(f"  [✓] {'llama-cpp-python':25s}: Installed")
    except ImportError:
        print(f"  [!] {'llama-cpp-python':25s}: NOT installed (Use --mock or install via pip)")

    # FAISS check
    try:
        import faiss
        print(f"  [✓] {'faiss':25s}: Installed")
    except ImportError:
        print(f"  [i] {'faiss':25s}: Not installed (Using NumPy vector store fallback)")


if __name__ == "__main__":
    check_env()
