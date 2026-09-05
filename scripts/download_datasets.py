"""Dataset Downloader / Pre-fetch Script."""

import json
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
datasets_dir = root_dir / "datasets"
datasets_dir.mkdir(parents=True, exist_ok=True)


def generate_local_datasets():
    """Pre-populates offline benchmark datasets in datasets/ directory."""
    print("[*] Pre-populating local dataset JSON files in datasets/...")

    qa_file = datasets_dir / "qa_dataset.json"
    if not qa_file.exists():
        qa_data = [
            {
                "id": "qa_001",
                "context": "Apple Silicon M2 uses unified memory architecture allowing CPU, GPU, and Neural Engine shared access.",
                "question": "What architecture does Apple Silicon M2 use?",
                "reference": "unified memory architecture",
            },
            {
                "id": "qa_002",
                "context": "llama.cpp accelerates GGUF models on macOS using Apple Metal API shaders.",
                "question": "Which API provides GPU acceleration for llama.cpp on macOS?",
                "reference": "Apple Metal API",
            },
        ]
        with open(qa_file, "w", encoding="utf-8") as f:
            json.dump(qa_data, f, indent=2)

    print(f"[✓] Datasets verified in: {datasets_dir}")


if __name__ == "__main__":
    generate_local_datasets()
