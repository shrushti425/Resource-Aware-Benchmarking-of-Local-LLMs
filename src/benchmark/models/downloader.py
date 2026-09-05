"""Hugging Face Model Downloader for GGUF files."""

import sys
from pathlib import Path
from typing import Optional
from benchmark.models.registry import ModelRegistry


class ModelDownloader:
    """Explicit model downloader from Hugging Face Hub."""

    def __init__(self, registry: Optional[ModelRegistry] = None):
        self.registry = registry or ModelRegistry()

    def download_model(self, model_id: str, force: bool = False) -> Path:
        """Downloads a specific GGUF model file explicitly requested by user."""
        model_cfg = self.registry.get_model(model_id)
        if not model_cfg:
            raise ValueError(f"Model ID '{model_id}' not found in registry.")

        target_file = self.registry.get_model_file_path(model_id)
        target_dir = target_file.parent
        target_dir.mkdir(parents=True, exist_ok=True)

        if target_file.exists() and not force:
            print(f"[+] Model '{model_id}' already downloaded at: {target_file}")
            return target_file

        print(f"[*] Downloading '{model_id}' ({model_cfg.file_size_gb} GB expected) from {model_cfg.repo_id}...")

        try:
            from huggingface_hub import hf_hub_download

            downloaded_path = hf_hub_download(
                repo_id=model_cfg.repo_id,
                filename=model_cfg.filename,
                local_dir=str(target_dir),
                local_dir_use_symlinks=False,
            )
            print(f"[✓] Successfully downloaded model to: {downloaded_path}")
            return Path(downloaded_path)
        except Exception as e:
            print(f"[!] Error downloading model '{model_id}': {e}", file=sys.stderr)
            raise RuntimeError(
                f"Failed to download GGUF model from Hugging Face ({model_cfg.repo_id}/{model_cfg.filename}). Error: {e}"
            )
