#!/usr/bin/env bash
set -e

echo "=== Local LLM Benchmark Setup for Apple Silicon (M2 8GB) ==="

# Check macOS platform
if [[ "$(uname)" != "Darwin" ]]; then
    echo "[!] Warning: This setup script is optimized for macOS Apple Silicon."
fi

echo "[*] Upgrading pip, setuptools, wheel..."
python3 -m pip install --upgrade pip setuptools wheel

echo "[*] Installing core dependencies..."
python3 -m pip install -r requirements.txt

echo "[*] Installing llama-cpp-python with Metal acceleration..."
CMAKE_ARGS="-DGGML_METAL=on" python3 -m pip install llama-cpp-python --no-cache-dir || echo "[!] Could not build llama-cpp-python with Metal. Falling back to default install."

echo "[✓] Environment setup completed successfully."
