# Resource-Aware Benchmarking of Local LLMs on Apple Silicon for Offline AI and RAG Applications

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Platform: macOS Apple Silicon](https://img.shields.io/badge/platform-macOS_M1%2FM2%2FM3-brightgreen.svg)](https://support.apple.com/en-us/HT211814)

A reproducible, hardware-aware research benchmarking suite specifically engineered to evaluate open Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) pipelines on resource-constrained **Apple Silicon (MacBook Air M2 with 8 GB Unified Memory)**.

---

## 1. Project Overview

Running LLMs locally on consumer laptops eliminates cloud API costs and preserves privacy. However, on an **8 GB Apple Silicon laptop**, memory is unified between the CPU, GPU, and macOS operating system. Models exceeding safe RAM allocations trigger severe memory pressure and disk swap thrashing.

This repository provides an automated research environment to evaluate **1B to 4B parameter models** (Llama-3.2, Qwen-2.5, Gemma-2) and selected 7B models under strict unified memory limits using `llama-cpp-python` with **Metal GPU acceleration**.

---

## 2. Research Questions

- **Primary Research Question:**
  > *How do local LLM family, model size, and quantization affect answer quality, inference speed, latency, and memory consumption on a resource-constrained Apple Silicon laptop?*

- **Secondary Research Question (Resource Efficiency):**
  > *Given an 8 GB unified memory limit, which local LLM configuration provides the best quality-efficiency trade-off?*

- **Secondary Research Question (RAG):**
  > *How does the choice of local generation model affect answer quality and performance when the retrieval system remains fixed?*

- **Research Contribution:**
  > *A reproducible resource-aware experimental framework for selecting practical local LLM configurations under strict consumer hardware constraints.*

---

## 3. System Architecture

```text
                                 +-----------------------------------+
                                 |   Apple Silicon Hardware Monitor  |
                                 |  (M2 Chip, RAM, Swap, Metal API)  |
                                 +-----------------------------------+
                                                   |
 +-----------------------+        +-----------------------------------+        +-----------------------+
 |    Model Registry     |  --->  |    llama.cpp / Metal Backend      |  --->  |   Benchmark Tasks     |
 | (Llama, Qwen, Gemma)  |        |  (GGUF, Q4_K_M, Q8, CPU Fallback) |        | (QA, RAG, JSON, etc.) |
 +-----------------------+        +-----------------------------------+        +-----------------------+
                                                   |
                                  +----------------------------------+
                                  | Resource & Performance Profiler  |
                                  |  (TTFT, TPS, RSS RAM, Latency)   |
                                  +----------------------------------+
                                                   |
                                  +----------------------------------+
                                  | Pareto & Recommendation System   |
                                  | (JSONL Storage, Dashboard, Paper)|
                                  +----------------------------------+
```

---

## 4. Hardware Requirements & Apple Silicon Optimization

The benchmark is explicitly calibrated for:

```text
Laptop: MacBook Air M2
RAM: 8 GB unified memory
Operating System: macOS (13.0+)
GPU: Integrated Apple Silicon GPU
Acceleration: Metal API
```

### Memory Guard Policy (8 GB RAM):
- **Safe Allocated Memory Threshold:** <= 4.0 GB
- **Infeasible Threshold:** Models requiring > 5.5 GB peak RAM are automatically flagged as `infeasible_for_current_hardware` to prevent system crashes.

---

## 5. Model Selection & Quantization

| Model ID | Family | Params (B) | Quantization | Expected Size | Hardware Feasibility |
|---|---|---|---|---|---|
| `qwen2.5-0.5b-q4` | Qwen | 0.5 | Q4_K_M | 0.39 GB | High |
| `llama-3.2-1b-q4` | Llama | 1.2 | Q4_K_M | 0.81 GB | High |
| `llama-3.2-1b-q8` | Llama | 1.2 | Q8_0 | 1.32 GB | High |
| `qwen2.5-1.5b-q4` | Qwen | 1.5 | Q4_K_M | 0.98 GB | High |
| `gemma-2-2b-q4` | Gemma | 2.6 | Q4_K_M | 1.65 GB | Medium |
| `llama-3.2-3b-q4` | Llama | 3.2 | Q4_K_M | 2.02 GB | Medium |
| `qwen2.5-3b-q4` | Qwen | 3.1 | Q4_K_M | 1.93 GB | Medium |
| `mistral-7b-q4` | Mistral | 7.2 | Q4_K_M | 4.37 GB | Impractical (8GB) |

---

## 6. Benchmark Tasks & Metrics

1. **Question Answering (QA):** Exact Match & Token F1
2. **Text Classification:** Accuracy & Macro Precision/Recall/F1
3. **Summarization:** ROUGE-1, ROUGE-2, ROUGE-L
4. **Information Extraction:** Entity Overlap F1
5. **Structured JSON Generation:** JSON Validity, Schema Key Compliance, Field Correctness
6. **Reasoning:** Step-by-Step Logic & Math Exact Match
7. **Retrieval-Augmented Generation (RAG):** Retrieval Recall@1/3/5, MRR, Answer Generation F1

---

## 7. Installation Instructions (macOS Apple Silicon)

```bash
# 1. Clone or navigate to the repository
cd local-llm-benchmark

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install llama-cpp-python with Metal GPU support
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --no-cache-dir

# 5. Install project in editable mode
pip install -e .
```

---

## 8. Usage Guide & CLI Commands

### Hardware Diagnostics
```bash
python -m benchmark hardware
```

### Model Management
```bash
# List registered models
python -m benchmark models list

# Verify local model files and memory feasibility
python -m benchmark models verify

# Explicitly download a model GGUF file
python -m benchmark models download --model llama-3.2-1b-q4
```

### Running Benchmarks
```bash
# 1. Zero-Download Synthetic Mock Run (instant dry run verification)
python -m benchmark run --profile quick --mock

# 2. Real Model Quick Run (requires downloaded model)
python -m benchmark run --profile quick

# 3. Standard Research Benchmark Run
python -m benchmark run --profile standard

# 4. Target specific model and task
python -m benchmark run --model llama-3.2-1b-q4 --task qa

# 5. Resume interrupted benchmark run
python -m benchmark run --profile standard --resume
```

### Analysis & Figures
```bash
# Display summary table of aggregated metrics
python -m benchmark evaluate

# Generate all 9 research figures in plots/
python -m benchmark analyze
```

### Hardware-Aware Model Recommendations
```bash
python -m benchmark recommend --max-memory 4.0 --max-latency 10.0 --min-quality 0.60
```

### Academic Research Report Generation
```bash
python -m benchmark report
```

### Streamlit Research Dashboard
```bash
streamlit run dashboard/app.py
```

---

## 9. Running Tests

Execute the complete test suite (unit tests use synthetic backends, requiring 0 model downloads):

```bash
pytest
```

---

## 10. Research Contribution & Limitations

### Contribution
Provides an empirical baseline and automated evaluation harness for selecting localized LLMs on Apple Silicon without relying on cloud APIs.

### Limitations
- **Unified RAM Boundary:** 8 GB RAM caps practical model size to <=4B.
- **Automated Evaluation:** Metrics focus on deterministic string overlap, exact match, and schema compliance rather than human qualitative feedback.
- **Runtime Variance:** macOS background process activity can introduce minor timing noise in latency measurements.
