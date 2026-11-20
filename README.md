# LocalLLM-Bench: Resource-Aware Benchmarking of Local LLMs on 8 GB Apple Silicon

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Platform: macOS Apple Silicon](https://img.shields.io/badge/platform-macOS_Apple_Silicon-brightgreen.svg)](https://support.apple.com/en-us/HT211814)

A reproducible benchmarking framework that evaluates open large language models and retrieval-augmented generation (RAG) pipelines under strict memory limits on a MacBook Air M2 with 8 GB of unified memory. It measures answer quality alongside speed, latency, and memory use, then recommends configurations that fit a given hardware budget.

---

## Table of Contents

1. [Overview](#overview)
2. [Motivation](#motivation)
3. [Research Questions](#research-questions)
4. [Key Features](#key-features)
5. [System Architecture](#system-architecture)
6. [Experimental Setup](#experimental-setup)
7. [Benchmark Tasks and Metrics](#benchmark-tasks-and-metrics)
8. [Repository Structure](#repository-structure)
9. [Installation](#installation)
10. [Usage](#usage)
11. [Results](#results)
12. [Testing](#testing)
13. [Limitations](#limitations)
14. [Future Work](#future-work)

---

## Overview

Running language models locally avoids cloud API costs and keeps data on the device. On an 8 GB Apple Silicon laptop, however, memory is shared between the CPU, the GPU, and macOS. A model that exceeds its safe allocation triggers memory pressure and disk swapping, which degrades both speed and stability.

This repository evaluates models from roughly 0.5B to 4B parameters, plus a 7B model as a boundary case, from the Llama 3.2, Qwen 2.5, Gemma 2, and Mistral families. Inference uses `llama-cpp-python` with Metal GPU acceleration and GGUF quantised weights. Each model is measured on a set of tasks for quality and profiled for time to first token, tokens per second, resident memory, and latency.

## Motivation

Published model comparisons usually report quality on high-end hardware and rarely say whether a model is practical on the machine a student or small team actually owns. Choosing a local model for a laptop is a trade-off between quality, speed, and memory, and the options (model family, size, quantisation level) multiply quickly.

This project was built to answer that choice empirically for one realistic and tightly constrained platform, and to make the method reusable so that anyone can repeat it on their own hardware.

## Research Questions

**Primary.** How do model family, model size, and quantisation affect answer quality, inference speed, latency, and memory consumption on a resource-constrained Apple Silicon laptop?

**Resource efficiency.** Given an 8 GB unified memory limit, which local LLM configuration offers the best trade-off between quality and efficiency?

**RAG.** When the retrieval system is held fixed, how does the choice of generation model affect answer quality and performance?

**Contribution.** A reproducible, resource-aware experimental framework for selecting practical local LLM configurations under consumer hardware constraints.

## Key Features

- Seven benchmark task types covering question answering, classification, summarisation, extraction, structured JSON output, reasoning, and RAG
- Resource profiling per run: time to first token, tokens per second, peak resident memory, and latency
- Memory guard that flags models likely to exceed safe memory as infeasible, instead of letting them crash the machine
- Model registry with verification and on-demand download of GGUF files
- Mock mode that runs the full pipeline with synthetic outputs and no model downloads
- Resumable runs, with results stored as JSONL
- Pareto analysis and a constraint-based recommender
- Nine generated research figures, a Streamlit dashboard, and an automated report generator
- Unit tests that run without any model downloads

## System Architecture

```text
                      +-----------------------------------+
                      |   Apple Silicon Hardware Monitor  |
                      |  (M2 chip, RAM, swap, Metal API)  |
                      +-----------------------------------+
                                        |
+---------------------+   +-----------------------------------+   +-----------------------+
|    Model Registry   |-->|     llama.cpp / Metal backend     |-->|    Benchmark Tasks    |
| (Llama, Qwen, Gemma)|   | (GGUF, Q4_K_M, Q8_0, CPU fallback)|   | (QA, RAG, JSON, etc.) |
+---------------------+   +-----------------------------------+   +-----------------------+
                                        |
                       +----------------------------------+
                       | Resource and performance profiler|
                       |  (TTFT, TPS, RSS RAM, latency)   |
                       +----------------------------------+
                                        |
                       +----------------------------------+
                       | Pareto analysis and recommender  |
                       | (JSONL storage, dashboard, report)|
                       +----------------------------------+
```

## Experimental Setup

**Hardware**

| Item | Specification |
|---|---|
| Laptop | MacBook Air M2 |
| Memory | 8 GB unified |
| Operating system | macOS 13.0 or later |
| GPU | Integrated Apple Silicon GPU |
| Acceleration | Metal |

**Memory guard policy**

- Safe allocation threshold: 4.0 GB
- Models requiring more than 5.5 GB peak memory are labelled `infeasible_for_current_hardware` and are not run.

**Models**

| Model ID | Family | Parameters (B) | Quantisation | File size (GB) | Feasibility on 8 GB |
|---|---|---|---|---|---|
| `qwen2.5-0.5b-q4` | Qwen | 0.5 | Q4_K_M | 0.39 | High |
| `llama-3.2-1b-q4` | Llama | 1.2 | Q4_K_M | 0.81 | High |
| `llama-3.2-1b-q8` | Llama | 1.2 | Q8_0 | 1.32 | High |
| `qwen2.5-1.5b-q4` | Qwen | 1.5 | Q4_K_M | 0.98 | High |
| `gemma-2-2b-q4` | Gemma | 2.6 | Q4_K_M | 1.65 | Medium |
| `llama-3.2-3b-q4` | Llama | 3.2 | Q4_K_M | 2.02 | Medium |
| `qwen2.5-3b-q4` | Qwen | 3.1 | Q4_K_M | 1.93 | Medium |
| `mistral-7b-q4` | Mistral | 7.2 | Q4_K_M | 4.37 | Impractical |

## Benchmark Tasks and Metrics

| Task | Metrics |
|---|---|
| Question answering | Exact match, token F1 |
| Text classification | Accuracy, macro precision, recall, and F1 |
| Summarisation | ROUGE-1, ROUGE-2, ROUGE-L |
| Information extraction | Entity overlap F1 |
| Structured JSON generation | JSON validity, schema key compliance, field correctness |
| Reasoning | Step-by-step logic and math exact match |
| Retrieval-augmented generation | Retrieval recall at 1, 3, and 5; mean reciprocal rank; answer F1 |

**Resource metrics:** time to first token, tokens per second, peak resident memory, and end-to-end latency.

## Repository Structure

```text
local-llm-benchmark/
├── benchmark/             # Core package: CLI, model registry, tasks, profiling, analysis
├── dashboard/
│   └── app.py             # Streamlit research dashboard
├── plots/                 # Generated figures
├── tests/                 # Unit tests (synthetic backends)
├── requirements.txt
└── pyproject.toml         # Package metadata for editable install
```

## Installation

Requires macOS on Apple Silicon and Python 3.9 or later.

```bash
cd local-llm-benchmark

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

# llama-cpp-python with Metal GPU support
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --no-cache-dir

# Install the project in editable mode
pip install -e .
```

## Usage

**Hardware diagnostics**

```bash
python -m benchmark hardware
```

**Model management**

```bash
python -m benchmark models list
python -m benchmark models verify
python -m benchmark models download --model llama-3.2-1b-q4
```

**Running benchmarks**

```bash
# Dry run with synthetic outputs (no downloads)
python -m benchmark run --profile quick --mock

# Quick run on real models
python -m benchmark run --profile quick

# Full research run
python -m benchmark run --profile standard

# One model and one task
python -m benchmark run --model llama-3.2-1b-q4 --task qa

# Resume an interrupted run
python -m benchmark run --profile standard --resume
```

**Analysis and reporting**

```bash
python -m benchmark evaluate     # summary table of aggregated metrics
python -m benchmark analyze      # generates the nine figures in plots/
python -m benchmark report       # generates the research report
streamlit run dashboard/app.py   # interactive dashboard
```

**Hardware-aware recommendation**

```bash
python -m benchmark recommend --max-memory 4.0 --max-latency 10.0 --min-quality 0.60
```

## Results

Results are generated by the commands above and written to `plots/` and the JSONL result files.

<!-- After running the standard profile, insert here: (1) the summary table from `python -m benchmark evaluate`, (2) the Pareto figure of quality against memory or latency, (3) the recommended configuration for the 4.0 GB / 10 s / 0.60 constraint, and (4) two or three sentences of analysis answering each research question. -->

## Testing

```bash
pytest
```

The unit tests use synthetic backends and require no model downloads.

## Limitations

- **Memory ceiling.** The 8 GB limit restricts practical model size to about 4B parameters.
- **Automated metrics only.** Quality is measured with string overlap, exact match, and schema compliance, not human judgement, so fluent but differently worded answers can be undercounted.
- **Timing variance.** Background activity in macOS adds noise to latency measurements.
- **Single platform.** Results apply to one machine and may not transfer to other Apple Silicon chips or memory sizes.

## Future Work

- Repeat the study on additional memory configurations (16 GB and above) to show how conclusions change
- Add human or LLM-as-judge evaluation for open-ended tasks
- Study the effect of context length and retrieval settings on RAG quality
- Compare additional quantisation schemes and runtimes
