# SamatNext-CL: A Consumer-GPU Hybrid Language Model Architecture for Low-Memory Training

> **Systems / Architecture Preprint**  
> **Target Hardware Class:** Consumer-grade Laptop GPUs (12GB VRAM class, e.g., NVIDIA RTX 4070 / 5070 Laptop GPUs).  
>
> **Important Framing:** This work does not claim state-of-the-art language modeling performance on standard benchmark leaderboards (e.g. MBPP, HumanEval). It presents a reproducible systems prototype exploring whether a hybrid recurrent/attention decoder can reduce active VRAM footprint and compute requirements on consumer hardware during training.

---

## 1. Project Overview

This repository contains the code, baseline telemetry, and LaTeX preprint source for **SamatNext-CL**, a hybrid GQA-DeltaNet language model designed for efficient, low-memory training on consumer hardware. 

The architecture alternates between:
*   **Differential Attention (Grouped Query Attention - GQA)**: For global context and retrieval.
*   **Gated DeltaNet Recurrence**: For linear-time sequential state compression.

We benchmark SamatNext-CL against a parameter-matched **Vanilla-GPT** baseline under the exact same hardware constraints and local environment.

---

## 2. Model Specifications

| Attribute | Vanilla-GPT (Baseline) | SamatNext-CL (Hybrid) |
| :--- | :---: | :---: |
| **Unique Parameters** | **561,642,904** | **432,517,324** |
| **Number of Layers** | 24 | 24 |
| **Hidden Size ($d_{model}$)** | 1024 | 1024 |
| **MLP / FFN Size** | 5,554 (SwiGLU) | 3,584 (Top-1 Sparse MoE) |
| **State Dimension** | *N/A (Softmax)* | `d_head=64` (recurrent state $S \in \mathbb{R}^{64 \times 64}$) |
| **Layer Interleaving** | 100% GQA Attention | Even Layers: DeltaNet Recurrence<br>Odd Layers: Attention / MLA |
| **Word Embeddings** | Tied | Tied |

---

## 3. Core Baseline Benchmarks

These results reflect step 1,000 of the **synthetic algorithmic curriculum** benchmark run on a single local GPU setup:

| Model | Loss | Perplexity (PPL) | Throughput (tokens/s) | Active VRAM Allocated | VRAM Memory Savings |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Vanilla-GPT** (Transformer) | 1.1141 | 3.0468 | 5,907.80 | 5,480.87 MiB | *Baseline* |
| **SamatNext-CL** (Hybrid) | **0.9904** | **2.6923** | **6,178.88** | **3,512.81 MiB** | **35.91% lower VRAM** |

*Note: Telemetry matrix logs are archived under `results/curriculum_experiment/telemetry_matrix.csv`.*

---

## 4. Key Systems Innovations

1.  **Low-Bit FP8 Quantization**: Converted **62.45% of parameters** (attentions QKV, SwiGLU expert gates, and output projections) to **TorchAO Float8Linear (E4M3)**, reducing initial parameter memory by **31.68%** (from 2,144.7 MiB down to 1,465.2 MiB).
2.  **Streamed Vocabulary-Blocked Loss**: Replaced standard Cross-Entropy with a custom online log-sum-exp streamed projection (`samatnext_fused_ops.py`) that processes the vocabulary in chunks of `block_vocab=4096`. This eliminates the need to materialize the massive `[16384 tokens, 50304 vocabulary]` logit tensor in VRAM, saving **~3.3 GB of VRAM** and accelerating the loss step by **2.85x**.
3.  **Fused DeltaNet Triton Kernels**: Integrates a fused chunk-wise linear RNN state update to accelerate recurrent steps inside compiler blocks under `torch.compile`.

---

## 5. Reproduction Instructions

### Prerequisites
Ensure you have a PyTorch environment with Triton and CUDA support installed:
```bash
pip install -r requirements.txt
```

### Step 1: Generate Synthetic Algorithmic Curriculum Shards
Create the mock curriculum token data needed to run the benchmarking loops:
```bash
python make_dummy_data.py --num-shards 5 --tokens-per-shard 10000000 --out-dir data/
```

### Step 2: Run Curriculum Benchmarks
Run both architectures through the 1,000-step curriculum benchmark run:
```bash
# Run SamatNext-CL Hybrid
python run_curriculum_experiment.py --model samatnext --steps 1000 --data-dir data/ --out-dir results/curriculum_experiment/

# Run Vanilla-GPT Transformer
python run_curriculum_experiment.py --model vanilla --steps 1000 --data-dir data/ --out-dir results/curriculum_experiment/
```
The benchmark will write live telemetry (throughput, allocated VRAM, loss, etc.) to `results/curriculum_experiment/telemetry_matrix.csv` for comparison against the baseline figures.
