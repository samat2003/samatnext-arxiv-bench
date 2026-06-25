# SamatNext-CL: A Consumer-GPU Hybrid Language Model Architecture for Low-Memory Training

> **Systems / Architecture Preprint**  
> **GitHub Repository:** [https://github.com/samat2003/samatnext-arxiv-bench](https://github.com/samat2003/samatnext-arxiv-bench)  
> **Target Hardware Class:** Consumer-grade Laptop GPUs (12GB VRAM class, e.g., NVIDIA RTX 4070 / 5070 Laptop GPUs).  
>
> **Important Framing:** This work presents a preliminary consumer-GPU systems prototype. We do not claim state-of-the-art language modeling or coding benchmark performance. The goal is to test whether a hybrid recurrent/attention decoder can reduce memory and analytical training FLOPs under a reproducible local benchmark on a 12GB laptop GPU.

---

## 1. Environment Specifications

All benchmarks were executed locally under a shared Windows/WSL environment. The hardware and software configuration of the local environment is detailed below:

| Field | Value |
| :--- | :--- |
| **GPU** | NVIDIA GeForce RTX 5070 Ti Laptop GPU |
| **VRAM / Memory Limit** | 12,227 MiB |
| **Compute Capability** | 12.0 (`sm_120`) |
| **CUDA Driver / Runtime** | 610.47 / 12.8 |
| **PyTorch Version** | 2.12.0.dev20260408+cu128 |
| **TorchAO Version** | 0.17.0 |
| **Operating System** | Windows / WSL2 (Ubuntu 24.04) |
| **Training Precision** | BF16 autocast; optional FP8 conversion utilities included but not used in the default benchmark |
| **Compiler Mode** | `max-autotune-no-cudagraphs` |

---

## 2. Comparative Model Specifications

| Attribute | Vanilla-GPT (Baseline) | SamatNext-CL (Hybrid) |
| :--- | :---: | :---: |
| **Unique Parameters** | **561,642,904** | **432,517,324** |
| **Number of Layers** | 24 | 24 |
| **Hidden Size ($d_{model}$)** | 1024 | 1024 |
| **MLP / FFN Size** | 5,554 (SwiGLU) | 3,584 (Top-1 Sparse MoE) |
| **State Dimension** | *N/A (Softmax)* | `d_head=64` (recurrent state $S \in \mathbb{R}^{64 \times 64}$) |
| **Layer Interleaving** | 100% dense causal SDPA attention | Even Layers: DeltaNet Recurrence<br>Odd Layers: Attention / MLA |
| **Word Embeddings** | Tied | Tied |

### Core Parameter and Compute Clarification
SamatNext-CL and Vanilla-GPT are both sub-billion-parameter models, but they differ in active compute. Vanilla-GPT uses dense Transformer blocks with most parameters active at every token. SamatNext-CL uses alternating recurrent/attention blocks and Top-1 routed feed-forward selection. In the current benchmark implementation, routing is represented through a masked expert computation path; therefore, the table reports architecture-level routed compute structure rather than a fully optimized sparse-execution kernel.

To define this clearly:
*   **Total stored parameters**: All parameters present in the model checkpoint (weights stored on disk).
*   **Active parameters/token**: Parameters actually used in the token's executed computational path.
*   **Analytical Active-Path FLOPs/token**: Architecture-specific active-path estimated compute per token modeled mathematically.

---

## 3. Core Baseline Benchmarks

These results reflect Step 1,000 of the **synthetic algorithmic curriculum** benchmark run on a single local GPU setup:

| Model | Total Params | Active Compute | Analytical Active-Path FLOPs/token | Throughput (tok/s) | Allocated VRAM | Final Loss (PPL) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Vanilla-GPT** (Baseline) | 561.6M | dense | $3.52 \times 10^9$ | 5,907.80 | 5,480.87 MiB | 1.1141 (3.047) |
| **SamatNext-CL** (Hybrid) | **432.5M** | **routed/masked** | **$\mathbf{1.47 \times 10^9}$** | **6,178.88** | **3,512.81 MiB** | **0.9904 (2.692)** |

*Note: FLOP values are architecture-specific analytical active-path estimates. They do not claim profiler-measured executed FLOPs for the current PyTorch implementation. SamatNext-CL achieves **35.91% lower peak allocated VRAM in our local synthetic benchmark configuration** compared to the baseline.*

---

## 4. Raw Telemetry Matrix Logs

Archive of the logged metrics comparison recorded during the 1,000-step training loop:

### SamatNext-CL Hybrid
*   Step 1: Loss = `11.0363`, PPL = `62084.76`, Throughput = `1253.74 tokens/s`, VRAM = `4315.68 MiB`, Analytical FLOPs/s = `1.84 TFLOP/s`
*   Step 100: Loss = `1.0053`, PPL = `2.7327`, Throughput = `6616.07 tokens/s`, VRAM = `3512.81 MiB`, Analytical FLOPs/s = `9.74 TFLOP/s`
*   Step 500: Loss = `0.9946`, PPL = `2.7036`, Throughput = `6604.20 tokens/s`, VRAM = `3512.81 MiB`, Analytical FLOPs/s = `9.72 TFLOP/s`
*   Step 1000: Loss = `0.9904`, PPL = `2.6923`, Throughput = `6178.88 tokens/s`, VRAM = `3512.81 MiB`, Analytical FLOPs/s = `9.09 TFLOP/s`

### Vanilla-GPT Transformer
*   Step 1: Loss = `11.0307`, PPL = `61740.59`, Throughput = `3487.48 tokens/s`, VRAM = `7841.21 MiB`, Analytical FLOPs/s = `12.28 TFLOP/s`
*   Step 100: Loss = `1.1314`, PPL = `3.1001`, Throughput = `5968.69 tokens/s`, VRAM = `5480.87 MiB`, Analytical FLOPs/s = `21.02 TFLOP/s`
*   Step 500: Loss = `1.0781`, PPL = `2.9391`, Throughput = `6798.62 tokens/s`, VRAM = `5480.87 MiB`, Analytical FLOPs/s = `23.94 TFLOP/s`
*   Step 1000: Loss = `1.1141`, PPL = `3.0468`, Throughput = `5907.80 tokens/s`, VRAM = `5480.87 MiB`, Analytical FLOPs/s = `20.80 TFLOP/s`

---

## 5. Limitations

*   **Preliminary Systems Prototype**: This is a preliminary systems/architecture prototype.
*   **Synthetic Local Benchmarks**: Benchmarks are synthetic and local, and are constrained to a personal laptop GPU environment.
*   **No Official Coding Benchmarks**: Tested on synthetic algorithmic curricula to analyze systems and memory bounds. No HumanEval or MBPP scores are claimed.
*   **Analytical Estimator**: FLOP values represent architecture-specific analytical active-path estimates rather than profiler-measured operator counts.
*   **Mismatched Baselines**: The Vanilla-GPT baseline has a larger total parameter count, while SamatNext uses sparse/routed active computation per token.

---

## 6. Reproduction Instructions

### Prerequisites
Install the dependencies inside your environment:
```bash
pip install -r requirements.txt
```

### Optional: Generate Dummy Binary Shards

`run_curriculum_experiment.py` generates the synthetic Micro-Lisp curriculum internally and does not require external data shards. The following script is kept only for archived production-style dummy-data validation:

```bash
python make_dummy_data.py
```

### Step 2: Run Curriculum Benchmarks
Run both architectures through the 1,000-step curriculum benchmark run:
```bash
# Run both models, Stage 1, seq_len 512
python run_curriculum_experiment.py \
  --stage 1 \
  --seq-len 512 \
  --steps 1000 \
  --model both

# Or separately:
python run_curriculum_experiment.py \
  --stage 1 \
  --seq-len 512 \
  --steps 1000 \
  --model SamatNext-CL

python run_curriculum_experiment.py \
  --stage 1 \
  --seq-len 512 \
  --steps 1000 \
  --model Vanilla-GPT
```
The benchmark will write live telemetry (throughput, allocated VRAM, loss, PPL) to `results/curriculum_experiment/telemetry_matrix.csv`.
