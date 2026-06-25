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
| **Training Precision** | FP8 (E4M3) via TorchAO `Float8Linear` |
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
| **Layer Interleaving** | 100% GQA Attention | Even Layers: DeltaNet Recurrence<br>Odd Layers: Attention / MLA |
| **Word Embeddings** | Tied | Tied |

---

## 3. Core Baseline Benchmarks

These results reflect Step 1,000 of the **synthetic algorithmic curriculum** benchmark run on a single local GPU setup:

| Model | Params | Analytical FLOPs/token | Throughput (tok/s) | Allocated VRAM | Final Loss (PPL) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Vanilla-GPT** (Baseline) | 561.6M | $3.52 \times 10^9$ | 5,907.80 | 5,480.87 MiB | 1.1141 (3.047) |
| **SamatNext-CL** (Hybrid) | **432.5M** | **$\mathbf{1.47 \times 10^9}$** | **6,178.88** | **3,512.81 MiB** | **0.9904 (2.692)** |

*Note: FLOP values represent architecture-specific analytical FLOP estimates. SamatNext-CL achieves **35.91% lower peak allocated VRAM in our local synthetic benchmark configuration**.*

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

*   **Scale Constraint**: Evaluated on sub-1B parameter models ($\sim$400M parameters).
*   **No Official Coding Benchmarks**: Tested on synthetic algorithmic curricula to analyze systems and memory bounds. No HumanEval or MBPP scores are claimed.
*   **Analytical Estimator**: FLOP values represent architecture-specific analytical FLOP estimates rather than measured operator-level counters.

---

## 6. Reproduction Instructions

### Prerequisites
Install the dependencies inside your environment:
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
The benchmark will write live telemetry (throughput, allocated VRAM, loss, PPL) to `results/curriculum_experiment/telemetry_matrix.csv`.
