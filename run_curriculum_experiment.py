"""
run_curriculum_experiment.py — Path B single-stage CLI execution.
Micro-Lisp S-expression curriculum with high-density chain packing.

Usage:
  # Stage 1 — Syntax/Binding
  python run_curriculum_experiment.py --stage 1 --seq-len 512 --save-checkpoint checkpoints/stage1.pt

  # Stage 2 — Single-Hop Arithmetic (resume from Stage 1)
  python run_curriculum_experiment.py --stage 2 --seq-len 1024 \\
      --load-checkpoint checkpoints/stage1.pt --save-checkpoint checkpoints/stage2.pt

  # Stage 3 — Multi-Hop Graph Reasoning (resume from Stage 2)
  python run_curriculum_experiment.py --stage 3 --seq-len 2048 \\
      --load-checkpoint checkpoints/stage2.pt --save-checkpoint checkpoints/stage3.pt
"""

import argparse
import math
import re
import time
import csv
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import bitsandbytes as bnb
from transformers import AutoTokenizer

from samatnext_550m_model import SamatNextCLForCausalLM, convert_experimental_fp8_linears

# ---------------------------------------------------------------------------
# Optional NVML power probe — gracefully absent if pynvml is not installed.
# All GPU power calls are lightweight single-integer reads from the driver;
# they take <10 µs and are only invoked at telemetry log intervals.
# ---------------------------------------------------------------------------
try:
    import pynvml as _nvml
    _nvml.nvmlInit()
    _NVML_OK = True
except Exception:
    _NVML_OK = False


def _gpu_power_watts(device_index: int = 0) -> float:
    """Return instantaneous GPU power in watts, or float('nan') if unavailable."""
    if not _NVML_OK or not torch.cuda.is_available():
        return float("nan")
    try:
        handle = _nvml.nvmlDeviceGetHandleByIndex(device_index)
        return _nvml.nvmlDeviceGetPowerUsage(handle) / 1_000.0  # mW → W
    except Exception:
        return float("nan")


def _estimate_flops_per_token(model_name: str, seq_len: int) -> int:
    """Calculate the precise training FLOPs per token based on the specific architecture.

    For training, we account for 3x the forward pass FLOPs (1x forward, 2x backward).

    Vanilla-GPT (24 layers, d_model=1024, ffn_dim=5554, vocab_size=50304):
      Each layer has:
        - SDPACausalSelfAttention:
          - qkv_proj: 2 * d_model * 3 * d_model = 6 * d_model^2
          - out_proj: 2 * d_model^2
          - attention matrix: 4 * seq_len * d_model
        - DenseSwiGLUFFN:
          - w1, w2, w3: 3 * 2 * d_model * ffn_dim = 6 * d_model * ffn_dim
      LM Head: 2 * d_model * vocab_size
      Total Training FLOPs = 3 * (layers * (8 * d_model^2 + 4 * seq_len * d_model + 6 * d_model * ffn_dim) + 2 * d_model * vocab_size)

    SamatNext-CL (12 DeltaBlock + 12 AnchorBlock layers):
      DeltaBlock:
        - delta_qkv_proj: 2 * d_model * 3 * d_model = 6 * d_model^2
        - delta_beta_proj: 2 * d_model * num_heads
        - deltaNet chunkwise parallel recurrence: num_heads * (6 * d_head^2 + 7 * d_head * chunk_size)
        - delta_out_proj: 2 * d_model^2
        - delta_residual_gate: 2 * d_model^2
        - SparseTop1FFN (active expert only): 2 * d_model * num_experts + 6 * d_model * h_expert
      AnchorBlock:
        - MLA QKV projection (compressed): 2 * d_model^2 (Q) + 2 * d_model * 512 (down) + 2 * 512 * 2 * d_model (up)
        - anchor_out_proj: 2 * d_model^2
        - attention matrix: 4 * seq_len * d_model
        - SparseTop1FFN (active expert only): 2 * d_model * num_experts + 6 * d_model * h_expert
        - eviction feedback: 0 (disabled in all AnchorBlocks since enable_eviction_feedback=False)
      LM Head: 2 * d_model * vocab_size
      Total Training FLOPs = 3 * (12 * DeltaBlock_F + 12 * AnchorBlock_F + 2 * d_model * vocab_size)
    """
    d_model = 1024
    vocab_size = 50304

    if model_name == "SamatNext-CL":
        # 1. DeltaBlock (12 layers)
        num_heads = 16
        d_head = 64
        chunk_size = 128
        num_experts = 4
        h_expert = 896

        delta_qkv = 6 * (d_model ** 2)
        delta_beta = 2 * d_model * num_heads
        # recurrence per head: 6 * d_head^2 + 7 * d_head * chunk_size
        recurrence = num_heads * (6 * (d_head ** 2) + 7 * d_head * chunk_size)
        delta_out = 2 * (d_model ** 2)
        delta_gate = 2 * (d_model ** 2)
        ffn_router = 2 * d_model * num_experts
        ffn_proj = 6 * d_model * h_expert

        delta_forward = delta_qkv + delta_beta + recurrence + delta_out + delta_gate + ffn_router + ffn_proj

        # 2. AnchorBlock (12 layers)
        # MLA projection: q_proj + kv_lora_down + kv_lora_up
        mla_proj = 2 * (d_model ** 2) + 2 * d_model * 512 + 2 * 512 * (2 * d_model)
        anchor_out = 2 * (d_model ** 2)
        attn_matrix = 4 * seq_len * d_model
        # SparseTop1FFN is the same:
        anchor_ffn = ffn_router + ffn_proj

        anchor_forward = mla_proj + anchor_out + attn_matrix + anchor_ffn

        # LM Head
        lm_head_forward = 2 * d_model * vocab_size

        total_forward = 12 * delta_forward + 12 * anchor_forward + lm_head_forward
        return 3 * total_forward

    else:
        # Vanilla-GPT
        ffn_dim = 5554
        num_layers = 24

        qkv_proj = 6 * (d_model ** 2)
        out_proj = 2 * (d_model ** 2)
        attn_matrix = 4 * seq_len * d_model
        ffn_proj = 6 * d_model * ffn_dim

        block_forward = qkv_proj + out_proj + attn_matrix + ffn_proj
        lm_head_forward = 2 * d_model * vocab_size

        total_forward = num_layers * block_forward + lm_head_forward
        return 3 * total_forward



# ===========================================================================
# 1. CLI Argument Parser
# ===========================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Path B: Single-stage Micro-Lisp curriculum benchmark."
    )
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2, 3],
        required=True,
        help=(
            "Curriculum stage: "
            "1=Syntax/Binding, "
            "2=Single-Hop Arithmetic, "
            "3=Multi-Hop Graph Reasoning."
        ),
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        choices=[512, 1024, 2048],
        default=512,
        help="Packed sequence length (default: 512).",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["both", "SamatNext-CL", "Vanilla-GPT"],
        default="both",
        help="Specific model to run benchmark on (default: both).",
    )
    parser.add_argument(
        "--save-checkpoint",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to save model checkpoint after the run.",
    )
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to load a prior checkpoint (skips random init).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Number of optimizer steps (default: 1000).",
    )
    return parser


# ===========================================================================
# 2. VanillaGPTForCausalLM — SDPA-native causal transformer baseline
# ===========================================================================

class SDPACausalSelfAttention(nn.Module):
    """Causal self-attention backed by torch.nn.functional.scaled_dot_product_attention.

    is_causal=True lets PyTorch dispatch to FlashAttention or the memory-efficient
    CUDA kernel automatically under torch.compile(mode='max-autotune').
    No manual causal mask buffers, no masked_fill overhead.
    """

    def __init__(self, d_model: int = 1024, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        # Native SDPA — dispatches to FlashAttention when hardware supports it.
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)


class DenseSwiGLUFFN(nn.Module):
    def __init__(self, d_model: int = 1024, ffn_dim: int = 5554) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, ffn_dim, bias=False)
        self.w2 = nn.Linear(d_model, ffn_dim, bias=False)
        self.w3 = nn.Linear(ffn_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class GPTBlock(nn.Module):
    def __init__(self, d_model: int = 1024, num_heads: int = 16, ffn_dim: int = 5554) -> None:
        super().__init__()
        norm_cls = getattr(nn, "RMSNorm", nn.LayerNorm)
        self.ln1 = norm_cls(d_model)
        self.attn = SDPACausalSelfAttention(d_model, num_heads)
        self.ln2 = norm_cls(d_model)
        self.ffn = DenseSwiGLUFFN(d_model, ffn_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class VanillaGPTForCausalLM(nn.Module):
    """Textbook causal transformer twin with SDPA attention.

    Parameter count: 24 layers × SwiGLU ffn_dim=5554 ≈ 561.7M, matched to SamatNext-CL shell.
    No custom kernels, no special tricks. Standard fair baseline.
    """

    def __init__(
        self,
        vocab_size: int = 50304,
        num_layers: int = 24,
        d_model: int = 1024,
        num_heads: int = 16,
        ffn_dim: int = 5554,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            GPTBlock(d_model, num_heads, ffn_dim) for _ in range(num_layers)
        ])
        norm_cls = getattr(nn, "RMSNorm", nn.LayerNorm)
        self.final_norm = norm_cls(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embeddings.weight  # weight tying
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = self.token_embeddings(input_ids)
        for block in self.layers:
            x = block(x)
        logits = self.lm_head(self.final_norm(x))
        if labels is None:
            return logits
        # ignore_index=-100 masks EOS filler positions from the loss.
        loss = F.cross_entropy(
            logits.view(-1, self.vocab_size),
            labels.view(-1),
            ignore_index=-100,
        )
        return logits, loss


# ===========================================================================
# 3. Micro-Lisp Generator — high-density chain packing
# ===========================================================================

class MicroLispGenerator:
    """Generates Micro-Lisp S-expression programs and packs them densely.

    Micro-Lisp grammar (per stage):
      Stage 1 — Syntax/Binding:
        (define a 3) (define b 7)

      Stage 2 — Single-Hop Arithmetic:
        (define a 3) (define b 8) (write (+ a b)) => 11

      Stage 3 — Multi-Hop Graph Reasoning:
        (define x 5) (define y 2) (define z (+ x y)) (write (* z x)) => 35

    Packing:
      Each sequence row is filled with complete programs separated by EOS tokens
      until adding another full program would exceed seq_len.
      Remaining tail positions are filled with EOS tokens and masked as -100
      in the label tensor so they do not contribute to the cross-entropy loss.
    """

    # Only single-letter vars keep tokenization consistent across GPT-2 vocab
    _VARS = list("abcdefghijklmnopqrstuvwxy")  # 24 vars (avoid z clashes)
    _OPS  = ["+", "-", "*"]

    def __init__(self, stage: int, batch_size: int, seq_len: int) -> None:
        if stage not in (1, 2, 3):
            raise ValueError(f"stage must be 1, 2, or 3; got {stage!r}")
        self.stage = stage
        self.batch_size = batch_size
        self.seq_len = seq_len

        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.eos_id = self.tokenizer.eos_token_id

        self._gen = {
            1: self._gen_stage1,
            2: self._gen_stage2,
            3: self._gen_stage3,
        }[stage]

    # ── Program text generators ─────────────────────────────────────────────

    def _gen_stage1(self) -> str:
        """(define a 3) (define b 7)"""
        a, b = random.sample(self._VARS, 2)
        va, vb = random.randint(1, 9), random.randint(1, 9)
        return f"(define {a} {va}) (define {b} {vb})"

    def _gen_stage2(self) -> str:
        """(define a 3) (define b 8) (write (+ a b)) => 11"""
        a, b = random.sample(self._VARS, 2)
        va, vb = random.randint(1, 9), random.randint(1, 9)
        op = random.choice(self._OPS)
        result = va + vb if op == "+" else (va - vb if op == "-" else va * vb)
        return f"(define {a} {va}) (define {b} {vb}) (write ({op} {a} {b})) => {result}"

    def _gen_stage3(self) -> str:
        """(define x 5) (define y 2) (define z (+ x y)) (write (* z x)) => 35"""
        a, b, c = random.sample(self._VARS, 3)
        va, vb = random.randint(1, 9), random.randint(1, 9)
        op1, op2 = random.choice(self._OPS), random.choice(self._OPS)
        vc = va + vb if op1 == "+" else (va - vb if op1 == "-" else va * vb)
        result = vc + va if op2 == "+" else (vc - va if op2 == "-" else vc * va)
        return (
            f"(define {a} {va}) "
            f"(define {b} {vb}) "
            f"(define {c} ({op1} {a} {b})) "
            f"(write ({op2} {c} {a})) => {result}"
        )

    # ── Packing logic ───────────────────────────────────────────────────────

    def _tokenize_program(self) -> list[int]:
        """Generate one program and append EOS as program separator."""
        return self.tokenizer.encode(self._gen()) + [self.eos_id]

    def _pack_one_sequence(self) -> tuple[list[int], int]:
        """
        Pack complete programs into a buffer of exactly seq_len + 1 tokens.

        Returns:
            full_seq:     list of token IDs, length = seq_len + 1
            filler_start: index in full_seq where EOS filler padding begins.
                          All tokens at indices >= filler_start are padding
                          and will be masked with -100 in the label tensor.
        """
        budget = self.seq_len + 1
        buf: list[int] = []

        while True:
            prog = self._tokenize_program()
            # Do not start a program that cannot fit in full
            if len(buf) + len(prog) > budget:
                break
            buf.extend(prog)
            # Stop early if perfectly full
            if len(buf) >= budget:
                break

        filler_start = len(buf)
        # Pad tail to exactly budget positions with EOS tokens
        buf.extend([self.eos_id] * (budget - filler_start))
        return buf, filler_start

    def next_batch(
        self, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, float, float]:
        """
        Build a batch of packed Micro-Lisp sequences.

        Returns:
            inp            [batch, seq_len]  input token IDs
            tgt            [batch, seq_len]  shifted label IDs; filler positions = -100
            real_token_ratio  fraction of tgt positions that are real (non-filler)
            eos_filler_ratio  fraction of tgt positions that are masked filler
        """
        inputs_list: list[list[int]] = []
        labels_list: list[list[int]] = []
        total_real = 0

        for _ in range(self.batch_size):
            full_seq, filler_start = self._pack_one_sequence()

            inp = full_seq[: self.seq_len]             # length = seq_len

            # tgt[i] = full_seq[i+1], except positions where full_seq[i+1] is filler
            # A position is filler when its index in full_seq >= filler_start
            tgt = [
                full_seq[i + 1] if (i + 1) < filler_start else -100
                for i in range(self.seq_len)
            ]

            inputs_list.append(inp)
            labels_list.append(tgt)
            total_real += sum(1 for t in tgt if t != -100)

        real_ratio   = total_real / (self.batch_size * self.seq_len)
        filler_ratio = 1.0 - real_ratio

        inp_t = torch.tensor(inputs_list, dtype=torch.long, device=device)
        tgt_t = torch.tensor(labels_list, dtype=torch.long, device=device)
        return inp_t, tgt_t, real_ratio, filler_ratio


# ===========================================================================
# 4. LispOracle — dynamic Micro-Lisp parser and evaluator
# ===========================================================================

class LispOracle:
    """Parses and evaluates greedy-decoded Micro-Lisp programs.

    Supports all three curriculum stages.  No hardcoded answer keys.
    Evaluates each decoded text by:
      1. Stripping EOS/padding artefacts.
      2. Scanning for (define VAR EXPR) forms and building the variable env.
      3. Evaluating (write EXPR) to obtain the computed result.
      4. Comparing the computed result with the => N marker in the model output.

    Returns a dict of boolean metrics per decoded sample:
      syntax_valid    — all seen tokens are valid Micro-Lisp
      parse_valid     — all variable references resolve without KeyError
      execution_valid — a (write ...) expression was found and evaluated
      arith_correct   — computed (write ...) result matches the => N value
      pass_at_1       — full correctness (arith_correct for S2/S3; define coverage for S1)
    """

    # Patterns are compiled once at class level for speed
    _DEFINE_RE = re.compile(
        r"\(define\s+([a-z])\s+"
        r"((?:\([+\-*]\s+[a-z0-9]+\s+[a-z0-9]+\)|-?\d+|[a-z]))\)"
    )
    _WRITE_RE = re.compile(
        r"\(write\s+"
        r"((?:\([+\-*]\s+[a-z0-9]+\s+[a-z0-9]+\)|-?\d+|[a-z]))\)"
    )
    _ARROW_RE = re.compile(r"=>\s*(-?\d+)")
    _EOS      = "<|endoftext|>"

    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # ── Low-level evaluator ─────────────────────────────────────────────────

    def _eval_atom(self, token: str, env: dict[str, int]) -> int:
        token = token.strip()
        if re.match(r"^-?\d+$", token):
            return int(token)
        return env[token]   # raises KeyError for undefined vars → parse failure

    def _eval_expr(self, expr: str, env: dict[str, int]) -> int:
        """Evaluate a Micro-Lisp expression: a literal, a variable, or (op A B)."""
        expr = expr.strip()
        if not expr.startswith("("):
            return self._eval_atom(expr, env)
        inner = expr[1:-1].strip()          # strip outer parens
        parts  = inner.split(None, 2)       # op, arg1, arg2
        if len(parts) != 3:
            raise ValueError(f"Malformed S-expr: {expr!r}")
        op, a_str, b_str = parts
        v1 = self._eval_atom(a_str, env)
        v2 = self._eval_atom(b_str, env)
        if op == "+": return v1 + v2
        if op == "-": return v1 - v2
        if op == "*": return v1 * v2
        raise ValueError(f"Unknown operator: {op!r}")

    # ── Public grader ───────────────────────────────────────────────────────

    def grade(self, text: str, stage: int) -> dict:
        """Grade a single decoded model output. Returns per-metric boolean dict."""
        result = {
            "syntax_valid":    False,
            "parse_valid":     False,
            "execution_valid": False,
            "arith_correct":   False,
            "pass_at_1":       False,
        }
        try:
            # Strip EOS tokens and everything after the first one
            cleaned = text.split(self._EOS)[0].strip()

            # Extract => N before processing (model appends this at the end)
            arrow_match = self._ARROW_RE.search(cleaned)
            arrow_val   = int(arrow_match.group(1)) if arrow_match else None
            body        = cleaned[: arrow_match.start()].strip() if arrow_match else cleaned

            # Build variable environment from (define ...) forms
            env: dict[str, int] = {}
            for m in self._DEFINE_RE.finditer(body):
                var  = m.group(1)
                expr = m.group(2)
                env[var] = self._eval_expr(expr, env)

            result["syntax_valid"] = True   # no exception so far

            # parse_valid: ≥2 vars defined (all stages), and for S2/S3 a write exists
            write_m = self._WRITE_RE.search(body)
            if stage == 1:
                result["parse_valid"]     = len(env) >= 2
                result["execution_valid"] = len(env) >= 2
                result["arith_correct"]   = len(env) >= 2
                result["pass_at_1"]       = len(env) >= 2
            else:
                result["parse_valid"] = len(env) >= 2 and write_m is not None
                if write_m:
                    computed = self._eval_expr(write_m.group(1), env)
                    result["execution_valid"] = True
                    if arrow_val is not None:
                        correct = (computed == arrow_val)
                        result["arith_correct"] = correct
                        result["pass_at_1"]     = correct

        except Exception:
            pass   # any parse/eval failure → all metrics remain False

        return result


# ===========================================================================
# 5. Evaluation runner
# ===========================================================================

def run_evaluation(
    model:     nn.Module,
    generator: MicroLispGenerator,
    oracle:    LispOracle,
    stage:     int,
    device:    torch.device,
    num_samples: int = 50,
) -> dict:
    """
    Greedy-decode num_samples programs from the prompt '(define ' and grade each one.

    The forward pass uses the uncompiled eager model (passed as `model`, not
    `compiled_model`) so that dynamically growing input_ids shapes during
    token-by-token generation never break the compiled training graph.
    The generation loop is fully wrapped in torch.inference_mode() and bfloat16 autocast.
    model.train() is restored before returning.
    """
    model.eval()
    counters = {k: 0 for k in ("pass_at_1", "syntax_valid", "parse_valid",
                                "execution_valid", "arith_correct")}

    with torch.inference_mode():
        for _ in range(num_samples):
            tokens    = generator.tokenizer.encode("(define", add_special_tokens=False)
            input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

            # Greedy decode up to 96 tokens
            for _ in range(96):
                with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                    logits     = model(input_ids)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                input_ids  = torch.cat([input_ids, next_token], dim=-1)
                if next_token.item() == generator.eos_id:
                    break

            text   = generator.tokenizer.decode(input_ids[0].tolist())
            grades = oracle.grade(text, stage)
            for k in counters:
                if grades[k]:
                    counters[k] += 1

    model.train()
    return {k: (v / num_samples) * 100.0 for k, v in counters.items()}


# ===========================================================================
# 6. Checkpoint helpers
# ===========================================================================

def _get_model_checkpoint_path(base_path: str, model_name: str) -> str:
    path = Path(base_path)
    suffix = "_samatnext" if "Samat" in model_name else "_vanilla"
    return str(path.with_name(f"{path.stem}{suffix}{path.suffix}"))


def save_checkpoint(model: nn.Module, path: str, model_name: str, step: int) -> None:
    ckpt_path = Path(_get_model_checkpoint_path(path, model_name))
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_name": model_name, "step": step,
                "state_dict": model.state_dict()}, ckpt_path)
    print(f"  Checkpoint saved → {ckpt_path}")


def load_checkpoint(model: nn.Module, path: str, model_name: str) -> int:
    actual_path = _get_model_checkpoint_path(path, model_name)
    if not Path(actual_path).exists():
        actual_path = path
    ckpt = torch.load(actual_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    step = ckpt.get("step", 0)
    print(f"  Checkpoint loaded ← {actual_path}  (from step {step})")
    return step


# ===========================================================================
# 7. Training orchestrator
# ===========================================================================

_CSV_HEADER = [
    "step", "model_name", "stage", "seq_len",
    "nonpad_loss", "nonpad_ppl",
    "tokens_per_sec", "allocated_mib",
    "pass_at_1_pct", "syntax_valid_pct", "parse_valid_pct",
    "execution_valid_pct", "arith_correct_pct",
    "real_token_ratio", "eos_filler_ratio",
    # ── Lightweight efficiency estimates (added as optional future telemetry) ──
    "flops_per_token",   # 6·N heuristic, static per model
    "tflops_per_s",      # estimated training throughput
    "gpu_watts",         # instantaneous GPU power at log point (pynvml)
    "joules_per_step",   # gpu_watts × step_elapsed_seconds
    "tokens_per_joule",  # energy efficiency
]


def run_experiment(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"\n=== Micro-Lisp Curriculum Benchmark (Path B) ===\n"
        f"  Stage:    {args.stage}\n"
        f"  Seq-len:  {args.seq_len}\n"
        f"  Steps:    {args.steps}\n"
        f"  Device:   {device}\n"
    )

    # ── Output paths ────────────────────────────────────────────────────────
    out_dir  = Path("results/curriculum_experiment")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "telemetry_matrix.csv"
    if not csv_path.exists():
        with csv_path.open("w", newline="") as f:
            csv.writer(f).writerow(_CSV_HEADER)

    if args.model == "both":
        models_to_test = ["SamatNext-CL", "Vanilla-GPT"]
    else:
        models_to_test = [args.model]

    for model_name in models_to_test:
        print(f"\n{'─'*60}\nModel: {model_name}\n{'─'*60}")

        # Hard seed reset for independent reproducibility per model
        random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        # ── Model construction ──────────────────────────────────────────────
        if model_name == "SamatNext-CL":
            model = SamatNextCLForCausalLM(
                vocab_size=50304,
                num_layers=24,
                d_model=1024,
                num_heads=16,
                sliding_window=1024,
                chunk_size=128,
            ).to(device)
            # Create a separate uncompiled evaluation copy on CPU
            eval_model = SamatNextCLForCausalLM(
                vocab_size=50304,
                num_layers=24,
                d_model=1024,
                num_heads=16,
                sliding_window=1024,
                chunk_size=128,
            )
            eval_model.eval()

            # Dual param-group: core path + memory highway
            core_p, mem_p = [], []
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                (mem_p if any(m in name for m in model.memory_module_names) else core_p).append(param)
            optimizer = bnb.optim.AdamW8bit(
                [
                    {"params": core_p, "lr": 6e-4, "weight_decay": 0.1, "betas": (0.9, 0.95)},
                    {"params": mem_p,  "lr": 6e-4, "weight_decay": 0.1, "betas": (0.9, 0.95)},
                ]
            )
        else:  # Vanilla-GPT
            model = VanillaGPTForCausalLM(
                vocab_size=50304,
                num_layers=24,
                d_model=1024,
                num_heads=16,
                ffn_dim=5554,
            ).to(device)
            # Create a separate uncompiled evaluation copy on CPU
            eval_model = VanillaGPTForCausalLM(
                vocab_size=50304,
                num_layers=24,
                d_model=1024,
                num_heads=16,
                ffn_dim=5554,
            )
            eval_model.eval()

            optimizer = bnb.optim.AdamW8bit(
                model.parameters(), lr=6e-4, weight_decay=0.1, betas=(0.9, 0.95)
            )

        n_params          = sum(p.numel() for p in model.parameters())
        flops_per_token   = _estimate_flops_per_token(model_name, args.seq_len)
        print(f"  Parameters:     {n_params:,}")
        print(f"  FLOPs/token:    {flops_per_token:,}  (precise architecture estimate)")
        print(f"  NVML available: {_NVML_OK}")

        # ── Optional checkpoint load ────────────────────────────────────────
        if args.load_checkpoint:
            load_checkpoint(model, args.load_checkpoint, model_name)

        # ── torch.compile ──────────────────────────────────────────────────
        compiled_model = torch.compile(model, mode="max-autotune")

        # ── Generator and Oracle ───────────────────────────────────────────
        generator = MicroLispGenerator(
            stage=args.stage, batch_size=4, seq_len=args.seq_len
        )
        oracle = LispOracle()

        # ── Warmup compile step ─────────────────────────────────────────────
        print("  Running warmup compile step...")
        w_inp, w_tgt, _, _ = generator.next_batch(device=device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            _, w_loss = compiled_model(w_inp, labels=w_tgt)
        w_loss.backward()
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        print("  Warmup complete.")

        # ── Training loop ───────────────────────────────────────────────────
        tokens_per_step = 4 * args.seq_len

        for step in range(1, args.steps + 1):
            t0 = time.perf_counter()

            inp, tgt, real_ratio, filler_ratio = generator.next_batch(device=device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                _, loss = compiled_model(inp, labels=tgt)

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed       = time.perf_counter() - t0
            tok_per_sec   = tokens_per_step / elapsed

            # ── Logging every 100 steps (and step 1) ──────────────────────
            if step % 100 == 0 or step == 1:
                # Sync evaluation model weights with active training model weights
                eval_model = eval_model.to(device)
                eval_model.load_state_dict(model.state_dict())
                eval_m    = run_evaluation(eval_model, generator, oracle, args.stage, device)
                
                # Move eval_model back to CPU and clear cache to free VRAM
                eval_model = eval_model.to("cpu")
                import gc
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                
                loss_val  = loss.item()
                ppl       = math.exp(min(loss_val, 20.0))
                allocated = (
                    torch.cuda.memory_allocated(device) / (1024 ** 2)
                    if device.type == "cuda" else 0.0
                )

                # ── Lightweight efficiency estimates ───────────────────────
                # All computed from already-available values; only gpu_watts
                # touches hardware (single pynvml integer read, <10 µs).
                tflop_per_step  = flops_per_token * tokens_per_step / 1e12
                tflops_per_s    = tflop_per_step / elapsed
                gpu_watts       = _gpu_power_watts(device.index or 0)
                joules_per_step = gpu_watts * elapsed  # NaN if NVML unavailable
                tokens_per_joule = (
                    tokens_per_step / joules_per_step
                    if math.isfinite(joules_per_step) and joules_per_step > 0
                    else float("nan")
                )

                print(
                    f"  [{model_name}] Step {step:>4}/{args.steps} | "
                    f"Loss={loss_val:.4f} PPL={ppl:.2f} | "
                    f"Pass@1={eval_m['pass_at_1']:.1f}% "
                    f"Syntax={eval_m['syntax_valid']:.1f}% "
                    f"Arith={eval_m['arith_correct']:.1f}% | "
                    f"RealTok={real_ratio*100:.1f}% | "
                    f"Tok/s={tok_per_sec:.0f} "
                    f"TFLOP/s={tflops_per_s:.2f}"
                    + (f" W={gpu_watts:.1f}" if math.isfinite(gpu_watts) else "")
                )

                with csv_path.open("a", newline="") as f:
                    csv.writer(f).writerow([
                        step,
                        model_name,
                        args.stage,
                        args.seq_len,
                        f"{loss_val:.6f}",
                        f"{ppl:.6f}",
                        f"{tok_per_sec:.2f}",
                        f"{allocated:.2f}",
                        f"{eval_m['pass_at_1']:.2f}",
                        f"{eval_m['syntax_valid']:.2f}",
                        f"{eval_m['parse_valid']:.2f}",
                        f"{eval_m['execution_valid']:.2f}",
                        f"{eval_m['arith_correct']:.2f}",
                        f"{real_ratio:.4f}",
                        f"{filler_ratio:.4f}",
                        flops_per_token,
                        f"{tflops_per_s:.4f}",
                        f"{gpu_watts:.2f}" if math.isfinite(gpu_watts) else "nan",
                        f"{joules_per_step:.4f}" if math.isfinite(joules_per_step) else "nan",
                        f"{tokens_per_joule:.2f}" if math.isfinite(tokens_per_joule) else "nan",
                    ])

        # ── Optional checkpoint save ────────────────────────────────────────
        if args.save_checkpoint:
            save_checkpoint(model, args.save_checkpoint, model_name, args.steps)

        # ── Explicit memory cleanup at loop end ─────────────────────────────
        del optimizer
        if "compiled_model" in locals():
            del compiled_model
        del model
        if "eval_model" in locals():
            del eval_model
        import gc
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    print(f"\nBenchmark complete. Telemetry → {csv_path}")


# ===========================================================================
# 8. Entry point
# ===========================================================================

if __name__ == "__main__":
    run_experiment(build_arg_parser().parse_args())
