import math
import traceback
from dataclasses import dataclass

import torch
from torch.autograd import gradcheck

from samatnext_fused_ops import fused_deltanet_chunk_metrics


PRODUCTION_VALID = False
PRODUCTION_INVALID_REASON = (
    "candidate computes q*k+v only and omits DeltaNet recurrence and beta dependence"
)

CHUNK_SIZE = 128
BF16_SEQ_LEN = 1024
GRADCHECK_EPS = 1e-6
GRADCHECK_ATOL = 1e-4
SHADOW_ATOL = 1e-3
SHADOW_RTOL = 1e-3


class TritonDeltaNetCandidate(torch.autograd.Function):
    """Autograd surface around the current samatnext_fused_ops Triton kernel.

    The current kernel takes Q, K, and V only. This wrapper accepts beta so the
    validator can compare it to the exact DeltaNet reference signature and
    expose the missing beta dependency as a gradient mismatch.
    """

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        if q.shape != k.shape or q.shape != v.shape:
            raise ValueError("q, k, and v must share shape [B, H, T, D]")
        if beta.shape != q.shape[:-1]:
            raise ValueError("beta must have shape [B, H, T]")
        ctx.save_for_backward(q, k, beta)
        return fused_deltanet_chunk_metrics(q, k, v)

    @staticmethod
    def backward(
        ctx,
        grad_out: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, beta = ctx.saved_tensors
        grad_q = grad_out * k
        grad_k = grad_out * q
        grad_v = grad_out
        grad_beta = torch.zeros_like(beta)
        return grad_q, grad_k, grad_v, grad_beta


def triton_candidate(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    return TritonDeltaNetCandidate.apply(q, k, v, beta)


def reference_recurrent_deltanet(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """Exact recurrent DeltaNet update used as the mathematical reference."""
    bsz, heads, seq_len, d_head = q.shape
    scale = 1.0 / math.sqrt(d_head)
    memory = torch.zeros(
        bsz,
        heads,
        d_head,
        d_head,
        device=q.device,
        dtype=q.dtype,
    )
    outputs = []
    for token_idx in range(seq_len):
        q_t = q[:, :, token_idx]
        k_t = k[:, :, token_idx]
        v_t = v[:, :, token_idx]
        beta_t = beta[:, :, token_idx].unsqueeze(-1)
        pred = torch.einsum("bhd,bhde->bhe", k_t, memory)
        delta = (v_t - pred) * beta_t
        memory = memory + torch.einsum("bhd,bhe->bhde", k_t, delta)
        outputs.append(torch.einsum("bhd,bhde->bhe", q_t * scale, memory))
    return torch.stack(outputs, dim=2)


@dataclass(frozen=True)
class StressCase:
    name: str
    saturated_beta: bool
    aligned_keys: bool
    q_scale: float = 0.20
    k_noise: float = 0.01
    v_scale: float = 0.20


STRESS_CASES = (
    StressCase(
        name="saturated_beta_gates",
        saturated_beta=True,
        aligned_keys=False,
    ),
    StressCase(
        name="high_condition_number_keys",
        saturated_beta=False,
        aligned_keys=True,
        k_noise=1e-4,
    ),
    StressCase(
        name="multi_chunk_error_cascade",
        saturated_beta=True,
        aligned_keys=True,
        k_noise=5e-4,
    ),
)


def make_inputs(
    case: StressCase,
    *,
    bsz: int,
    heads: int,
    seq_len: int,
    d_head: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    q = torch.randn(
        bsz,
        heads,
        seq_len,
        d_head,
        generator=generator,
        device=device,
        dtype=dtype,
    ) * case.q_scale
    v = torch.randn(
        bsz,
        heads,
        seq_len,
        d_head,
        generator=generator,
        device=device,
        dtype=dtype,
    ) * case.v_scale

    if case.aligned_keys:
        base = torch.randn(
            bsz,
            heads,
            1,
            d_head,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        base = base / base.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        noise = torch.randn(
            bsz,
            heads,
            seq_len,
            d_head,
            generator=generator,
            device=device,
            dtype=dtype,
        ) * case.k_noise
        k = base.expand(-1, -1, seq_len, -1).contiguous() + noise
    else:
        k = torch.randn(
            bsz,
            heads,
            seq_len,
            d_head,
            generator=generator,
            device=device,
            dtype=dtype,
        ) * 0.20

    if case.saturated_beta:
        beta = 0.95 + 0.049 * torch.rand(
            bsz,
            heads,
            seq_len,
            generator=generator,
            device=device,
            dtype=dtype,
        )
    else:
        beta = 0.05 + 0.90 * torch.rand(
            bsz,
            heads,
            seq_len,
            generator=generator,
            device=device,
            dtype=dtype,
        )

    return (
        q.detach().requires_grad_(True),
        k.detach().requires_grad_(True),
        v.detach().requires_grad_(True),
        beta.detach().requires_grad_(True),
    )


def run_gradcheck(device: torch.device) -> list[dict[str, str]]:
    rows = []
    for case_idx, case in enumerate(STRESS_CASES):
        q, k, v, beta = make_inputs(
            case,
            bsz=1,
            heads=1,
            seq_len=8,
            d_head=4,
            dtype=torch.float64,
            device=device,
            seed=1000 + case_idx,
        )
        gradcheck(
            triton_candidate,
            (q, k, v, beta),
            eps=GRADCHECK_EPS,
            atol=GRADCHECK_ATOL,
            raise_exception=True,
        )
        rows.append({"case": case.name, "gradcheck": "pass"})
    return rows


def clone_for_reference(
    tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(t.detach().clone().requires_grad_(True) for t in tensors)


def max_violation_report(
    case_name: str,
    var_name: str,
    ref_grad: torch.Tensor,
    tri_grad: torch.Tensor,
) -> dict[str, object]:
    abs_delta = (ref_grad.float() - tri_grad.float()).abs()
    denom = ref_grad.float().abs().clamp_min(1e-12)
    rel_delta = abs_delta / denom
    max_abs = abs_delta.max()
    flat_idx = int(max_abs.argmax().item())
    idx = tuple(int(i) for i in torch.unravel_index(torch.tensor(flat_idx), abs_delta.shape))
    token_pos = idx[2] if ref_grad.ndim >= 3 else idx[-1]
    return {
        "case": case_name,
        "var": var_name,
        "index": idx,
        "chunk": int(token_pos // CHUNK_SIZE),
        "token": int(token_pos),
        "max_abs": float(max_abs.item()),
        "max_rel": float(rel_delta.max().item()),
        "ref_value": float(ref_grad[idx].float().item()),
        "triton_value": float(tri_grad[idx].float().item()),
    }


def run_shadow_check(device: torch.device) -> list[dict[str, object]]:
    rows = []
    failures = []
    for case_idx, case in enumerate(STRESS_CASES):
        generator = torch.Generator(device=device)
        generator.manual_seed(3000 + case_idx)
        base_inputs = make_inputs(
            case,
            bsz=1,
            heads=2,
            seq_len=BF16_SEQ_LEN,
            d_head=8,
            dtype=torch.bfloat16,
            device=device,
            seed=2000 + case_idx,
        )
        ref_q, ref_k, ref_v, ref_beta = clone_for_reference(base_inputs)
        tri_q, tri_k, tri_v, tri_beta = clone_for_reference(base_inputs)

        upstream = torch.randn(
            ref_q.shape,
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        ref_loss = (reference_recurrent_deltanet(ref_q, ref_k, ref_v, ref_beta).float() * upstream).sum()
        tri_loss = (triton_candidate(tri_q, tri_k, tri_v, tri_beta).float() * upstream).sum()
        ref_loss.backward()
        tri_loss.backward()

        grad_pairs = {
            "Q": (ref_q.grad, tri_q.grad),
            "K": (ref_k.grad, tri_k.grad),
            "V": (ref_v.grad, tri_v.grad),
            "beta": (ref_beta.grad, tri_beta.grad),
        }
        for var_name, (ref_grad, tri_grad) in grad_pairs.items():
            report = max_violation_report(case.name, var_name, ref_grad, tri_grad)
            rows.append(report)
            if not torch.allclose(ref_grad, tri_grad, atol=SHADOW_ATOL, rtol=SHADOW_RTOL):
                failures.append(report)

    if failures:
        print("BF16 shadow gradient mismatches detected:")
        for failure in failures:
            print(
                "case={case} var={var} index={index} chunk={chunk} token={token} "
                "max_abs={max_abs:.8f} max_rel={max_rel:.8f} "
                "ref_value={ref_value:.8f} triton_value={triton_value:.8f}".format(**failure)
            )
        raise AssertionError(f"{len(failures)} gradient tensors exceeded atol={SHADOW_ATOL}, rtol={SHADOW_RTOL}")
    return rows


def format_success_table(rows: list[dict[str, object]]) -> str:
    lines = [
        "| case | variable | max_abs_delta | max_rel_delta |",
        "|---|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case']} | {row['var']} | {row['max_abs']:.8f} | {row['max_rel']:.8f} |"
        )
    return "\n".join(lines)


def main() -> None:
    print(f"production_valid={PRODUCTION_VALID} reason={PRODUCTION_INVALID_REASON}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to validate the Triton DeltaNet kernel")
    device = torch.device("cuda")
    run_gradcheck(device)
    shadow_rows = run_shadow_check(device)
    print(format_success_table(shadow_rows))
    print()
    print("DeltaNet Triton kernel validation passed.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
