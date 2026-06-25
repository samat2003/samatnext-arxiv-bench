import math

import torch
import triton
import triton.language as tl


@triton.jit
def _deltanet_chunk_metrics_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    q = tl.load(q_ptr + offsets, mask=mask, other=0.0)
    k = tl.load(k_ptr + offsets, mask=mask, other=0.0)
    v = tl.load(v_ptr + offsets, mask=mask, other=0.0)
    metric = q * k + v
    tl.store(out_ptr + offsets, metric, mask=mask)


class FusedLinearCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        targets: torch.Tensor,
        block_vocab: int,
        ignore_index: int,
    ) -> torch.Tensor:
        if hidden.ndim != 2:
            raise ValueError("hidden must have shape [tokens, d_model]")
        if weight.ndim != 2:
            raise ValueError("weight must have shape [vocab_size, d_model]")
        if targets.ndim != 1:
            raise ValueError("targets must have shape [tokens]")
        if hidden.shape[0] != targets.shape[0]:
            raise ValueError("hidden and targets must have the same token dimension")
        if hidden.shape[1] != weight.shape[1]:
            raise ValueError("hidden and weight d_model dimensions must match")

        tokens = hidden.shape[0]
        vocab_size = weight.shape[0]
        if block_vocab <= 0:
            raise ValueError("block_vocab must be positive")
        valid_mask = targets != ignore_index
        targets_in_range = (targets >= 0) & (targets < vocab_size)
        valid_targets = (~valid_mask) | targets_in_range
        if hasattr(torch, "_assert_async"):
            torch._assert_async(valid_targets.all(), "target is outside vocabulary")
        elif not bool(valid_targets.all()):
            raise ValueError("target is outside vocabulary")

        compute_dtype = torch.bfloat16 if hidden.is_cuda else torch.float32
        hidden_compute = hidden.to(compute_dtype)

        row_max = torch.full((tokens,), -float("inf"), device=hidden.device, dtype=torch.float32)
        row_sum = torch.zeros((tokens,), device=hidden.device, dtype=torch.float32)
        target_logits = torch.zeros((tokens,), device=hidden.device, dtype=torch.float32)

        # Online log-sum-exp eliminates the previous second projection pass.
        # Each vocabulary block is projected exactly once while maintaining a
        # numerically stable running maximum and rescaled exponential sum.
        for start in range(0, vocab_size, block_vocab):
            end = min(start + block_vocab, vocab_size)
            weight_chunk = weight[start:end].to(compute_dtype)
            logits = hidden_compute.matmul(weight_chunk.t()).float()
            block_max = logits.max(dim=1).values
            new_max = torch.maximum(row_max, block_max)
            row_sum = row_sum * torch.exp(row_max - new_max)
            row_sum += torch.exp(logits - new_max[:, None]).sum(dim=1)
            row_max = new_max

            local_targets = targets - start
            target_mask = valid_mask & (local_targets >= 0) & (local_targets < end - start)
            safe_targets = local_targets.clamp(0, end - start - 1)
            selected_logits = logits.gather(1, safe_targets.unsqueeze(1)).squeeze(1)
            target_logits = torch.where(target_mask, selected_logits, target_logits)

        logsumexp = row_max + torch.log(row_sum)
        per_token_loss = (logsumexp - target_logits).masked_fill(~valid_mask, 0.0)
        valid_count = valid_mask.sum()
        loss = per_token_loss.sum() / valid_count
        ctx.save_for_backward(hidden, weight, targets, logsumexp)
        ctx.block_vocab = block_vocab
        ctx.ignore_index = ignore_index
        return loss

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        hidden, weight, targets, logsumexp = ctx.saved_tensors
        block_vocab = ctx.block_vocab
        ignore_index = ctx.ignore_index
        tokens = hidden.shape[0]
        vocab_size = weight.shape[0]
        compute_dtype = torch.bfloat16 if hidden.is_cuda else torch.float32
        valid_mask = targets != ignore_index
        valid_count = valid_mask.sum().clamp_min(1).to(grad_output.dtype)
        loss_scale = grad_output / valid_count

        hidden_f32 = hidden.float()
        weight_f32 = weight.float()
        hidden_compute = hidden.to(compute_dtype)

        grad_hidden = torch.zeros_like(hidden_f32)
        grad_weight = torch.zeros_like(weight_f32)

        for start in range(0, vocab_size, block_vocab):
            end = min(start + block_vocab, vocab_size)
            weight_chunk = weight[start:end]
            logits = hidden_compute.matmul(weight_chunk.to(compute_dtype).t()).float()
            probs = torch.exp(logits - logsumexp[:, None])
            probs *= valid_mask.unsqueeze(1)

            local_targets = targets - start
            target_mask = valid_mask & (local_targets >= 0) & (local_targets < end - start)
            safe_targets = local_targets.clamp(0, end - start - 1)
            probs.scatter_add_(
                1,
                safe_targets.unsqueeze(1),
                -target_mask.to(probs.dtype).unsqueeze(1),
            )

            probs *= loss_scale
            grad_hidden += probs.matmul(weight_f32[start:end])
            grad_weight[start:end] = probs.t().matmul(hidden_f32)

        return grad_hidden.to(hidden.dtype), grad_weight.to(weight.dtype), None, None, None


def fused_linear_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    block_vocab: int = 4096,
    ignore_index: int = -100,
) -> torch.Tensor:
    flat_hidden = hidden.reshape(-1, hidden.shape[-1]).contiguous()
    flat_targets = targets.reshape(-1).contiguous()
    return FusedLinearCrossEntropyFunction.apply(
        flat_hidden,
        weight,
        flat_targets,
        block_vocab,
        ignore_index,
    )


class FusedDeltaNetChunkFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if q.shape != k.shape or q.shape != v.shape:
            raise ValueError("q, k, and v must have identical shapes")
        out = torch.empty_like(q)
        if q.is_cuda:
            grid = (triton.cdiv(q.numel(), 1024),)
            _deltanet_chunk_metrics_kernel[grid](q, k, v, out, q.numel(), BLOCK=1024)
        else:
            out.copy_(q * k + v)
        ctx.save_for_backward(q, k)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k = ctx.saved_tensors
        return grad_out * k, grad_out * q, grad_out


def fused_deltanet_chunk_metrics(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    return FusedDeltaNetChunkFunction.apply(q, k, v)
