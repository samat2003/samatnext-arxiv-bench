"""Reference SamatNext-CL components.

This module intentionally favors direct, readable tensor code over custom
kernels. The training DeltaNet path uses chunkwise BLAS-style tensor operations
and a causal triangular solve for the intra-chunk feedback correction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterator, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


MODEL_SEMANTICS_VERSION = 2


@dataclass(frozen=True)
class PrecisionConversionRecord:
    module_name: str
    original_backend: str
    converted_backend: str
    status: str
    target_precision: str
    parameter_count: int
    reason: str


def causal_prefix_summary(values: Tensor, output_dim: int) -> Tensor:
    """Return causal prefix sums scaled by sqrt(prefix length).

    This is the linear equivalent of multiplying each chunk by a lower
    triangular matrix and then adding all preceding chunk sums.
    """
    if values.ndim != 3:
        raise ValueError("values must have shape [batch, sequence, features]")
    if output_dim <= 0:
        raise ValueError("output_dim must be positive")
    seq_len = values.shape[1]
    if seq_len == 0:
        return values.new_empty(values.shape[0], 0, output_dim)

    counts = torch.arange(1, seq_len + 1, device=values.device, dtype=torch.float32)
    summary = values.cumsum(dim=1) * torch.rsqrt(counts).to(values.dtype).view(1, -1, 1)
    if summary.shape[-1] < output_dim:
        repeats = math.ceil(output_dim / summary.shape[-1])
        summary = summary.repeat(1, 1, repeats)
    return summary[:, :, :output_dim]


@dataclass(frozen=True)
class SamatNextCLConfig:
    d_model: int = 32
    num_heads: int = 4
    h_sliding: int = 2
    sliding_window: int = 4
    chunk_size: int = 4

    def __post_init__(self) -> None:
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if not 0 <= self.h_sliding <= self.num_heads:
            raise ValueError("h_sliding must be in [0, num_heads]")
        if self.sliding_window <= 0:
            raise ValueError("sliding_window must be positive")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")


class SamatNextCLFinalSpec(nn.Module):
    """Small reference module for DeltaNet plus split-head anchor attention."""

    def __init__(
        self,
        d_model: int = 32,
        num_heads: int = 4,
        h_sliding: int = 2,
        sliding_window: int = 4,
        chunk_size: int = 4,
        num_layers: int = 1,
        enable_eviction_feedback: bool = False,
    ) -> None:
        super().__init__()
        self.config = SamatNextCLConfig(
            d_model=d_model,
            num_heads=num_heads,
            h_sliding=h_sliding,
            sliding_window=sliding_window,
            chunk_size=chunk_size,
        )
        self.d_model = self.config.d_model
        self.num_heads = self.config.num_heads
        self.h_sliding = self.config.h_sliding
        self.h_global = self.num_heads - self.h_sliding
        self.sliding_window = self.config.sliding_window
        self.chunk_size = self.config.chunk_size
        self.num_layers = num_layers
        self.enable_eviction_feedback = enable_eviction_feedback
        self.d_head = self.d_model // self.num_heads

        self.delta_q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.delta_k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.delta_v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.delta_beta_proj = nn.Linear(self.d_model, self.num_heads)
        self.delta_out_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.delta_residual_gate = nn.Linear(self.d_model, self.d_model)

        self.anchor_q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.anchor_k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.anchor_v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.anchor_out_proj = nn.Linear(self.d_model, self.d_model, bias=False)

        self.evict_compressor = nn.Linear(self.d_model, self.d_model)
        self.evict_importance = nn.Linear(self.d_model, 1)
        self.write_key_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.write_val_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.decay_gate_proj = nn.Linear(self.d_model, self.d_model)
        self.memory_query_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.memory_out_proj = nn.Linear(self.d_model, self.d_model, bias=False)

    def _shape_heads(self, x: Tensor) -> Tensor:
        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, self.num_heads, self.d_head)

    def _build_split_head_anchor_allowed_mask(
        self, seq_len: int, device: torch.device | str
    ) -> Tensor:
        """Return a [H, T, T] causal mask with sliding limits on early heads."""
        if seq_len < 0:
            raise ValueError("seq_len must be non-negative")

        positions = torch.arange(seq_len, device=device)
        query_pos = positions[:, None]
        key_pos = positions[None, :]
        causal = key_pos <= query_pos

        mask = causal.expand(self.num_heads, seq_len, seq_len).clone()
        if self.h_sliding:
            lower_bound = query_pos - self.sliding_window + 1
            sliding = causal & (key_pos >= lower_bound)
            mask[: self.h_sliding] = sliding
        return mask

    def split_head_anchor_block(self, hidden: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Causal split-head attention with sliding heads and global heads."""
        if hidden.ndim != 3 or hidden.shape[-1] != self.d_model:
            raise ValueError("hidden must have shape [B, T, d_model]")

        bsz, seq_len, _ = hidden.shape
        q = self._shape_heads(self.anchor_q_proj(hidden)).transpose(1, 2)
        k = self._shape_heads(self.anchor_k_proj(hidden)).transpose(1, 2)
        v = self._shape_heads(self.anchor_v_proj(hidden)).transpose(1, 2)

        head_outputs = []
        if self.h_sliding:
            q_sliding = q[:, : self.h_sliding]
            k_sliding = k[:, : self.h_sliding]
            v_sliding = v[:, : self.h_sliding]
            if self.sliding_window >= seq_len:
                head_outputs.append(
                    F.scaled_dot_product_attention(
                        q_sliding,
                        k_sliding,
                        v_sliding,
                        dropout_p=0.0,
                        is_causal=True,
                    )
                )
            else:
                positions = torch.arange(seq_len, device=hidden.device)
                query_pos = positions[:, None]
                key_pos = positions[None, :]
                sliding_mask = (key_pos <= query_pos) & (
                    key_pos >= query_pos - self.sliding_window + 1
                )
                head_outputs.append(
                    F.scaled_dot_product_attention(
                        q_sliding,
                        k_sliding,
                        v_sliding,
                        attn_mask=sliding_mask,
                        dropout_p=0.0,
                    )
                )

        if self.h_global:
            head_outputs.append(
                F.scaled_dot_product_attention(
                    q[:, self.h_sliding :],
                    k[:, self.h_sliding :],
                    v[:, self.h_sliding :],
                    dropout_p=0.0,
                    is_causal=True,
                )
            )

        attended = torch.cat(head_outputs, dim=1)
        attended = attended.transpose(1, 2).reshape(bsz, seq_len, self.d_model)
        updated = self.anchor_out_proj(attended)

        expired_k = k[:, : self.h_sliding].transpose(1, 2).contiguous()
        expired_v = v[:, : self.h_sliding].transpose(1, 2).contiguous()
        return hidden + updated, expired_k, expired_v

    def gated_deltanet_block(self, hidden: Tensor) -> Tensor:
        if self.training:
            return self._chunkwise_parallel_deltanet(hidden)
        return self._sequential_recurrent_deltanet(hidden)

    def _chunkwise_parallel_deltanet(self, hidden: Tensor) -> Tensor:
        """Chunkwise parallel DeltaNet path aligned to the sequential reference.

        For each chunk, the delta rule

            e_t = beta_t * (v_t - k_t @ M_{t-1})

        is solved concurrently over the chunk length. Since
        M_{t-1} = M_0 + sum_{i<t} outer(k_i, e_i), the chunk errors satisfy a
        lower-triangular system:

            (I + diag(beta) @ tril(KK^T, -1)) @ E = beta * (V - K @ M_0)

        The chunk memory handoff is sequential across chunks, but there is no
        token-by-token loop inside the chunk computation.
        """
        if hidden.ndim != 3 or hidden.shape[-1] != self.d_model:
            raise ValueError("hidden must have shape [B, T, d_model]")

        bsz, seq_len, _ = hidden.shape
        if seq_len == 0:
            return hidden

        chunk_len = self.chunk_size
        pad_len = (chunk_len - seq_len % chunk_len) % chunk_len
        if pad_len:
            padded_hidden = F.pad(hidden, (0, 0, 0, pad_len))
        else:
            padded_hidden = hidden

        padded_len = padded_hidden.shape[1]
        num_chunks = padded_len // chunk_len

        q = self._shape_heads(self.delta_q_proj(padded_hidden))
        k = F.normalize(self._shape_heads(self.delta_k_proj(padded_hidden)), p=2.0, dim=-1)
        v = self._shape_heads(self.delta_v_proj(padded_hidden))
        beta = torch.sigmoid(self.delta_beta_proj(padded_hidden))

        q_chunks = q.view(bsz, num_chunks, chunk_len, self.num_heads, self.d_head)
        k_chunks = k.view(bsz, num_chunks, chunk_len, self.num_heads, self.d_head)
        v_chunks = v.view(bsz, num_chunks, chunk_len, self.num_heads, self.d_head)
        beta_chunks = beta.view(bsz, num_chunks, chunk_len, self.num_heads)

        memory = hidden.new_zeros(bsz, self.num_heads, self.d_head, self.d_head)
        out_chunks = []
        identity = torch.eye(chunk_len, dtype=hidden.dtype, device=hidden.device)
        causal_lower = torch.tril(
            torch.ones(chunk_len, chunk_len, dtype=torch.bool, device=hidden.device),
            diagonal=0,
        )
        scale = 1.0 / math.sqrt(self.d_head)

        for chunk_idx in range(num_chunks):
            q_chunk = q_chunks[:, chunk_idx].transpose(1, 2)
            k_chunk = k_chunks[:, chunk_idx].transpose(1, 2)
            v_chunk = v_chunks[:, chunk_idx].transpose(1, 2)
            beta_chunk = beta_chunks[:, chunk_idx].transpose(1, 2)

            inherited = torch.einsum("bhcd,bhde->bhce", k_chunk, memory)
            rhs = (v_chunk - inherited) * beta_chunk.unsqueeze(-1)

            covariance = torch.einsum("bhcd,bhsd->bhcs", k_chunk, k_chunk)
            feedback = covariance.masked_fill(~torch.tril(causal_lower, diagonal=-1), 0.0)
            system = identity + beta_chunk.unsqueeze(-1) * feedback

            flat_system = system.reshape(
                bsz * self.num_heads, chunk_len, chunk_len
            )
            flat_rhs = rhs.reshape(bsz * self.num_heads, chunk_len, self.d_head)
            errors = torch.linalg.solve_triangular(
                flat_system,
                flat_rhs,
                upper=False,
                unitriangular=False,
            ).view(bsz, self.num_heads, chunk_len, self.d_head)

            # --- Decompose q @ M_t to avoid the (B,H,C,D,D) outer-product cumsum ---
            mem_contribution = torch.einsum("bhcd,bhde->bhce", q_chunk * scale, memory)
            intra_attn = torch.einsum("bhcd,bhsd->bhcs", q_chunk * scale, k_chunk)
            intra_attn = intra_attn.masked_fill(~causal_lower, 0.0)
            intra_contribution = torch.einsum("bhcs,bhse->bhce", intra_attn, errors)
            chunk_out = mem_contribution + intra_contribution
            out_chunks.append(chunk_out.transpose(1, 2))

            # Memory update: sum of outer products across the whole chunk
            memory = memory + torch.einsum("bhcd,bhce->bhde", k_chunk, errors)

        recurrent = torch.cat(out_chunks, dim=1).reshape(bsz, padded_len, self.d_model)
        recurrent = recurrent[:, :seq_len]
        recurrent = self.delta_out_proj(recurrent)
        gate = torch.sigmoid(self.delta_residual_gate(hidden))
        return hidden + gate * recurrent

    def _sequential_recurrent_deltanet(self, hidden: Tensor) -> Tensor:
        """Token-by-token DeltaNet recurrence used as eval/reference behavior."""
        if hidden.ndim != 3 or hidden.shape[-1] != self.d_model:
            raise ValueError("hidden must have shape [B, T, d_model]")

        bsz, seq_len, _ = hidden.shape
        q = self._shape_heads(self.delta_q_proj(hidden))
        k = self._shape_heads(self.delta_k_proj(hidden))
        v = self._shape_heads(self.delta_v_proj(hidden))
        beta = torch.sigmoid(self.delta_beta_proj(hidden))

        memory = hidden.new_zeros(bsz, self.num_heads, self.d_head, self.d_head)
        outs = []
        scale = 1.0 / math.sqrt(self.d_head)
        for t in range(seq_len):
            qt = q[:, t]
            kt = F.normalize(k[:, t], p=2.0, dim=-1)
            vt = v[:, t]
            bt = beta[:, t].unsqueeze(-1)

            pred = torch.einsum("bhd,bhde->bhe", kt, memory)
            delta = (vt - pred) * bt
            memory = memory + torch.einsum("bhd,bhe->bhde", kt, delta)
            out_t = torch.einsum("bhd,bhde->bhe", qt * scale, memory)
            outs.append(out_t)

        recurrent = torch.stack(outs, dim=1).reshape(bsz, seq_len, self.d_model)
        recurrent = self.delta_out_proj(recurrent)
        gate = torch.sigmoid(self.delta_residual_gate(hidden))
        return hidden + gate * recurrent

    def read_chunk_delayed_memory(
        self,
        hidden_chunk: Tensor,
        memory_k: Optional[Tensor],
        memory_v: Optional[Tensor],
    ) -> Tensor:
        if memory_k is None or memory_v is None or self.h_sliding == 0:
            return hidden_chunk
        if memory_k.numel() == 0 or memory_v.numel() == 0:
            return hidden_chunk

        memory_summary = memory_v.mean(dim=(1, 2), keepdim=False)
        repeats = math.ceil(self.d_model / memory_summary.shape[-1])
        memory_full = memory_summary.repeat(1, repeats)[:, : self.d_model].unsqueeze(1)

        if not self.enable_eviction_feedback:
            scalar_summary = memory_full.mean(dim=-1, keepdim=True)
            return hidden_chunk + scalar_summary.expand_as(hidden_chunk) / math.sqrt(self.d_model)

        memory_full = memory_full.expand(-1, hidden_chunk.shape[1], -1)
        compressed = torch.tanh(self.evict_compressor(memory_full))
        importance = torch.sigmoid(self.evict_importance(memory_full))
        write_values = self.write_val_proj(memory_full)
        write_keys = self.write_key_proj(hidden_chunk)
        queries = self.memory_query_proj(hidden_chunk)
        decay = torch.sigmoid(self.decay_gate_proj(hidden_chunk))
        key_query_gate = torch.sigmoid((write_keys * queries).mean(dim=-1, keepdim=True))
        highway = self.memory_out_proj((compressed + write_values) * importance * decay * key_query_gate)
        return hidden_chunk + highway / math.sqrt(self.d_model)

    def write_evictions_to_memory(
        self,
        old_k: Optional[Tensor],
        old_v: Optional[Tensor],
        expired_k: Tensor,
        expired_v: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        if self.h_sliding == 0:
            empty = expired_k[:, :0].contiguous()
            return empty, empty

        if old_k is not None and old_v is not None:
            new_k = torch.cat([old_k, expired_k], dim=1)
            new_v = torch.cat([old_v, expired_v], dim=1)
        else:
            new_k, new_v = expired_k, expired_v

        max_tokens = self.sliding_window
        return new_k[:, -max_tokens:].contiguous(), new_v[:, -max_tokens:].contiguous()

    def forward_blueprint_pseudo_code(self, hidden: Tensor) -> Tensor:
        """Run chunks with delayed read-before-write memory visibility."""
        if hidden.ndim != 3 or hidden.shape[-1] != self.d_model:
            raise ValueError("hidden must have shape [B, T, d_model]")

        for _ in range(max(1, self.num_layers)):
            hidden = self._forward_one_layer(hidden)
        return hidden

    def _forward_one_layer(self, hidden: Tensor) -> Tensor:
        memory_k: Optional[Tensor] = None
        memory_v: Optional[Tensor] = None
        output_chunks = []

        for start in range(0, hidden.shape[1], self.chunk_size):
            chunk = hidden[:, start : start + self.chunk_size]

            chunk = self.read_chunk_delayed_memory(chunk, memory_k, memory_v)
            chunk = self.gated_deltanet_block(chunk)
            chunk, expired_k, expired_v = self.split_head_anchor_block(chunk)
            output_chunks.append(chunk)

            memory_k, memory_v = self.write_evictions_to_memory(
                memory_k, memory_v, expired_k, expired_v
            )

        return torch.cat(output_chunks, dim=1)

    def parameter_groups_for_optimizer(self, base_lr: float) -> list[dict[str, object]]:
        memory_names = (
            "evict_compressor",
            "evict_importance",
            "write_key_proj",
            "write_val_proj",
            "decay_gate_proj",
            "memory_query_proj",
            "memory_out_proj",
        )
        base_params = []
        memory_params = []
        for name, param in self.named_parameters():
            if any(memory_name in name for memory_name in memory_names):
                memory_params.append(param)
            else:
                base_params.append(param)
        return [
            {"params": base_params, "lr": base_lr},
            {"params": memory_params, "lr": base_lr},
        ]


SamatNextCLReference = SamatNextCLFinalSpec
SamatNextCL = SamatNextCLFinalSpec
SamatNextCLFinalSpecification = SamatNextCLFinalSpec

SamatNextCLFinalSpec.mock_gated_deltanet_block = SamatNextCLFinalSpec.gated_deltanet_block
SamatNextCLFinalSpec.mock_split_head_anchor_block = SamatNextCLFinalSpec.split_head_anchor_block


class MLAKVProjection(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_head: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_head
        self.in_features = d_model
        self.out_features = 3 * d_model
        self.kv_lora_down = nn.Linear(d_model, 512, bias=False)
        self.kv_lora_up = nn.Linear(512, 2 * d_model, bias=False)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
    @property
    def weight(self) -> torch.Tensor:
        kv_weight = self.kv_lora_up.weight.data @ self.kv_lora_down.weight.data
        return torch.cat([self.q_proj.weight.data, kv_weight], dim=0)
    @property
    def bias(self) -> Optional[torch.Tensor]:
        return None
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.q_proj(x)
        kv_latent = self.kv_lora_down(x)
        kv = self.kv_lora_up(kv_latent)
        k, v = kv.chunk(2, dim=-1)
        return q, k, v

class SparseTop1FFN(nn.Module):
    """Top-1 Sparse FFN using token grouping with static tensor contractions to avoid graph breaks."""

    def __init__(self, d_model: int, ffn_dim: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.ffn_dim = ffn_dim
        self.num_experts = 4
        self.h_expert = ffn_dim // self.num_experts

        self.router = nn.Linear(d_model, self.num_experts, bias=False)
        self.gate_weights = nn.Parameter(torch.empty(self.num_experts, self.h_expert, d_model))
        self.up_weights = nn.Parameter(torch.empty(self.num_experts, self.h_expert, d_model))
        self.down_weights = nn.Parameter(torch.empty(self.num_experts, d_model, self.h_expert))
        self.init_weights()

    def init_weights(self) -> None:
        nn.init.normal_(self.gate_weights, std=0.02)
        nn.init.normal_(self.up_weights, std=0.02)
        nn.init.normal_(self.down_weights, std=0.02)
        nn.init.zeros_(self.router.weight)

    @property
    def ffn_gate_proj(self) -> nn.Module:
        class LinearFacade(nn.Module):
            def __init__(self, weight: Tensor):
                super().__init__()
                self.weight = weight
                self.in_features = weight.shape[-1]
                self.out_features = weight.shape[0] * weight.shape[1]
                self.bias = None
            def state_dict(self, *args, **kwargs):
                return {"weight": self.weight.data.view(self.out_features, self.in_features)}
            def load_state_dict(self, state_dict, strict=True, assign=False):
                w = state_dict["weight"]
                self.weight.data.copy_(w.view(self.weight.shape))
                return nn.modules.module._IncompatibleKeys([], [])
        return LinearFacade(self.gate_weights)

    @property
    def ffn_up_proj(self) -> nn.Module:
        class LinearFacade(nn.Module):
            def __init__(self, weight: Tensor):
                super().__init__()
                self.weight = weight
                self.in_features = weight.shape[-1]
                self.out_features = weight.shape[0] * weight.shape[1]
                self.bias = None
            def state_dict(self, *args, **kwargs):
                return {"weight": self.weight.data.view(self.out_features, self.in_features)}
            def load_state_dict(self, state_dict, strict=True, assign=False):
                w = state_dict["weight"]
                self.weight.data.copy_(w.view(self.weight.shape))
                return nn.modules.module._IncompatibleKeys([], [])
        return LinearFacade(self.up_weights)

    @property
    def ffn_down_proj(self) -> nn.Module:
        class LinearFacade(nn.Module):
            def __init__(self, weight: Tensor):
                super().__init__()
                self.weight = weight
                self.in_features = weight.shape[0] * weight.shape[2]
                self.out_features = weight.shape[1]
                self.bias = None
            def state_dict(self, *args, **kwargs):
                # Flat weight representation: [out_features, in_features] -> [1024, 3584]
                # self.weight shape: [4, 1024, 896] -> permute/reshape to [1024, 3584]
                # permute to [1024, 4, 896] then reshape to [1024, 3584]
                w = self.weight.data.permute(1, 0, 2).reshape(self.out_features, self.in_features)
                return {"weight": w}
            def load_state_dict(self, state_dict, strict=True, assign=False):
                w = state_dict["weight"] # [1024, 3584]
                # reshape back to [1024, 4, 896], permute to [4, 1024, 896]
                w_orig = w.view(self.out_features, self.weight.shape[0], self.weight.shape[2]).permute(1, 0, 2)
                self.weight.data.copy_(w_orig)
                return nn.modules.module._IncompatibleKeys([], [])
        return LinearFacade(self.down_weights)

    def forward(self, x: Tensor) -> Tensor:
        # x shape: [B, T, D]
        B, T, D = x.shape
        router_logits = self.router(x) # [B, T, E]
        idx = router_logits.argmax(dim=-1) # [B, T]
        idx = torch.clamp(idx, 0, self.num_experts - 1)
        mask = F.one_hot(idx, num_classes=self.num_experts).to(x.dtype) # [B, T, E]

        # Check if converted FP8 weights exist (e.g. from convert_experimental_fp8_linears)
        if hasattr(self, "ffn_gate_proj_converted"):
            # If converted to FP8 Linear, represent as a standard SwiGLU projection structure
            # and multiply with expert routing mask.
            gate_out = self.ffn_gate_proj_converted(x) # [B, T, F_dim]
            up_out = self.ffn_up_proj_converted(x)     # [B, T, F_dim]
            
            # Since F_dim = E * H (e.g. 4 * 896), we view it as [B, T, E, H]
            fuzz_gate = gate_out.view(B, T, self.num_experts, self.h_expert)
            fuzz_up = up_out.view(B, T, self.num_experts, self.h_expert)
            
            gate_act = torch.einsum('bteh, bte -> bth', fuzz_gate, mask)
            up_act = torch.einsum('bteh, bte -> bth', fuzz_up, mask)
            
            intermediate = F.silu(gate_act) * up_act # [B, T, H]
            
            inter_mapped = torch.einsum('bth, bte -> bteh', intermediate, mask)
            
            # Map inter_mapped back to flat intermediate representation [B, T, F_dim]
            inter_flat = inter_mapped.view(B, T, self.ffn_dim)
            output = self.ffn_down_proj_converted(inter_flat)
        else:
            # Project inputs across ALL experts simultaneously (Ultra-low VRAM footprint)
            # x: [B, T, D], gate_weights: [E, H, D] -> [B, T, E, H]
            fuzz_gate = torch.einsum('btd, ehd -> bteh', x, self.gate_weights)
            fuzz_up = torch.einsum('btd, ehd -> bteh', x, self.up_weights)

            # Filter down to the active expert's hidden activations via the mask
            gate_act = torch.einsum('bteh, bte -> bth', fuzz_gate, mask) # [B, T, H]
            up_act = torch.einsum('bteh, bte -> bth', fuzz_up, mask)     # [B, T, H]

            # Core SwiGLU activation
            intermediate = F.silu(gate_act) * up_act # [B, T, H]

            # Map intermediate back to expert space and apply down projection
            inter_mapped = torch.einsum('bth, bte -> bteh', intermediate, mask)
            # Contract with down weights: [B, T, E, H] against [E, D, H] -> [B, T, D]
            output = torch.einsum('bteh, edh -> btd', inter_mapped, self.down_weights)

        # Scale output by router probability to propagate gradients back to the router
        probs = F.softmax(router_logits, dim=-1)
        gate_probs = probs.gather(dim=-1, index=idx.unsqueeze(-1)).squeeze(-1)
        return output * gate_probs.unsqueeze(-1)

class DeltaBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, sliding_window: int, chunk_size: int, h_sliding: Optional[int] = None, ffn_dim: Optional[int] = None, d_evict: int = 256, enable_eviction_feedback: bool = True) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.chunk_size = chunk_size
        norm_cls = getattr(nn, 'RMSNorm', nn.LayerNorm)
        self.delta_norm = norm_cls(d_model)
        self.ffn_norm = norm_cls(d_model)
        self.delta_qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.delta_beta_proj = nn.Linear(d_model, num_heads)
        self.delta_out_proj = nn.Linear(d_model, d_model, bias=False)
        self.delta_residual_gate = nn.Linear(d_model, d_model)
        ffn_dim = ffn_dim if ffn_dim is not None else int(3.5 * d_model)
        self.ffn = SparseTop1FFN(d_model, ffn_dim)
        self.register_buffer('identity', torch.eye(chunk_size), persistent=False)
        self.register_buffer('causal_lower', torch.tril(torch.ones(chunk_size, chunk_size, dtype=torch.bool)), persistent=False)
        self.register_buffer('causal_strict_lower', torch.tril(torch.ones(chunk_size, chunk_size, dtype=torch.bool), diagonal=-1), persistent=False)
    @property
    def ffn_gate_proj(self) -> nn.Linear:
        return self.ffn.ffn_gate_proj
    @property
    def ffn_up_proj(self) -> nn.Linear:
        return self.ffn.ffn_up_proj
    @property
    def ffn_down_proj(self) -> nn.Linear:
        return self.ffn.ffn_down_proj
    def _shape_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, self.num_heads, self.d_head)
    def _chunkwise_parallel_deltanet(self, hidden: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = hidden.shape
        if seq_len == 0:
            return hidden
        chunk_len = self.chunk_size
        pad_len = (chunk_len - seq_len % chunk_len) % chunk_len
        padded_hidden = F.pad(hidden, (0, 0, 0, pad_len)) if pad_len else hidden
        padded_len = padded_hidden.shape[1]
        num_chunks = padded_len // chunk_len
        q_raw, k_raw, v_raw = self.delta_qkv_proj(padded_hidden).chunk(3, dim=-1)
        q = F.normalize(self._shape_heads(F.elu(q_raw) + 1.0), p=2.0, dim=-1)
        k = F.normalize(self._shape_heads(F.elu(k_raw) + 1.0), p=2.0, dim=-1)
        v = self._shape_heads(v_raw)
        beta = torch.sigmoid(self.delta_beta_proj(padded_hidden))
        q_chunks = q.view(bsz, num_chunks, chunk_len, self.num_heads, self.d_head)
        k_chunks = k.view(bsz, num_chunks, chunk_len, self.num_heads, self.d_head)
        v_chunks = v.view(bsz, num_chunks, chunk_len, self.num_heads, self.d_head)
        beta_chunks = beta.view(bsz, num_chunks, chunk_len, self.num_heads)
        memory = hidden.new_zeros(bsz, self.num_heads, self.d_head, self.d_head)
        out_chunks = []
        identity = self.identity.to(device=hidden.device, dtype=hidden.dtype)
        causal_lower = self.causal_lower.to(device=hidden.device, dtype=hidden.dtype)
        causal_strict_lower = self.causal_strict_lower.to(device=hidden.device, dtype=hidden.dtype)
        scale = 1.0 / math.sqrt(self.d_head)
        for chunk_idx in range(num_chunks):
            q_chunk = q_chunks[:, chunk_idx].transpose(1, 2)
            k_chunk = k_chunks[:, chunk_idx].transpose(1, 2)
            v_chunk = v_chunks[:, chunk_idx].transpose(1, 2)
            beta_chunk = beta_chunks[:, chunk_idx].transpose(1, 2)
            inherited = torch.einsum('bhcd,bhde->bhce', k_chunk, memory)
            rhs = (v_chunk - inherited) * beta_chunk.unsqueeze(-1)
            covariance = torch.einsum('bhcd,bhsd->bhcs', k_chunk, k_chunk)
            feedback = covariance.masked_fill(causal_strict_lower.logical_not(), 0.0)
            system = identity + beta_chunk.unsqueeze(-1) * feedback
            flat_system = system.reshape(bsz * self.num_heads, chunk_len, chunk_len)
            flat_rhs = rhs.reshape(bsz * self.num_heads, chunk_len, self.d_head)
            errors = torch.linalg.solve_triangular(flat_system, flat_rhs, upper=False, unitriangular=False).view(bsz, self.num_heads, chunk_len, self.d_head)
            mem_contribution = torch.einsum('bhcd,bhde->bhce', q_chunk * scale, memory)
            intra_attn = torch.einsum('bhcd,bhsd->bhcs', q_chunk * scale, k_chunk)
            intra_attn = intra_attn.masked_fill(causal_lower.logical_not(), 0.0)
            intra_contribution = torch.einsum('bhcs,bhse->bhce', intra_attn, errors)
            chunk_out = mem_contribution + intra_contribution
            out_chunks.append(chunk_out.transpose(1, 2))
            memory = memory + torch.einsum('bhcd,bhce->bhde', k_chunk, errors)
        recurrent = torch.cat(out_chunks, dim=1).reshape(bsz, padded_len, self.d_model)
        recurrent = recurrent[:, :seq_len]
        recurrent = self.delta_out_proj(recurrent)
        gate = torch.sigmoid(self.delta_residual_gate(hidden))
        return gate * recurrent
    def _sequential_recurrent_deltanet(self, hidden: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = hidden.shape
        q_raw, k_raw, v = self.delta_qkv_proj(hidden).chunk(3, dim=-1)
        q = self._shape_heads(F.elu(q_raw) + 1.0)
        k = self._shape_heads(F.elu(k_raw) + 1.0)
        v = self._shape_heads(v)
        beta = torch.sigmoid(self.delta_beta_proj(hidden))
        memory = hidden.new_zeros(bsz, self.num_heads, self.d_head, self.d_head)
        outs = []
        scale = 1.0 / math.sqrt(self.d_head)
        for t in range(seq_len):
            qt = F.normalize(q[:, t], p=2.0, dim=-1)
            kt = F.normalize(k[:, t], p=2.0, dim=-1)
            vt = v[:, t]
            bt = beta[:, t].unsqueeze(-1)
            pred = torch.einsum('bhd,bhde->bhe', kt, memory)
            delta = (vt - pred) * bt
            memory = memory + torch.einsum('bhd,bhe->bhde', kt, delta)
            outs.append(torch.einsum('bhd,bhde->bhe', qt * scale, memory))
        recurrent = torch.stack(outs, dim=1).reshape(bsz, seq_len, self.d_model)
        recurrent = self.delta_out_proj(recurrent)
        gate = torch.sigmoid(self.delta_residual_gate(hidden))
        return gate * recurrent
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        delta_input = self.delta_norm(hidden)
        if self.training:
            delta_update = self._chunkwise_parallel_deltanet(delta_input)
        else:
            delta_update = self._sequential_recurrent_deltanet(delta_input)
        hidden = hidden + delta_update
        ffn_input = self.ffn_norm(hidden)
        ffn_out = self.ffn(ffn_input)
        return hidden + ffn_out

class AnchorBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, sliding_window: int, chunk_size: int, h_sliding: Optional[int] = None, ffn_dim: Optional[int] = None, d_evict: int = 256, enable_eviction_feedback: bool = True) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.sliding_window = sliding_window
        self.h_sliding = h_sliding if h_sliding is not None else math.ceil(num_heads * 0.8)
        self.h_global = num_heads - self.h_sliding
        self.enable_eviction_feedback = enable_eviction_feedback
        norm_cls = getattr(nn, 'RMSNorm', nn.LayerNorm)
        self.anchor_norm = norm_cls(d_model)
        self.ffn_norm = norm_cls(d_model)
        self.anchor_qkv_proj = MLAKVProjection(d_model, num_heads, self.d_head)
        self.anchor_out_proj = nn.Linear(d_model, d_model, bias=False)
        self.anchor_sliding_qkv_proj = None
        self.anchor_global_qkv_proj = None
        self.anchor_qkv_split_offsets = None
        ffn_dim = ffn_dim if ffn_dim is not None else int(3.5 * d_model)
        self.ffn = SparseTop1FFN(d_model, ffn_dim)
        self.evict_compressor = nn.Linear(d_model, d_evict)
        self.evict_importance = nn.Linear(d_model, 1)
        self.write_val_proj = nn.Linear(d_model, d_evict, bias=False)
        self.memory_out_proj = nn.Linear(d_evict, d_model, bias=False)
    @property
    def ffn_gate_proj(self) -> nn.Linear:
        return self.ffn.ffn_gate_proj
    @property
    def ffn_up_proj(self) -> nn.Linear:
        return self.ffn.ffn_up_proj
    @property
    def ffn_down_proj(self) -> nn.Linear:
        return self.ffn.ffn_down_proj
    def _shape_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, self.num_heads, self.d_head)
    def _split_head_anchor_attention(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len, _ = hidden.shape
        if self.anchor_sliding_qkv_proj is not None:
            if self.anchor_global_qkv_proj is None:
                raise RuntimeError('split anchor QKV is missing its global projection')
            sliding_dim = self.h_sliding * self.d_head
            global_dim = self.h_global * self.d_head
            q_sliding_raw, k_sliding_raw, v_sliding_raw = self.anchor_sliding_qkv_proj(hidden).split(sliding_dim, dim=-1)
            if global_dim:
                q_global_raw, k_global_raw, v_global_raw = self.anchor_global_qkv_proj(hidden).split(global_dim, dim=-1)
                q_raw = torch.cat((q_sliding_raw, q_global_raw), dim=-1)
                k_raw = torch.cat((k_sliding_raw, k_global_raw), dim=-1)
                v = torch.cat((v_sliding_raw, v_global_raw), dim=-1)
            else:
                q_raw, k_raw, v = q_sliding_raw, k_sliding_raw, v_sliding_raw
        else:
            q_raw, k_raw, v = self.anchor_qkv_proj(hidden)
        q = self._shape_heads(q_raw).transpose(1, 2)
        k = self._shape_heads(k_raw).transpose(1, 2)
        v = self._shape_heads(v).transpose(1, 2)
        head_outputs = []
        if self.h_sliding:
            q_sliding = q[:, :self.h_sliding]
            k_sliding = k[:, :self.h_sliding]
            v_sliding = v[:, :self.h_sliding]
            if self.sliding_window >= seq_len:
                head_outputs.append(F.scaled_dot_product_attention(q_sliding, k_sliding, v_sliding, dropout_p=0.0, is_causal=True))
            else:
                positions = torch.arange(seq_len, device=hidden.device)
                query_pos = positions[:, None]
                key_pos = positions[None, :]
                sliding_mask = (key_pos <= query_pos) & (key_pos >= query_pos - self.sliding_window + 1)
                head_outputs.append(F.scaled_dot_product_attention(q_sliding, k_sliding, v_sliding, attn_mask=sliding_mask, dropout_p=0.0))
        if self.h_global:
            head_outputs.append(F.scaled_dot_product_attention(q[:, self.h_sliding:], k[:, self.h_sliding:], v[:, self.h_sliding:], dropout_p=0.0, is_causal=True))
        attended = torch.cat(head_outputs, dim=1)
        attended = attended.transpose(1, 2).reshape(bsz, seq_len, self.d_model)
        out = self.anchor_out_proj(attended)
        sliding_values = v[:, :self.h_sliding].transpose(1, 2).contiguous()
        return out, sliding_values
    def _causal_prefix_feedback(self, hidden: torch.Tensor, sliding_values: torch.Tensor) -> torch.Tensor:
        if not self.enable_eviction_feedback or sliding_values.numel() == 0:
            return torch.zeros_like(hidden)
        bsz, seq_len, _, _ = sliding_values.shape
        values_flat = sliding_values.reshape(bsz, seq_len, -1)
        summary = causal_prefix_summary(values_flat, self.d_model)
        compressed = torch.tanh(self.evict_compressor(summary))
        write_val = self.write_val_proj(summary)
        master_gate = torch.sigmoid(self.evict_importance(summary))
        feedback = self.memory_out_proj(compressed + write_val) * master_gate
        return feedback
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        anchor_update, sliding_values = self._split_head_anchor_attention(self.anchor_norm(hidden))
        hidden = hidden + anchor_update
        hidden = hidden + self._causal_prefix_feedback(hidden, sliding_values)
        ffn_input = self.ffn_norm(hidden)
        ffn_out = self.ffn(ffn_input)
        return hidden + ffn_out
    def split_anchor_qkv_projection(self, sliding_projection: nn.Module, global_projection: nn.Module) -> dict[str, tuple[int, int]]:
        if self.anchor_qkv_proj is None:
            raise RuntimeError('anchor_qkv_proj has already been split')
        sliding_dim = self.h_sliding * self.d_head
        global_dim = self.h_global * self.d_head
        if sliding_dim + global_dim != self.d_model:
            raise RuntimeError(f'invalid anchor QKV split: sliding_dim={sliding_dim}, global_dim={global_dim}, d_model={self.d_model}')
        expected_sliding = (3 * sliding_dim, self.d_model)
        expected_global = (3 * global_dim, self.d_model)
        if tuple(sliding_projection.weight.shape) != expected_sliding:
            raise RuntimeError(f'sliding QKV weight shape {tuple(sliding_projection.weight.shape)} does not match {expected_sliding}')
        if tuple(global_projection.weight.shape) != expected_global:
            raise RuntimeError(f'global QKV weight shape {tuple(global_projection.weight.shape)} does not match {expected_global}')
        offsets = {
            'q_sliding': (0, sliding_dim),
            'q_global': (sliding_dim, self.d_model),
            'k_sliding': (self.d_model, self.d_model + sliding_dim),
            'k_global': (self.d_model + sliding_dim, 2 * self.d_model),
            'v_sliding': (2 * self.d_model, 2 * self.d_model + sliding_dim),
            'v_global': (2 * self.d_model + sliding_dim, 3 * self.d_model),
        }
        source = self.anchor_qkv_proj.weight.detach()
        sliding_rows = torch.cat([source[start:end] for key, (start, end) in offsets.items() if key.endswith('sliding')], dim=0)
        global_rows = torch.cat([source[start:end] for key, (start, end) in offsets.items() if key.endswith('global')], dim=0)
        with torch.no_grad():
            sliding_projection.weight.copy_(sliding_rows.to(device=sliding_projection.weight.device, dtype=sliding_projection.weight.dtype))
            global_projection.weight.copy_(global_rows.to(device=global_projection.weight.device, dtype=global_projection.weight.dtype))
        self.anchor_sliding_qkv_proj = sliding_projection
        self.anchor_global_qkv_proj = global_projection
        self.anchor_qkv_proj = None
        self.anchor_qkv_split_offsets = offsets
        return offsets

class SamatNextCLBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        sliding_window: int,
        chunk_size: int,
        h_sliding: Optional[int] = None,
        ffn_dim: Optional[int] = None,
        d_evict: int = 256,
        enable_eviction_feedback: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.sliding_window = sliding_window
        self.chunk_size = chunk_size
        self.h_sliding = h_sliding if h_sliding is not None else math.ceil(num_heads * 0.8)
        self.h_global = num_heads - self.h_sliding
        self.enable_eviction_feedback = enable_eviction_feedback
        
        norm_cls = getattr(nn, 'RMSNorm', nn.LayerNorm)
        self.delta_norm = norm_cls(d_model)
        self.ffn_norm = norm_cls(d_model)
        self.delta_qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.delta_beta_proj = nn.Linear(d_model, num_heads)
        self.delta_out_proj = nn.Linear(d_model, d_model, bias=False)
        self.delta_residual_gate = nn.Linear(d_model, d_model)
        
        self.anchor_norm = norm_cls(d_model)
        self.anchor_qkv_proj = MLAKVProjection(d_model, num_heads, self.d_head)
        self.anchor_out_proj = nn.Linear(d_model, d_model, bias=False)
        
        ffn_dim = ffn_dim if ffn_dim is not None else int(3.5 * d_model)
        self.ffn = SparseTop1FFN(d_model, ffn_dim)
        
        self.evict_compressor = nn.Linear(d_model, d_evict)
        self.evict_importance = nn.Linear(d_model, 1)
        self.write_val_proj = nn.Linear(d_model, d_evict, bias=False)
        self.memory_out_proj = nn.Linear(d_evict, d_model, bias=False)
        
        self.register_buffer('identity', torch.eye(chunk_size), persistent=False)
        self.register_buffer('causal_lower', torch.tril(torch.ones(chunk_size, chunk_size, dtype=torch.bool)), persistent=False)
        self.register_buffer('causal_strict_lower', torch.tril(torch.ones(chunk_size, chunk_size, dtype=torch.bool), diagonal=-1), persistent=False)
        
        self.anchor_sliding_qkv_proj = None
        self.anchor_global_qkv_proj = None
        self.anchor_qkv_split_offsets = None

    @property
    def ffn_gate_proj(self) -> nn.Linear:
        return self.ffn.ffn_gate_proj
    @property
    def ffn_up_proj(self) -> nn.Linear:
        return self.ffn.ffn_up_proj
    @property
    def ffn_down_proj(self) -> nn.Linear:
        return self.ffn.ffn_down_proj

    def _shape_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, self.num_heads, self.d_head)

    def _chunkwise_parallel_deltanet(self, hidden: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = hidden.shape
        if seq_len == 0:
            return hidden
        chunk_len = self.chunk_size
        pad_len = (chunk_len - seq_len % chunk_len) % chunk_len
        padded_hidden = F.pad(hidden, (0, 0, 0, pad_len)) if pad_len else hidden
        padded_len = padded_hidden.shape[1]
        num_chunks = padded_len // chunk_len
        q_raw, k_raw, v_raw = self.delta_qkv_proj(padded_hidden).chunk(3, dim=-1)
        q = F.normalize(self._shape_heads(F.elu(q_raw) + 1.0), p=2.0, dim=-1)
        k = F.normalize(self._shape_heads(F.elu(k_raw) + 1.0), p=2.0, dim=-1)
        v = self._shape_heads(v_raw)
        beta = torch.sigmoid(self.delta_beta_proj(padded_hidden))
        q_chunks = q.view(bsz, num_chunks, chunk_len, self.num_heads, self.d_head)
        k_chunks = k.view(bsz, num_chunks, chunk_len, self.num_heads, self.d_head)
        v_chunks = v.view(bsz, num_chunks, chunk_len, self.num_heads, self.d_head)
        beta_chunks = beta.view(bsz, num_chunks, chunk_len, self.num_heads)
        memory = hidden.new_zeros(bsz, self.num_heads, self.d_head, self.d_head)
        out_chunks = []
        identity = self.identity.to(device=hidden.device, dtype=hidden.dtype)
        causal_lower = self.causal_lower.to(device=hidden.device, dtype=hidden.dtype)
        causal_strict_lower = self.causal_strict_lower.to(device=hidden.device, dtype=hidden.dtype)
        scale = 1.0 / math.sqrt(self.d_head)
        for chunk_idx in range(num_chunks):
            q_chunk = q_chunks[:, chunk_idx].transpose(1, 2)
            k_chunk = k_chunks[:, chunk_idx].transpose(1, 2)
            v_chunk = v_chunks[:, chunk_idx].transpose(1, 2)
            beta_chunk = beta_chunks[:, chunk_idx].transpose(1, 2)
            inherited = torch.einsum('bhcd,bhde->bhce', k_chunk, memory)
            rhs = (v_chunk - inherited) * beta_chunk.unsqueeze(-1)
            covariance = torch.einsum('bhcd,bhsd->bhcs', k_chunk, k_chunk)
            feedback = covariance.masked_fill(causal_strict_lower.logical_not(), 0.0)
            system = identity + beta_chunk.unsqueeze(-1) * feedback
            flat_system = system.reshape(bsz * self.num_heads, chunk_len, chunk_len)
            flat_rhs = rhs.reshape(bsz * self.num_heads, chunk_len, self.d_head)
            errors = torch.linalg.solve_triangular(flat_system, flat_rhs, upper=False, unitriangular=False).view(bsz, self.num_heads, chunk_len, self.d_head)
            mem_contribution = torch.einsum('bhcd,bhde->bhce', q_chunk * scale, memory)
            intra_attn = torch.einsum('bhcd,bhsd->bhcs', q_chunk * scale, k_chunk)
            intra_attn = intra_attn.masked_fill(causal_lower.logical_not(), 0.0)
            intra_contribution = torch.einsum('bhcs,bhse->bhce', intra_attn, errors)
            chunk_out = mem_contribution + intra_contribution
            out_chunks.append(chunk_out.transpose(1, 2))
            memory = memory + torch.einsum('bhcd,bhce->bhde', k_chunk, errors)
        recurrent = torch.cat(out_chunks, dim=1).reshape(bsz, padded_len, self.d_model)
        recurrent = recurrent[:, :seq_len]
        recurrent = self.delta_out_proj(recurrent)
        gate = torch.sigmoid(self.delta_residual_gate(hidden))
        return gate * recurrent

    def _sequential_recurrent_deltanet(self, hidden: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = hidden.shape
        q_raw, k_raw, v = self.delta_qkv_proj(hidden).chunk(3, dim=-1)
        q = self._shape_heads(F.elu(q_raw) + 1.0)
        k = self._shape_heads(F.elu(k_raw) + 1.0)
        v = self._shape_heads(v)
        beta = torch.sigmoid(self.delta_beta_proj(hidden))
        memory = hidden.new_zeros(bsz, self.num_heads, self.d_head, self.d_head)
        outs = []
        scale = 1.0 / math.sqrt(self.d_head)
        for t in range(seq_len):
            qt = F.normalize(q[:, t], p=2.0, dim=-1)
            kt = F.normalize(k[:, t], p=2.0, dim=-1)
            vt = v[:, t]
            bt = beta[:, t].unsqueeze(-1)
            pred = torch.einsum('bhd,bhde->bhe', kt, memory)
            delta = (vt - pred) * bt
            memory = memory + torch.einsum('bhd,bhe->bhde', kt, delta)
            outs.append(torch.einsum('bhd,bhde->bhe', qt * scale, memory))
        recurrent = torch.stack(outs, dim=1).reshape(bsz, seq_len, self.d_model)
        recurrent = self.delta_out_proj(recurrent)
        gate = torch.sigmoid(self.delta_residual_gate(hidden))
        return gate * recurrent

    def _split_head_anchor_attention(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len, _ = hidden.shape
        if self.anchor_sliding_qkv_proj is not None:
            if self.anchor_global_qkv_proj is None:
                raise RuntimeError('split anchor QKV is missing its global projection')
            sliding_dim = self.h_sliding * self.d_head
            global_dim = self.h_global * self.d_head
            q_sliding_raw, k_sliding_raw, v_sliding_raw = self.anchor_sliding_qkv_proj(hidden).split(sliding_dim, dim=-1)
            if global_dim:
                q_global_raw, k_global_raw, v_global_raw = self.anchor_global_qkv_proj(hidden).split(global_dim, dim=-1)
                q_raw = torch.cat((q_sliding_raw, q_global_raw), dim=-1)
                k_raw = torch.cat((k_sliding_raw, k_global_raw), dim=-1)
                v = torch.cat((v_sliding_raw, v_global_raw), dim=-1)
            else:
                q_raw, k_raw, v = q_sliding_raw, k_sliding_raw, v_sliding_raw
        else:
            q_raw, k_raw, v = self.anchor_qkv_proj(hidden)
        q = self._shape_heads(q_raw).transpose(1, 2)
        k = self._shape_heads(k_raw).transpose(1, 2)
        v = self._shape_heads(v).transpose(1, 2)
        head_outputs = []
        if self.h_sliding:
            q_sliding = q[:, :self.h_sliding]
            k_sliding = k[:, :self.h_sliding]
            v_sliding = v[:, :self.h_sliding]
            if self.sliding_window >= seq_len:
                head_outputs.append(F.scaled_dot_product_attention(q_sliding, k_sliding, v_sliding, dropout_p=0.0, is_causal=True))
            else:
                positions = torch.arange(seq_len, device=hidden.device)
                query_pos = positions[:, None]
                key_pos = positions[None, :]
                sliding_mask = (key_pos <= query_pos) & (key_pos >= query_pos - self.sliding_window + 1)
                head_outputs.append(F.scaled_dot_product_attention(q_sliding, k_sliding, v_sliding, attn_mask=sliding_mask, dropout_p=0.0))
        if self.h_global:
            head_outputs.append(F.scaled_dot_product_attention(q[:, self.h_sliding:], k[:, self.h_sliding:], v[:, self.h_sliding:], dropout_p=0.0, is_causal=True))
        attended = torch.cat(head_outputs, dim=1)
        attended = attended.transpose(1, 2).reshape(bsz, seq_len, self.d_model)
        out = self.anchor_out_proj(attended)
        sliding_values = v[:, :self.h_sliding].transpose(1, 2).contiguous()
        return out, sliding_values

    def _causal_prefix_feedback(self, hidden: torch.Tensor, sliding_values: torch.Tensor) -> torch.Tensor:
        if not self.enable_eviction_feedback or sliding_values.numel() == 0:
            return torch.zeros_like(hidden)
        bsz, seq_len, _, _ = sliding_values.shape
        values_flat = sliding_values.reshape(bsz, seq_len, -1)
        summary = causal_prefix_summary(values_flat, self.d_model)
        compressed = torch.tanh(self.evict_compressor(summary))
        write_val = self.write_val_proj(summary)
        master_gate = torch.sigmoid(self.evict_importance(summary))
        feedback = self.memory_out_proj(compressed + write_val) * master_gate
        return feedback

    def split_anchor_qkv_projection(self, sliding_projection: nn.Module, global_projection: nn.Module) -> dict[str, tuple[int, int]]:
        if self.anchor_qkv_proj is None:
            raise RuntimeError('anchor_qkv_proj has already been split')
        sliding_dim = self.h_sliding * self.d_head
        global_dim = self.h_global * self.d_head
        if sliding_dim + global_dim != self.d_model:
            raise RuntimeError(f'invalid anchor QKV split: sliding_dim={sliding_dim}, global_dim={global_dim}, d_model={self.d_model}')
        expected_sliding = (3 * sliding_dim, self.d_model)
        expected_global = (3 * global_dim, self.d_model)
        if tuple(sliding_projection.weight.shape) != expected_sliding:
            raise RuntimeError(f'sliding QKV weight shape {tuple(sliding_projection.weight.shape)} does not match {expected_sliding}')
        if tuple(global_projection.weight.shape) != expected_global:
            raise RuntimeError(f'global QKV weight shape {tuple(global_projection.weight.shape)} does not match {expected_global}')
        offsets = {
            'q_sliding': (0, sliding_dim),
            'q_global': (sliding_dim, self.d_model),
            'k_sliding': (self.d_model, self.d_model + sliding_dim),
            'k_global': (self.d_model + sliding_dim, 2 * self.d_model),
            'v_sliding': (2 * self.d_model, 2 * self.d_model + sliding_dim),
            'v_global': (2 * self.d_model + sliding_dim, 3 * self.d_model),
        }
        source = self.anchor_qkv_proj.weight.detach()
        sliding_rows = torch.cat([source[start:end] for key, (start, end) in offsets.items() if key.endswith('sliding')], dim=0)
        global_rows = torch.cat([source[start:end] for key, (start, end) in offsets.items() if key.endswith('global')], dim=0)
        with torch.no_grad():
            sliding_projection.weight.copy_(sliding_rows.to(device=sliding_projection.weight.device, dtype=sliding_projection.weight.dtype))
            global_projection.weight.copy_(global_rows.to(device=global_projection.weight.device, dtype=global_projection.weight.dtype))
        self.anchor_sliding_qkv_proj = sliding_projection
        self.anchor_global_qkv_proj = global_projection
        self.anchor_qkv_proj = None
        self.anchor_qkv_split_offsets = offsets
        return offsets

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        delta_input = self.delta_norm(hidden)
        if self.training:
            delta_update = self._chunkwise_parallel_deltanet(delta_input)
        else:
            delta_update = self._sequential_recurrent_deltanet(delta_input)
        hidden = hidden + delta_update
        
        anchor_update, sliding_values = self._split_head_anchor_attention(self.anchor_norm(hidden))
        hidden = hidden + anchor_update
        hidden = hidden + self._causal_prefix_feedback(hidden, sliding_values)
        
        ffn_input = self.ffn_norm(hidden)
        ffn_out = self.ffn(ffn_input)
        return hidden + ffn_out



class SamatNextCLForCausalLM(nn.Module):
    """Parameter-unrolled SamatNext-CL causal language model."""

    memory_module_names = (
        "evict_compressor",
        "evict_importance",
        "write_val_proj",
        "memory_out_proj",
    )
    inactive_experimental_memory_fields = (
        "h_mem",
        "d_mem_key",
        "d_mem_value",
        "topk_write_ratio",
    )

    def __init__(
        self,
        vocab_size: int = 50304,
        num_layers: int = 24,
        d_model: int = 1024,
        num_heads: int = 16,
        sliding_window: int = 1024,
        chunk_size: int = 128,
        ffn_dim: Optional[int] = 3584,
        h_mem: int = 24,
        d_mem_key: int = 64,
        d_mem_value: int = 64,
        d_evict: int = 256,
        topk_write_ratio: float = 0.15,
        enable_eviction_feedback: bool = True,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.sliding_window = sliding_window
        self.chunk_size = chunk_size
        self.ffn_dim = ffn_dim if ffn_dim is not None else int(3.5 * d_model)
        self.h_mem = h_mem
        self.d_mem_key = d_mem_key
        self.d_mem_value = d_mem_value
        self.d_evict = d_evict
        self.topk_write_ratio = topk_write_ratio
        self.enable_eviction_feedback = enable_eviction_feedback
        self.model_semantics_version = MODEL_SEMANTICS_VERSION
        self.inactive_experimental_memory_config = {
            "h_mem": h_mem,
            "d_mem_key": d_mem_key,
            "d_mem_value": d_mem_value,
            "topk_write_ratio": topk_write_ratio,
        }
        self.gradient_checkpointing = False

        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList()
        for idx in range(num_layers):
            is_even = (idx % 2 == 0)
            block_cls = DeltaBlock if is_even else AnchorBlock
            layer_feedback = enable_eviction_feedback and (idx % 4 == 0)
            self.layers.append(
                block_cls(
                    d_model=d_model,
                    num_heads=num_heads,
                    sliding_window=sliding_window,
                    chunk_size=chunk_size,
                    h_sliding=math.ceil(num_heads * 0.8),
                    ffn_dim=self.ffn_dim,
                    d_evict=d_evict,
                    enable_eviction_feedback=layer_feedback,
                )
            )
        norm_cls = getattr(nn, 'RMSNorm', nn.LayerNorm)
        self.final_norm = norm_cls(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.apply(self._init_weights)
        self.lm_head.weight = self.token_embeddings.weight

    @property
    def embed_tokens(self) -> nn.Embedding:
        return self.token_embeddings

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        self.gradient_checkpointing = bool(enabled)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward_hidden(self, input_ids: Tensor) -> Tensor:
        hidden = self.token_embeddings(input_ids)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                hidden = checkpoint(layer, hidden, use_reentrant=False)
            else:
                hidden = layer(hidden)
        return self.final_norm(hidden)

    def forward(self, input_ids: Tensor, labels: Optional[Tensor] = None) -> Tensor | tuple[Tensor, Tensor]:
        logits = self.lm_head(self.forward_hidden(input_ids))
        if labels is None:
            return logits
        loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), labels.reshape(-1))
        return logits, loss

    def parameter_groups_for_optimizer(self, base_lr: float) -> list[dict[str, object]]:
        core_params = []
        memory_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if any(memory_name in name for memory_name in self.memory_module_names):
                memory_params.append(param)
            else:
                core_params.append(param)
        return [
            {"params": core_params, "lr": base_lr},
            {"params": memory_params, "lr": base_lr},
        ]

    def iter_core_and_memory_grad_norms(self) -> Iterator[tuple[str, Tensor]]:
        for name, param in self.named_parameters():
            if param.grad is None:
                continue
            group = "memory" if any(
                memory_name in name for memory_name in self.memory_module_names
            ) else "core"
            yield group, param.grad.detach().norm()


def convert_experimental_fp8_linears(
    model: SamatNextCLForCausalLM,
    fp8_factory: Callable[[nn.Linear, str, int], nn.Module],
) -> tuple[list[PrecisionConversionRecord], bool]:
    before_count = sum(param.numel() for param in model.parameters())
    records: list[PrecisionConversionRecord] = []
    param_diff = 0

    def record(
        name: str,
        original: nn.Module,
        converted: nn.Module,
        status: str,
        precision: str,
        reason: str,
    ) -> None:
        records.append(
            PrecisionConversionRecord(
                module_name=name,
                original_backend=type(original).__name__,
                converted_backend=type(converted).__name__,
                status=status,
                target_precision=precision,
                parameter_count=sum(param.numel() for param in converted.parameters()),
                reason=reason,
            )
        )

    for layer_index, block in enumerate(model.layers):
        prefix = f"layers.{layer_index}"
        
        # 1. Handle Anchor Attention (only present in AnchorBlock)
        if hasattr(block, "anchor_qkv_proj") and block.anchor_qkv_proj is not None:
            source_qkv = block.anchor_qkv_proj
            source_qkv_params = sum(p.numel() for p in source_qkv.parameters())
            
            sliding_dim = block.h_sliding * block.d_head
            global_dim = block.h_global * block.d_head
            sliding = fp8_factory(
                source_qkv,
                f"{prefix}.anchor_sliding_qkv_proj",
                3 * sliding_dim,
            )
            global_projection = nn.Linear(
                block.d_model,
                3 * global_dim,
                bias=False,
                device=block.anchor_out_proj.weight.device,
                dtype=torch.float16,
            )
            block.split_anchor_qkv_projection(sliding, global_projection)
            
            new_params = sum(p.numel() for p in sliding.parameters()) + sum(p.numel() for p in global_projection.parameters())
            param_diff += (new_params - source_qkv_params)
            
            record(
                f"{prefix}.anchor_sliding_qkv_proj",
                source_qkv,
                sliding,
                "converted",
                "FP8 E4M3",
                f"sliding heads={block.h_sliding}; Q/K/V row slices isolated",
            )
            record(
                f"{prefix}.anchor_global_qkv_proj",
                source_qkv,
                global_projection,
                "protected",
                "FP16",
                f"global heads={block.h_global}; protected context path",
            )

        # 2. Convert eligible FFN and Attention projections
        for attribute in (
            "ffn_gate_proj",
            "ffn_up_proj",
            "ffn_down_proj",
        ):
            if hasattr(block, "ffn") and hasattr(block.ffn, attribute):
                source = getattr(block.ffn, attribute)
                converted = fp8_factory(source, f"{prefix}.{attribute}", source.out_features)
                
                # Check if this is a LinearFacade wrapper and we should update the raw weights instead of adding converted module
                if type(source).__name__ == "LinearFacade":
                    # Update parameter count tracking
                    source_params = source.weight.numel()
                    new_params = sum(p.numel() for p in converted.parameters())
                    param_diff += (new_params - source_params)
                    
                    # Store converted linear in a special sub-module slot so it gets run/represented properly
                    # and remove the old parameter from the parameter list to avoid duplicates
                    if attribute == "ffn_gate_proj":
                        block.ffn.ffn_gate_proj_converted = converted
                        if hasattr(block.ffn, "gate_weights"):
                            delattr(block.ffn, "gate_weights")
                    elif attribute == "ffn_up_proj":
                        block.ffn.ffn_up_proj_converted = converted
                        if hasattr(block.ffn, "up_weights"):
                            delattr(block.ffn, "up_weights")
                    elif attribute == "ffn_down_proj":
                        block.ffn.ffn_down_proj_converted = converted
                        if hasattr(block.ffn, "down_weights"):
                            delattr(block.ffn, "down_weights")
                else:
                    setattr(block.ffn, attribute, converted)
                
                record(
                    f"{prefix}.{attribute}",
                    source,
                    converted,
                    "converted",
                    "FP8 E4M3",
                    "eligible dense projection",
                )

        # anchor_out_proj is a direct attribute of AnchorBlock
        if hasattr(block, "anchor_out_proj") and block.anchor_out_proj is not None:
            source = block.anchor_out_proj
            converted = fp8_factory(source, f"{prefix}.anchor_out_proj", source.out_features)
            block.anchor_out_proj = converted
            record(
                f"{prefix}.anchor_out_proj",
                source,
                converted,
                "converted",
                "FP8 E4M3",
                "eligible dense projection",
            )

        # 3. Protect other paths
        for attribute in (
            "delta_qkv_proj",
            "delta_beta_proj",
            "delta_out_proj",
            "delta_residual_gate",
            "delta_norm",
            "anchor_norm",
            "ffn_norm",
        ):
            if hasattr(block, attribute) and getattr(block, attribute) is not None:
                module = getattr(block, attribute)
                record(
                    f"{prefix}.{attribute}",
                    module,
                    module,
                    "protected",
                    "FP32/BF16 autocast",
                    "DeltaNet, normalization, or residual-protection boundary",
                )

        for attribute in (
            "evict_compressor",
            "evict_importance",
            "write_val_proj",
            "memory_out_proj",
        ):
            if hasattr(block, attribute) and getattr(block, attribute) is not None:
                module = getattr(block, attribute)
                record(
                    f"{prefix}.{attribute}",
                    module,
                    module,
                    "protected",
                    "FP32/BF16 autocast",
                    "memory feedback paths",
                )

    record(
        "token_embeddings",
        model.token_embeddings,
        model.token_embeddings,
        "protected",
        "FP32/BF16 autocast",
        "embedding and tied output weight",
    )
    record(
        "final_norm",
        model.final_norm,
        model.final_norm,
        "protected",
        "FP32/BF16 autocast",
        "normalization",
    )
    record(
        "lm_head",
        model.lm_head,
        model.lm_head,
        "protected",
        "FP32/BF16 autocast",
        "weight-tied streamed CE projection",
    )

    expected_after = before_count + param_diff
    after_count = sum(param.numel() for param in model.parameters())
    if after_count != expected_after:
        raise RuntimeError(
            f"FP8 conversion changed parameter count: {expected_after} -> {after_count}"
        )
    rope_present = any("rope" in name.lower() or "rotary" in name.lower() for name, _ in model.named_modules())
    return records, rope_present
