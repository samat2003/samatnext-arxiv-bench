import copy

import pytest
import torch

from samatnext_550m_model import SamatNextCLBlock


def make_block(device: torch.device) -> SamatNextCLBlock:
    return SamatNextCLBlock(
        d_model=8,
        num_heads=2,
        sliding_window=4,
        chunk_size=2,
        ffn_dim=16,
        d_evict=4,
    ).to(device)


def compare_delta_paths(device: torch.device, use_autocast: bool) -> None:
    torch.manual_seed(12)
    chunked = make_block(device)
    sequential = copy.deepcopy(chunked)
    chunked_input = torch.randn(2, 4, 8, device=device, requires_grad=True)
    sequential_input = chunked_input.detach().clone().requires_grad_(True)
    upstream = torch.randn_like(chunked_input)

    with torch.amp.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=use_autocast,
    ):
        chunked_out = chunked._chunkwise_parallel_deltanet(chunked_input)
        sequential_out = sequential._sequential_recurrent_deltanet(sequential_input)
        chunked_loss = (chunked_out * upstream).float().sum()
        sequential_loss = (sequential_out * upstream).float().sum()
    chunked_loss.backward()
    sequential_loss.backward()

    tolerance = dict(rtol=3e-2, atol=3e-2) if use_autocast else dict(rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(chunked_out, sequential_out, **tolerance)
    torch.testing.assert_close(chunked_input.grad, sequential_input.grad, **tolerance)
    for (name_a, param_a), (name_b, param_b) in zip(
        chunked.named_parameters(), sequential.named_parameters()
    ):
        assert name_a == name_b
        if param_a.grad is not None or param_b.grad is not None:
            torch.testing.assert_close(param_a.grad, param_b.grad, **tolerance)


def test_fp32_chunkwise_forward_and_gradient_parity():
    compare_delta_paths(torch.device("cpu"), use_autocast=False)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
    reason="CUDA BF16 is unavailable",
)
def test_cuda_bf16_chunkwise_forward_and_gradient_parity():
    compare_delta_paths(torch.device("cuda"), use_autocast=True)
