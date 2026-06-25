import math

import pytest
import torch
import torch.nn.functional as F

from samatnext_fused_ops import fused_linear_cross_entropy


@pytest.mark.parametrize("block_vocab", [3, 5, 16])
@pytest.mark.parametrize(
    "targets",
    [
        torch.tensor([0, 2, 6, 8, 4, 1]),
        torch.tensor([0, -100, 6, -100, 4, 1]),
        torch.full((6,), -100),
    ],
    ids=["all_valid", "some_ignored", "all_ignored"],
)
def test_fused_ce_loss_and_gradient_parity(block_vocab: int, targets: torch.Tensor):
    torch.manual_seed(10)
    hidden = torch.randn(6, 7, requires_grad=True)
    weight = torch.randn(9, 7, requires_grad=True)
    ref_hidden = hidden.detach().clone().requires_grad_(True)
    ref_weight = weight.detach().clone().requires_grad_(True)

    actual = fused_linear_cross_entropy(
        hidden,
        weight,
        targets,
        block_vocab=block_vocab,
        ignore_index=-100,
    )
    expected = F.cross_entropy(ref_hidden @ ref_weight.t(), targets, ignore_index=-100)
    if targets.eq(-100).all():
        assert math.isnan(float(actual.detach()))
        assert math.isnan(float(expected.detach()))
    else:
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    actual.backward()
    expected.backward()
    torch.testing.assert_close(hidden.grad, ref_hidden.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(weight.grad, ref_weight.grad, rtol=1e-5, atol=1e-6)


def test_fused_ce_rejects_out_of_range_target():
    hidden = torch.randn(3, 4, requires_grad=True)
    weight = torch.randn(5, 4, requires_grad=True)
    with pytest.raises((ValueError, RuntimeError)):
        fused_linear_cross_entropy(hidden, weight, torch.tensor([0, 5, 1]), block_vocab=2)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
    reason="CUDA BF16 is unavailable",
)
def test_fused_ce_cuda_bf16_parity():
    torch.manual_seed(17)
    hidden = torch.randn(8, 16, device="cuda", requires_grad=True)
    weight = torch.randn(23, 16, device="cuda", requires_grad=True)
    targets = torch.tensor([0, 2, -100, 8, 22, 1, -100, 7], device="cuda")
    ref_hidden = hidden.detach().clone().requires_grad_(True)
    ref_weight = weight.detach().clone().requires_grad_(True)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        actual = fused_linear_cross_entropy(hidden, weight, targets, block_vocab=7)
        expected = F.cross_entropy(ref_hidden @ ref_weight.t(), targets, ignore_index=-100)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(hidden.grad, ref_hidden.grad, rtol=4e-2, atol=4e-2)
    torch.testing.assert_close(weight.grad, ref_weight.grad, rtol=4e-2, atol=4e-2)
