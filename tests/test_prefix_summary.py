import torch

from samatnext_550m_model import causal_prefix_summary


def triangular_reference(values: torch.Tensor, output_dim: int) -> torch.Tensor:
    seq_len = values.shape[1]
    lower = torch.tril(torch.ones(seq_len, seq_len, dtype=values.dtype, device=values.device))
    sums = torch.bmm(lower.expand(values.shape[0], -1, -1), values)
    counts = torch.arange(1, seq_len + 1, device=values.device, dtype=torch.float32)
    summary = sums * torch.rsqrt(counts).to(values.dtype).view(1, -1, 1)
    if summary.shape[-1] < output_dim:
        repeats = (output_dim + summary.shape[-1] - 1) // summary.shape[-1]
        summary = summary.repeat(1, 1, repeats)
    return summary[:, :, :output_dim]


def test_cumsum_prefix_matches_triangular_reference():
    torch.manual_seed(3)
    values = torch.randn(2, 7, 5)
    actual = causal_prefix_summary(values, output_dim=12)
    expected = triangular_reference(values, output_dim=12)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_prefix_summary_is_causal():
    values = torch.randn(1, 6, 4)
    changed = values.clone()
    changed[:, 4:] = torch.randn_like(changed[:, 4:])
    first = causal_prefix_summary(values, output_dim=4)
    second = causal_prefix_summary(changed, output_dim=4)
    torch.testing.assert_close(first[:, :4], second[:, :4], rtol=0.0, atol=0.0)
