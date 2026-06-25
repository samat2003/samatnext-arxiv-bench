import copy

import torch
from torch import nn

from production_train_550m import CONFIG
from samatnext_550m_model import SamatNextCLBlock, SamatNextCLForCausalLM


def tiny_model() -> SamatNextCLForCausalLM:
    return SamatNextCLForCausalLM(
        vocab_size=64,
        num_layers=2,
        d_model=16,
        num_heads=4,
        sliding_window=4,
        chunk_size=2,
        ffn_dim=32,
        d_evict=8,
    )


def test_active_config_parameter_count_and_weight_tying():
    with torch.device("meta"):
        model = SamatNextCLForCausalLM(**CONFIG)
    assert sum(param.numel() for param in model.parameters()) == 432_517_324
    assert model.lm_head.weight is model.token_embeddings.weight


def test_tiny_forward_shape_and_finiteness():
    model = tiny_model().eval()
    input_ids = torch.randint(0, 64, (2, 4))
    with torch.no_grad():
        logits = model(input_ids)
    assert logits.shape == (2, 4, 64)
    assert torch.isfinite(logits).all()


def test_zeroed_block_is_exact_identity_with_identity_gradient():
    block = SamatNextCLBlock(
        d_model=16,
        num_heads=4,
        sliding_window=4,
        chunk_size=2,
        ffn_dim=32,
        d_evict=8,
    ).train()
    for module in block.modules():
        if isinstance(module, nn.Linear):
            nn.init.zeros_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif type(module).__name__ == "SparseTop1FFN":
            nn.init.zeros_(module.gate_weights)
            nn.init.zeros_(module.up_weights)
            nn.init.zeros_(module.down_weights)
    hidden = torch.randn(2, 4, 16, requires_grad=True)
    output = block(hidden)
    torch.testing.assert_close(output, hidden, rtol=0.0, atol=0.0)
    output.sum().backward()
    torch.testing.assert_close(hidden.grad, torch.ones_like(hidden), rtol=0.0, atol=0.0)


def test_future_tokens_do_not_change_earlier_logits():
    torch.manual_seed(4)
    model = tiny_model().eval()
    first = torch.tensor([[1, 2, 3, 4]])
    second = first.clone()
    second[:, 3] = 19
    with torch.no_grad():
        first_logits = model(first)
        second_logits = model(second)
    torch.testing.assert_close(first_logits[:, :3], second_logits[:, :3], rtol=1e-5, atol=1e-6)


def test_inactive_memory_fields_are_explicit():
    model = tiny_model()
    assert set(model.inactive_experimental_memory_config) == {
        "h_mem",
        "d_mem_key",
        "d_mem_value",
        "topk_write_ratio",
    }


def test_activation_checkpointing_forward_and_gradients_match():
    torch.manual_seed(8)
    eager = tiny_model().train()
    checkpointed = copy.deepcopy(eager).train()
    checkpointed.set_gradient_checkpointing(True)
    input_ids = torch.randint(0, 64, (2, 4))

    eager_loss = eager(input_ids).float().sum()
    checkpointed_loss = checkpointed(input_ids).float().sum()
    eager_loss.backward()
    checkpointed_loss.backward()

    torch.testing.assert_close(eager_loss, checkpointed_loss, rtol=1e-6, atol=1e-6)
    for eager_param, checkpointed_param in zip(eager.parameters(), checkpointed.parameters()):
        torch.testing.assert_close(
            eager_param.grad,
            checkpointed_param.grad,
            rtol=1e-5,
            atol=1e-6,
        )
