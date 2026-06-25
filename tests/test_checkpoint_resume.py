from pathlib import Path
import random

import numpy as np
import pytest
import torch

from production_train_550m import (
    CheckpointCompatibilityError,
    CooldownMetadata,
    MemmapTokenStream,
    apply_checkpoint_payload,
    load_checkpoint_payload,
    save_checkpoint,
)
from samatnext_550m_model import SamatNextCLForCausalLM


TINY_CONFIG = {
    "vocab_size": 32,
    "num_layers": 1,
    "d_model": 8,
    "num_heads": 2,
    "sliding_window": 4,
    "chunk_size": 2,
    "ffn_dim": 16,
    "d_evict": 4,
}


def make_stream(path: Path) -> MemmapTokenStream:
    path.mkdir(exist_ok=True)
    np.asarray(list(range(32)) * 20, dtype=np.uint16).tofile(path / "shard_000.bin")
    return MemmapTokenStream(path, batch_size=1, seq_len=4)


def make_model_optimizer():
    model = SamatNextCLForCausalLM(**TINY_CONFIG).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return model, optimizer


def one_step(model, optimizer, stream):
    x, y = stream.next_batch(torch.device("cpu"))
    optimizer.zero_grad(set_to_none=True)
    _, loss = model(x, labels=y)
    random_scale = torch.rand(()) + float(np.random.random()) * 0.01 + random.random() * 0.01
    (loss * random_scale).backward()
    optimizer.step()
    return loss.detach()


def assert_nested_close(first, second):
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_nested_close(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for left, right in zip(first, second):
            assert_nested_close(left, right)
    else:
        assert first == second


def test_checkpoint_round_trip_matches_uninterrupted_run(tmp_path: Path):
    torch.manual_seed(21)
    np.random.seed(21)
    random.seed(21)
    uninterrupted, uninterrupted_optimizer = make_model_optimizer()
    stream_a = make_stream(tmp_path / "data")
    one_step(uninterrupted, uninterrupted_optimizer, stream_a)
    one_step(uninterrupted, uninterrupted_optimizer, stream_a)

    cooldown = CooldownMetadata("1F", 500, 600, 100, 640, "stable-hash")
    checkpoint = save_checkpoint(
        tmp_path / "checkpoints",
        2,
        uninterrupted,
        uninterrupted_optimizer,
        stream_a,
        cooldown,
        micro_batch_size=1,
        gradient_accumulation_steps=1,
        compile_state={"enabled": False, "backend": None, "mode": None},
        model_config=TINY_CONFIG,
        prune=False,
    )
    uninterrupted_next_loss = one_step(uninterrupted, uninterrupted_optimizer, stream_a)

    torch.manual_seed(999)
    np.random.seed(999)
    random.seed(999)
    resumed, resumed_optimizer = make_model_optimizer()
    stream_b = make_stream(tmp_path / "data")
    payload = load_checkpoint_payload(checkpoint)
    step = apply_checkpoint_payload(
        payload,
        resumed,
        resumed_optimizer,
        stream_b,
        cooldown,
        expected_model_config=TINY_CONFIG,
    )
    resumed_next_loss = one_step(resumed, resumed_optimizer, stream_b)

    assert step == 2
    torch.testing.assert_close(uninterrupted_next_loss, resumed_next_loss, rtol=0.0, atol=0.0)
    for first, second in zip(uninterrupted.parameters(), resumed.parameters()):
        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    assert_nested_close(uninterrupted_optimizer.state_dict(), resumed_optimizer.state_dict())
    assert stream_a.state_dict() == stream_b.state_dict()


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"checkpoint_schema_version": 2},
        {"checkpoint_schema_version": 2, "model_semantics_version": 1},
    ],
)
def test_old_or_unversioned_checkpoint_is_rejected(tmp_path: Path, payload):
    path = tmp_path / "old.pt"
    torch.save(payload, path)
    with pytest.raises(CheckpointCompatibilityError):
        load_checkpoint_payload(path)
    with pytest.raises(NotImplementedError, match="migration is not implemented"):
        load_checkpoint_payload(path, allow_checkpoint_migration=True)


def test_checkpoint_missing_stream_state_is_rejected(tmp_path: Path):
    model, optimizer = make_model_optimizer()
    stream = make_stream(tmp_path / "data")
    cooldown = CooldownMetadata("1F", 500, 600, 100, 640, "stable-hash")
    path = save_checkpoint(
        tmp_path / "checkpoints",
        0,
        model,
        optimizer,
        stream,
        cooldown,
        micro_batch_size=1,
        gradient_accumulation_steps=1,
        compile_state={"enabled": False},
        model_config=TINY_CONFIG,
        prune=False,
    )
    payload = torch.load(path, weights_only=False)
    del payload["data_stream"]
    broken = tmp_path / "broken.pt"
    torch.save(payload, broken)
    with pytest.raises(CheckpointCompatibilityError, match="data_stream"):
        load_checkpoint_payload(broken)
