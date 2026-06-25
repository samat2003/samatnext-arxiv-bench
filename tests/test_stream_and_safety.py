import json
from pathlib import Path

import numpy as np
import pytest
import torch

from production_train_550m import (
    CheckpointCompatibilityError,
    CooldownMetadata,
    MemmapTokenStream,
    NonFiniteTrainingError,
    attention_configuration_warning,
    build_arg_parser,
    load_and_validate_cooldown_metadata,
    safe_optimizer_step,
    save_checkpoint,
    validate_run_args,
)


def write_shard(path: Path, values: list[int]) -> None:
    np.asarray(values, dtype=np.uint16).tofile(path)


def test_stream_uses_exact_flat_bt_plus_one_layout(tmp_path: Path):
    write_shard(tmp_path / "shard_000.bin", list(range(40)))
    stream = MemmapTokenStream(tmp_path, batch_size=2, seq_len=3)
    x1, y1 = stream.next_batch(torch.device("cpu"))
    x2, y2 = stream.next_batch(torch.device("cpu"))
    assert x1.flatten().tolist() == list(range(6))
    assert y1.flatten().tolist() == list(range(1, 7))
    assert x2.flatten().tolist() == list(range(6, 12))
    assert y2.flatten().tolist() == list(range(7, 13))
    assert stream.absolute_token_offset() == 12


def test_stream_crosses_shards_without_skips(tmp_path: Path):
    write_shard(tmp_path / "shard_000.bin", list(range(5)))
    write_shard(tmp_path / "shard_001.bin", list(range(5, 20)))
    stream = MemmapTokenStream(tmp_path, batch_size=2, seq_len=3)
    x, y = stream.next_batch(torch.device("cpu"))
    assert x.flatten().tolist() == list(range(6))
    assert y.flatten().tolist() == list(range(1, 7))
    assert stream.state_dict()["shard_path"] == "shard_001.bin"
    assert stream.state_dict()["token_offset"] == 1


def valid_metadata() -> dict[str, object]:
    return {
        "tokens_written": 100,
        "cooldown_start_token": 80,
        "phases": [
            {
                "phase_id": "1F",
                "start_token": 80,
                "end_token": 100,
                "tokens_written": 20,
                "status": "complete",
            }
        ],
    }


def test_missing_or_incomplete_cooldown_metadata_is_fatal(tmp_path: Path):
    with pytest.raises(ValueError):
        load_and_validate_cooldown_metadata(tmp_path)
    (tmp_path / "metadata.json").write_text(json.dumps({"phases": []}))
    with pytest.raises(ValueError):
        load_and_validate_cooldown_metadata(tmp_path)
    metadata = valid_metadata()
    metadata["phases"][0]["status"] = "partial"
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        load_and_validate_cooldown_metadata(tmp_path)


def test_valid_cooldown_metadata(tmp_path: Path):
    (tmp_path / "metadata.json").write_text(json.dumps(valid_metadata()))
    result = load_and_validate_cooldown_metadata(tmp_path)
    assert result.start_token == 80
    assert len(result.metadata_sha256) == 64


def test_start_step_without_resume_fails():
    args = build_arg_parser().parse_args(["--fresh-run", "--start-step", "3"])
    with pytest.raises(ValueError, match="requires --resume"):
        validate_run_args(args)


def test_attention_warning_is_explicit():
    assert attention_configuration_warning(512, 1024) is not None
    assert attention_configuration_warning(512, 128) is None


class CountingSGD(torch.optim.SGD):
    def __init__(self, params):
        super().__init__(params, lr=0.1)
        self.step_calls = 0

    def step(self, closure=None):
        self.step_calls += 1
        return super().step(closure)


def test_nonfinite_loss_skips_optimizer_step():
    model = torch.nn.Linear(2, 2)
    optimizer = CountingSGD(model.parameters())
    with pytest.raises(NonFiniteTrainingError):
        safe_optimizer_step(model, optimizer, torch.tensor(float("nan")))
    assert optimizer.step_calls == 0


def test_nonfinite_gradient_skips_optimizer_step():
    model = torch.nn.Linear(2, 2)
    optimizer = CountingSGD(model.parameters())
    for param in model.parameters():
        param.grad = torch.full_like(param, float("inf"))
    with pytest.raises(NonFiniteTrainingError):
        safe_optimizer_step(model, optimizer, torch.tensor(1.0))
    assert optimizer.step_calls == 0


def test_nonfinite_model_cannot_be_checkpointed(tmp_path: Path):
    write_shard(tmp_path / "shard_000.bin", list(range(100)))
    stream = MemmapTokenStream(tmp_path, 1, 2)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    with torch.no_grad():
        model.weight[0, 0] = float("nan")
    cooldown = CooldownMetadata("1F", 80, 100, 20, 100, "hash")
    with pytest.raises(NonFiniteTrainingError):
        save_checkpoint(
            tmp_path / "checkpoints",
            1,
            model,
            optimizer,
            stream,
            cooldown,
            micro_batch_size=1,
            gradient_accumulation_steps=1,
            compile_state={"enabled": False},
            model_config={"tiny": True},
        )
    assert not list((tmp_path / "checkpoints").glob("*.pt"))
