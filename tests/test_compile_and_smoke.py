import argparse

import pytest
import torch

import production_train_550m as trainer
from production_train_550m import (
    ACTIVE_PARAMETER_COUNT,
    BATCH_SIZE,
    CONFIG,
    GRADIENT_ACCUMULATION_STEPS,
    MICRO_BATCH_SIZE,
    SEQ_LEN,
    SMOKE_TELEMETRY_COLUMNS,
    SamatNextCLHiddenStates,
    SmokePhaseError,
    build_arg_parser,
    compile_graph_break_count,
    compile_hidden_model,
    effective_smoke_steps,
    estimated_dense_training_flops,
    run_full_model_smoke,
    run_micro_smoke,
    smoke_shape,
    smoke_telemetry_header,
    tiny_smoke_config,
    validate_smoke_mode_args,
)
from samatnext_550m_model import SamatNextCLForCausalLM


def test_compile_path_with_explicit_eager_backend():
    model = SamatNextCLForCausalLM(**tiny_smoke_config()).train()
    hidden_model = SamatNextCLHiddenStates(model)
    compiled, state = compile_hidden_model(
        hidden_model,
        enabled=True,
        backend="eager",
        mode=None,
    )
    inputs = torch.randint(0, 128, (1, 8))
    output = compiled(inputs)
    output.sum().backward()
    assert state == {"enabled": True, "backend": "eager", "mode": None}
    assert compile_graph_break_count() is None or compile_graph_break_count() >= 0


def test_compile_failure_is_not_hidden():
    model = torch.nn.Linear(2, 2)
    with pytest.raises(RuntimeError, match="torch.compile setup failed"):
        compile_hidden_model(model, enabled=True, backend="not-a-backend", mode=None)


def test_two_step_cpu_micro_smoke(capsys):
    args = build_arg_parser().parse_args(
        [
            "--smoke-test",
            "--smoke-steps",
            "2",
            "--smoke-device",
            "cpu",
            "--no-compile",
            "--detailed-timing",
        ]
    )
    result = run_micro_smoke(args)
    captured = capsys.readouterr().out
    assert result["updated"] is True
    assert len(result["losses"]) == 2
    assert all(torch.isfinite(torch.tensor(result["losses"])))
    assert all(item for item in result["timings"])
    assert all(item for item in result["phase_memory_mib"])
    assert "smoke_status=passed" in captured
    assert "| allocated_mib |" in captured


def test_full_model_smoke_flag_and_shape_are_active_without_allocation():
    args = build_arg_parser().parse_args(
        ["--smoke-test", "--smoke-full-model", "--smoke-steps", "1"]
    )
    shape = smoke_shape(full_model=args.smoke_full_model)
    assert args.smoke_full_model is True
    assert shape.config == CONFIG
    assert shape.seq_len == SEQ_LEN == 512
    assert shape.global_batch_size == BATCH_SIZE == 32
    assert shape.micro_batch_size == MICRO_BATCH_SIZE == 4
    assert shape.gradient_accumulation_steps == GRADIENT_ACCUMULATION_STEPS == 8
    assert ACTIVE_PARAMETER_COUNT == 432_517_324


@pytest.mark.parametrize("steps", [1, 2, 5])
def test_supported_smoke_step_counts_are_accepted(steps):
    args = build_arg_parser().parse_args(
        ["--smoke-test", "--smoke-steps", str(steps)]
    )
    assert args.smoke_steps == steps


@pytest.mark.parametrize("steps", [0, 6, 9, 10, 100])
def test_unsupported_smoke_step_counts_are_rejected(steps, capsys):
    with pytest.raises(SystemExit) as exc_info:
        build_arg_parser().parse_args(
            ["--smoke-test", "--smoke-steps", str(steps)]
        )
    assert exc_info.value.code == 2
    assert "--smoke-steps must be in [1, 5]" in capsys.readouterr().err


def test_compact_telemetry_has_every_required_column():
    rendered = smoke_telemetry_header()
    assert rendered.startswith("| step |")
    for column in SMOKE_TELEMETRY_COLUMNS:
        assert f"| {column} " in rendered


def test_diagnostic_flag_selects_exactly_ten_steps_without_relaxing_normal_smoke():
    args = build_arg_parser().parse_args(
        [
            "--smoke-test",
            "--smoke-full-model",
            "--diagnostic-10step",
            "--no-compile",
            "--activation-checkpointing",
        ]
    )
    validate_smoke_mode_args(args)
    assert effective_smoke_steps(args) == 10
    assert args.smoke_steps == 2


@pytest.mark.parametrize(
    "missing_flag",
    ["--no-compile", "--activation-checkpointing"],
)
def test_diagnostic_requires_eager_activation_checkpointed_path(missing_flag):
    command = [
        "--smoke-test",
        "--smoke-full-model",
        "--diagnostic-10step",
        "--no-compile",
        "--activation-checkpointing",
    ]
    command.remove(missing_flag)
    args = build_arg_parser().parse_args(command)
    with pytest.raises(ValueError, match="requires"):
        validate_smoke_mode_args(args)


def test_dense_training_flop_estimate_is_clearly_reproducible():
    tokens = 32 * 512
    assert estimated_dense_training_flops(
        parameter_count=ACTIVE_PARAMETER_COUNT,
        tokens=tokens,
    ) == 6 * ACTIVE_PARAMETER_COUNT * tokens


def _full_smoke_args():
    return build_arg_parser().parse_args(
        [
            "--smoke-test",
            "--smoke-full-model",
            "--smoke-steps",
            "1",
            "--no-compile",
        ]
    )


def test_simulated_model_init_oom_reports_phase_and_never_checkpoints(
    monkeypatch,
    capsys,
):
    checkpoint_calls = []
    monkeypatch.setattr(
        trainer,
        "save_checkpoint",
        lambda *args, **kwargs: checkpoint_calls.append((args, kwargs)),
    )

    def fail_model_factory(config):
        del config
        raise torch.OutOfMemoryError("simulated model init OOM")

    result = run_full_model_smoke(
        _full_smoke_args(),
        device_override=torch.device("cpu"),
        model_factory=fail_model_factory,
    )
    captured = capsys.readouterr().out
    assert result.exit_code != 0
    assert result.failure_phase == "model_initialization_device_transfer"
    assert result.oom is True
    assert "phase=model_initialization_device_transfer" in captured
    assert "oom=true" in captured
    assert checkpoint_calls == []


def test_simulated_forward_oom_reports_phase_and_never_checkpoints(
    monkeypatch,
    capsys,
):
    checkpoint_calls = []
    monkeypatch.setattr(
        trainer,
        "save_checkpoint",
        lambda *args, **kwargs: checkpoint_calls.append((args, kwargs)),
    )

    def fail_train_step(*args, **kwargs):
        del args, kwargs
        raise SmokePhaseError("forward", torch.OutOfMemoryError("simulated forward OOM"))

    monkeypatch.setattr(trainer, "train_one_step", fail_train_step)
    result = run_full_model_smoke(
        _full_smoke_args(),
        device_override=torch.device("cpu"),
        model_factory=lambda config: SamatNextCLForCausalLM(**tiny_smoke_config()),
        optimizer_factory=lambda model: torch.optim.SGD(model.parameters(), lr=1e-3),
    )
    captured = capsys.readouterr().out
    assert result.exit_code != 0
    assert result.failure_phase == "forward"
    assert result.oom is True
    assert "phase=forward" in captured
    assert "exit_code=2" in captured
    assert checkpoint_calls == []
