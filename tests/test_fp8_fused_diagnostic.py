import csv
import json
from pathlib import Path

import pytest
import torch

from production_train_550m import (
    SMOKE_TELEMETRY_COLUMNS,
    StepTelemetryWriter,
    build_arg_parser,
    compile_hidden_model,
    compile_loss_function,
    fp8_capability_check,
    smoke_telemetry_row,
    telemetry_graph_breaks,
    validate_smoke_mode_args,
)
from samatnext_550m_model import (
    SamatNextCLForCausalLM,
    convert_experimental_fp8_linears,
)


def test_fp8_cli_flags_exist_and_parse():
    args = build_arg_parser().parse_args(
        [
            "--smoke-test",
            "--smoke-full-model",
            "--diagnostic-10step",
            "--experimental-fp8-linears",
            "--activation-checkpointing",
            "--compile-mode",
            "max-autotune",
        ]
    )
    validate_smoke_mode_args(args)
    assert args.experimental_fp8_linears
    assert args.compile_mode == "max-autotune"
    assert build_arg_parser().parse_args(["--fp8-capability-check"]).fp8_capability_check
    assert build_arg_parser().parse_args(["--fp8-linear-microbench"]).fp8_linear_microbench


def test_experimental_fp8_requires_full_ten_step_path():
    args = build_arg_parser().parse_args(
        ["--smoke-test", "--experimental-fp8-linears"]
    )
    with pytest.raises(ValueError, match="full --diagnostic-10step"):
        validate_smoke_mode_args(args)


def test_graph_breaks_are_na_when_compile_is_disabled():
    assert telemetry_graph_breaks(False) is None


def test_jsonl_and_csv_records_are_single_line_and_schema_stable(tmp_path):
    memory = {
        "allocated": 1.0,
        "reserved": 2.0,
        "peak_allocated": 3.0,
        "peak_reserved": 4.0,
        "oom_count": 0.0,
        "allocation_retry_count": 0.0,
    }
    row = smoke_telemetry_row(
        step=1,
        loss=1.0,
        loss_delta=None,
        grad_norm=2.0,
        max_grad_abs=0.1,
        nonfinite_loss_count=0,
        nonfinite_gradient_count=0,
        step_seconds=1.0,
        tokens_per_step=16,
        timings={},
        memory_mib=memory,
        graph_breaks=None,
        compile_enabled=False,
        activation_checkpointing=True,
        fp8_status="available",
    )
    jsonl_path = tmp_path / "telemetry.jsonl"
    csv_path = tmp_path / "telemetry.csv"
    writer = StepTelemetryWriter(jsonl_path, csv_path)
    writer.write(row)
    writer.close()

    json_lines = jsonl_path.read_text().splitlines()
    assert len(json_lines) == 1
    record = json.loads(json_lines[0])
    assert list(record) == list(SMOKE_TELEMETRY_COLUMNS)
    assert record["graph_breaks"] == "n/a"
    assert record["compile"] == "disabled"
    assert record["activation_checkpointing"] == "enabled"
    assert record["checkpoint_save"] == "disabled"
    assert record["fp8_status"] == "available"

    with csv_path.open(newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == 1
    assert csv_rows[0]["graph_breaks"] == "n/a"
    assert len(csv_path.read_text().splitlines()) == 2


def test_missing_cuda_capability_is_reported_without_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    result = fp8_capability_check()
    assert result["status"] == "unavailable"
    assert result["micro_unit"]["passed"] is False


def test_qkv_split_offsets_and_protected_precision_map():
    model = SamatNextCLForCausalLM(
        vocab_size=128,
        num_layers=2,
        d_model=64,
        num_heads=16,
        sliding_window=32,
        chunk_size=8,
        ffn_dim=128,
        d_evict=16,
    )
    block = model.layers[1]
    before_count = sum(param.numel() for param in model.parameters())
    # Keep track of original MLAKVProjection to calculate expected parameter count
    original_qkv = block.anchor_qkv_proj
    original_qkv_params = sum(p.numel() for p in original_qkv.parameters())
    original = block.anchor_qkv_proj.weight.detach().clone()

    def mock_fp8_factory(source, name, out_features):
        del name
        converted = torch.nn.Linear(
            source.in_features,
            out_features,
            bias=source.bias is not None,
        )
        if out_features == source.out_features:
            converted.load_state_dict(source.state_dict())
        return converted

    records, rope_present = convert_experimental_fp8_linears(
        model,
        mock_fp8_factory,
    )
    sliding_dim = block.h_sliding * block.d_head
    expected_sliding = torch.cat(
        (
            original[:sliding_dim],
            original[64 : 64 + sliding_dim],
            original[128 : 128 + sliding_dim],
        )
    )
    expected_global = torch.cat(
        (
            original[sliding_dim:64],
            original[64 + sliding_dim : 128],
            original[128 + sliding_dim : 192],
        )
    )
    assert block.anchor_qkv_proj is None
    assert torch.equal(block.anchor_sliding_qkv_proj.weight, expected_sliding)
    assert torch.equal(
        block.anchor_global_qkv_proj.weight.float(),
        expected_global.half().float(),
    )
    expected_after = before_count - original_qkv_params + block.anchor_sliding_qkv_proj.weight.numel() + block.anchor_global_qkv_proj.weight.numel()
    assert sum(param.numel() for param in model.parameters()) == expected_after
    assert model.lm_head.weight is model.token_embeddings.weight
    assert rope_present is False

    by_name = {record.module_name: record for record in records}
    assert by_name["layers.1.anchor_sliding_qkv_proj"].target_precision == "FP8 E4M3"
    assert by_name["layers.1.anchor_global_qkv_proj"].target_precision == "FP16"
    for protected in (
        "layers.0.delta_qkv_proj",
        "layers.1.memory_out_proj",
        "layers.0.ffn_norm",
        "layers.1.ffn_norm",
        "token_embeddings",
        "lm_head",
    ):
        assert by_name[protected].status == "protected"


def test_loss_compile_path_is_explicit_and_callable():
    compiled, state = compile_loss_function(enabled=True, backend="eager", mode=None)
    hidden = torch.randn(4, 8, requires_grad=True)
    weight = torch.randn(16, 8, requires_grad=True)
    targets = torch.randint(0, 16, (4,))
    loss = compiled(hidden, weight, targets, block_vocab=4, ignore_index=-100)
    loss.backward()
    assert torch.isfinite(loss)
    assert state == {"enabled": True, "backend": "eager", "mode": None}


@pytest.mark.parametrize("compile_target", ["model", "loss"])
def test_inductor_compile_disables_cudagraph_output_reuse(monkeypatch, compile_target):
    captured = {}

    def fake_compile(target, **kwargs):
        captured.update(kwargs)
        return target

    monkeypatch.setattr(torch, "compile", fake_compile)
    if compile_target == "model":
        compile_hidden_model(
            torch.nn.Identity(), enabled=True, backend="inductor", mode="max-autotune"
        )
    else:
        compile_loss_function(enabled=True, backend="inductor", mode="max-autotune")

    assert captured["mode"] == "max-autotune-no-cudagraphs"
    assert "options" not in captured


def test_production_source_has_no_top_level_transformer_engine_imports():
    """Guard against accidental boot-time TE imports in the production script.

    Transformer Engine is an *optional* dependency that may not be installed.
    Any ``import transformer_engine`` or ``from transformer_engine`` that appears
    at module-level (i.e. not guarded by a try/except or hidden inside a
    function/string) would cause an ImportError at startup on machines without TE.

    The allowed patterns are:
    - ``import transformer_engine.xxx`` inside subprocess code strings (f-strings).
    - ``import transformer_engine.xxx`` inside function bodies guarded by try/except.

    This test uses a simple AST walk to flag only actual top-level Import nodes.
    """
    import ast

    source = Path("production_train_550m.py").read_text()
    tree = ast.parse(source)

    top_level_te_imports: list[str] = []
    for node in ast.iter_child_nodes(tree):
        # Only inspect statements at module scope (not inside functions or classes).
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            else:
                names = [node.module or ""]
            for name in names:
                if name.startswith("transformer_engine"):
                    top_level_te_imports.append(
                        f"line {node.lineno}: {ast.unparse(node)}"
                    )

    assert not top_level_te_imports, (
        "Found unexpected top-level transformer_engine import(s) in "
        "production_train_550m.py. All TE usage must be guarded inside "
        "function-level try/except blocks or subprocess code strings:\n"
        + "\n".join(top_level_te_imports)
    )
