import importlib
from pathlib import Path

from verify_deltanet_triton_kernel import PRODUCTION_VALID


def test_required_runtime_imports_are_available():
    for module in ("torch", "numpy", "bitsandbytes", "triton"):
        assert importlib.import_module(module) is not None


def test_requirements_match_active_runtime_imports():
    requirements = {
        line.strip().split("<", 1)[0].split("=", 1)[0]
        for line in Path("requirements-wsl.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }
    assert {"numpy", "bitsandbytes", "pytest", "triton"} <= requirements


def test_deltanet_triton_candidate_is_not_production_valid():
    assert PRODUCTION_VALID is False
