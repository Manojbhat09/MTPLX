from __future__ import annotations

import pytest

from mtplx.default_models import (
    DEFAULT_MODEL_VARIANT_ENV,
    is_verified_default_model_ref,
    select_default_model,
)
from mtplx import hardware as hardware_module
from mtplx.hardware import classify_apple_silicon_generation, detect_apple_silicon
from mtplx.profiles import DEFAULT_FP16_HF_MODEL_ID, DEFAULT_HF_MODEL_ID


@pytest.mark.parametrize(
    ("chip", "system", "machine", "expected"),
    [
        ("Apple M1", "Darwin", "arm64", "m1"),
        ("Apple M1 Pro", "Darwin", "arm64", "m1"),
        ("Apple M1 Max", "Darwin", "arm64", "m1"),
        ("Apple M1 Ultra", "Darwin", "arm64", "m1"),
        ("Apple M2", "Darwin", "arm64", "m2"),
        ("Apple M2 Pro", "Darwin", "arm64", "m2"),
        ("Apple M2 Max", "Darwin", "arm64", "m2"),
        ("Apple M2 Ultra", "Darwin", "arm64", "m2"),
        ("Apple M3", "Darwin", "arm64", "m3"),
        ("Apple M3 Max", "Darwin", "arm64", "m3"),
        ("Apple M4", "Darwin", "arm64", "m4"),
        ("Apple M5 Max", "Darwin", "arm64", "m5"),
        ("Intel Core i9", "Darwin", "x86_64", "intel"),
        ("", "Darwin", "arm64", "unknown"),
        ("", "Linux", "x86_64", "unknown"),
    ],
)
def test_classify_apple_silicon_generation(chip, system, machine, expected):
    assert classify_apple_silicon_generation(chip, system=system, machine=machine) == expected


def test_detect_apple_silicon_uses_system_profiler_when_sysctl_is_unparseable(monkeypatch):
    monkeypatch.setattr(hardware_module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(hardware_module.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(hardware_module, "_run_text", lambda *args, **kwargs: "Apple processor")
    monkeypatch.setattr(
        hardware_module,
        "_hardware_json",
        lambda: {"SPHardwareDataType": [{"chip_type": "Apple M2 Max"}]},
    )

    detected = detect_apple_silicon()

    assert detected["chip"] == "Apple M2 Max"
    assert detected["apple_silicon_generation"] == "m2"
    assert detected["is_apple_silicon"] is True


@pytest.mark.parametrize("generation", ["m1", "m2"])
def test_auto_default_uses_fp16_for_m1_m2(monkeypatch, generation):
    monkeypatch.delenv(DEFAULT_MODEL_VARIANT_ENV, raising=False)

    selection = select_default_model(
        hardware={
            "chip": f"Apple {generation.upper()} Max",
            "apple_silicon_generation": generation,
        }
    )

    assert selection.variant == "fp16"
    assert selection.precision == "FP16"
    assert selection.model == DEFAULT_FP16_HF_MODEL_ID
    assert "M1/M2" in selection.reason
    assert selection.auto_selected is True


@pytest.mark.parametrize("generation", ["m3", "m4", "m5", "unknown", "intel"])
def test_auto_default_uses_bf16_for_newer_unknown_and_intel(monkeypatch, generation):
    monkeypatch.delenv(DEFAULT_MODEL_VARIANT_ENV, raising=False)

    selection = select_default_model(
        hardware={
            "chip": "Apple M5 Max" if generation == "m5" else "",
            "apple_silicon_generation": generation,
        }
    )

    assert selection.variant == "bf16"
    assert selection.precision == "BF16"
    assert selection.model == DEFAULT_HF_MODEL_ID
    assert selection.auto_selected is True


def test_default_model_variant_env_override_forces_fp16(monkeypatch):
    monkeypatch.setenv(DEFAULT_MODEL_VARIANT_ENV, "fp16")

    selection = select_default_model(
        hardware={
            "chip": "Apple M5 Max",
            "apple_silicon_generation": "m5",
        }
    )

    assert selection.variant == "fp16"
    assert selection.model == DEFAULT_FP16_HF_MODEL_ID
    assert selection.auto_selected is False


def test_default_model_variant_env_override_forces_bf16(monkeypatch):
    monkeypatch.setenv(DEFAULT_MODEL_VARIANT_ENV, "bf16")

    selection = select_default_model(
        hardware={
            "chip": "Apple M1 Max",
            "apple_silicon_generation": "m1",
        }
    )

    assert selection.variant == "bf16"
    assert selection.model == DEFAULT_HF_MODEL_ID
    assert selection.auto_selected is False


def test_invalid_default_model_variant_env_falls_back_to_auto(monkeypatch):
    monkeypatch.setenv(DEFAULT_MODEL_VARIANT_ENV, "wat")

    selection = select_default_model(
        hardware={
            "chip": "Apple M2 Max",
            "apple_silicon_generation": "m2",
        }
    )

    assert selection.variant == "fp16"
    assert selection.model == DEFAULT_FP16_HF_MODEL_ID
    assert "ignored invalid" in selection.reason


def test_verified_default_refs_include_bf16_and_fp16():
    assert is_verified_default_model_ref(DEFAULT_HF_MODEL_ID)
    assert is_verified_default_model_ref(DEFAULT_FP16_HF_MODEL_ID)
    assert not is_verified_default_model_ref("someone/custom-model")
