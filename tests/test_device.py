"""TRIBE device resolution (CPU-only PyTorch vs tribev2 auto=cuda)."""

from __future__ import annotations

from neuroclaw.model.tribe_wrapper import (
    neuralset_feature_device,
    resolve_tribe_device,
    tribev2_extractor_config_update,
)


def test_resolve_tribe_device_explicit_cpu() -> None:
    assert resolve_tribe_device("cpu") == "cpu"


def test_resolve_auto_returns_supported_string() -> None:
    out = resolve_tribe_device("auto")
    assert out in ("cuda", "mps", "cpu")


def test_neuralset_feature_device_maps_mps_to_cpu() -> None:
    """HF extractors only support cpu/cuda; published YAML uses cuda."""
    assert neuralset_feature_device("mps") == "cpu"


def test_tribev2_extractor_config_update_covers_yaml_cuda_pins() -> None:
    d = tribev2_extractor_config_update("cpu")
    assert d["data.audio_feature.device"] == "cpu"
    assert d["data.text_feature.device"] == "cpu"
    assert "data.image_feature.image.device" in d
    assert "data.video_feature.image.device" in d
