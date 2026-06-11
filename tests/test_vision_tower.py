"""Vision tower, preprocessing, and spec resolution tests (no network, no real model)."""

from __future__ import annotations

import io
import json

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

from mtplx.vision import vision_spec_for_model_dir
from mtplx.vision.processing import (
    MAX_IMAGE_BYTES,
    decode_image,
    image_pad_token_count,
    preprocess_images,
    smart_resize,
)
from mtplx.vision.qwen3_vl_tower import Qwen3VLVisionConfig, Qwen3VLVisionTower

TINY_CONFIG = Qwen3VLVisionConfig(
    depth=2,
    hidden_size=32,
    intermediate_size=64,
    out_hidden_size=64,
    num_heads=2,
    patch_size=16,
    spatial_merge_size=2,
    temporal_patch_size=2,
    in_channels=3,
    num_position_embeddings=16,
    deepstack_visual_indexes=[0, 1],
)

TINY_PREPROCESSOR_CONFIG = {
    "patch_size": 16,
    "temporal_patch_size": 2,
    "merge_size": 2,
    "min_pixels": 32 * 32,
    "max_pixels": 16777216,
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
}


def _random_image(width: int, height: int) -> Image.Image:
    rng = np.random.default_rng(0)
    return Image.fromarray(
        rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)
    )


def test_tower_forward_shapes():
    pixel_values, grid_thw = preprocess_images(
        [_random_image(96, 64)], TINY_PREPROCESSOR_CONFIG
    )
    assert grid_thw == [(1, 4, 6)]
    assert pixel_values.shape == (24, 3 * 2 * 16 * 16)

    tower = Qwen3VLVisionTower(TINY_CONFIG)
    embeddings, deepstack = tower(pixel_values, grid_thw)
    mx.eval(embeddings)

    # (64 / 32) * (96 / 32) = 6 tokens after the 2x2 spatial merge.
    assert embeddings.shape == (6, 64)
    assert np.isfinite(np.array(embeddings, copy=False)).all()

    assert [layer for layer, _ in deepstack] == [0, 1]
    for _, features in deepstack:
        mx.eval(features)
        assert features.shape == (6, 64)


def test_tower_forward_multiple_images():
    pixel_values, grid_thw = preprocess_images(
        [_random_image(64, 64), _random_image(96, 64)], TINY_PREPROCESSOR_CONFIG
    )
    assert grid_thw == [(1, 4, 4), (1, 4, 6)]

    tower = Qwen3VLVisionTower(TINY_CONFIG)
    embeddings, deepstack = tower(pixel_values, grid_thw)
    mx.eval(embeddings)

    assert embeddings.shape == (4 + 6, 64)
    assert len(deepstack) == 2


def test_smart_resize_factor_rounding():
    assert smart_resize(100, 200, factor=32, min_pixels=1024, max_pixels=16777216) == (
        96,
        192,
    )


def test_smart_resize_min_pixel_clamp():
    # 32x32 = 1024 < 4096, beta = 2, both sides scale to 64.
    assert smart_resize(32, 32, factor=32, min_pixels=4096, max_pixels=16777216) == (
        64,
        64,
    )


def test_smart_resize_min_clamp_recovers_zero_rounding():
    # round(8 / 32) * 32 = 0; the min-pixel branch rescales to (32, 192).
    assert smart_resize(8, 64, factor=32, min_pixels=4096, max_pixels=16777216) == (
        32,
        192,
    )


def test_smart_resize_max_pixel_clamp():
    # beta = 1000 / 256, floor(1000 / beta / 32) * 32 = 256 on both sides.
    assert smart_resize(1000, 1000, factor=32, min_pixels=1024, max_pixels=65536) == (
        256,
        256,
    )


def test_smart_resize_rejects_extreme_aspect_ratio():
    with pytest.raises(ValueError):
        smart_resize(8050, 40, factor=32, min_pixels=1024, max_pixels=16777216)


def test_image_pad_token_count():
    assert image_pad_token_count((1, 4, 6)) == 6
    assert image_pad_token_count((2, 8, 10), merge_size=2) == 40


def test_decode_image_rejects_oversized_payload():
    with pytest.raises(ValueError):
        decode_image(b"\0" * (MAX_IMAGE_BYTES + 1))


def test_decode_image_rejects_oversized_dimensions():
    buffer = io.BytesIO()
    Image.new("RGB", (8001, 8)).save(buffer, format="PNG")
    with pytest.raises(ValueError):
        decode_image(buffer.getvalue())


def test_decode_image_rejects_garbage_bytes():
    with pytest.raises(ValueError):
        decode_image(b"definitely not an image")


def test_decode_image_roundtrip():
    buffer = io.BytesIO()
    Image.new("RGB", (40, 30), color=(200, 10, 10)).save(buffer, format="PNG")
    image = decode_image(buffer.getvalue())
    assert image.mode == "RGB"
    assert image.size == (40, 30)


def test_vision_spec_none_without_vision_config(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    assert vision_spec_for_model_dir(tmp_path) is None


def test_vision_spec_none_without_vision_tower_weights(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5", "vision_config": {"patch_size": 16}})
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.embed_tokens.weight": "model.safetensors"}})
    )
    assert vision_spec_for_model_dir(tmp_path) is None


def test_vision_spec_none_without_index(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5", "vision_config": {"patch_size": 16}})
    )
    assert vision_spec_for_model_dir(tmp_path) is None


def test_vision_spec_populated(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5",
                "image_token_id": 248056,
                "video_token_id": 248057,
                "vision_start_token_id": 248053,
                "vision_end_token_id": 248054,
                "vision_config": {
                    "patch_size": 16,
                    "spatial_merge_size": 2,
                    "temporal_patch_size": 2,
                    "out_hidden_size": 5120,
                },
            }
        )
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {"weight_map": {"vision_tower.pos_embed.weight": "model.safetensors"}}
        )
    )
    spec = vision_spec_for_model_dir(tmp_path)
    assert spec is not None
    assert spec.image_token_id == 248056
    assert spec.video_token_id == 248057
    assert spec.vision_start_token_id == 248053
    assert spec.vision_end_token_id == 248054
    assert spec.spatial_merge_size == 2
    assert spec.patch_size == 16
    assert spec.temporal_patch_size == 2
    assert spec.out_hidden_size == 5120
