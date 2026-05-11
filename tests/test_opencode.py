from __future__ import annotations

import json

from mtplx.opencode import (
    build_opencode_provider_config,
    merge_opencode_config,
    opencode_model_ref,
    write_opencode_config,
)


def test_opencode_model_ref_uses_provider_namespace():
    assert (
        opencode_model_ref("mtplx-qwen36-27b-optimized-quality")
        == "mtplx/mtplx-qwen36-27b-optimized-quality"
    )


def test_build_opencode_config_uses_raw_reasoning_no_hidden_max_tokens():
    payload = build_opencode_provider_config(
        base_url="http://127.0.0.1:18083/v1",
        model_id="mtplx-qwen36-27b-optimized-quality",
        context_window=262144,
    )

    provider = payload["provider"]["mtplx"]
    model = provider["models"]["mtplx-qwen36-27b-optimized-quality"]
    assert provider["npm"] == "@ai-sdk/openai-compatible"
    assert provider["options"]["baseURL"] == "http://127.0.0.1:18083/v1"
    assert provider["options"]["timeout"] is False
    assert provider["options"]["chunkTimeout"] == 900000
    assert model["reasoning"] is True
    assert model["interleaved"] == {"field": "reasoning_content"}
    assert model["tool_call"] is True
    assert model["limit"] == {"context": 262144, "output": 262144}
    assert model["options"]["enable_thinking"] is True
    assert "maxTokens" not in json.dumps(payload)


def test_merge_opencode_config_preserves_other_providers():
    fragment = build_opencode_provider_config(
        base_url="http://127.0.0.1:18083/v1",
        model_id="mtplx-qwen36-27b-optimized-quality",
    )

    merged = merge_opencode_config(
        {
            "provider": {"lmstudio": {"name": "LM Studio"}},
            "model": "lmstudio/foo",
        },
        config_fragment=fragment,
    )

    assert merged["provider"]["lmstudio"] == {"name": "LM Studio"}
    assert merged["provider"]["mtplx"]["models"]
    assert merged["model"] == "mtplx/mtplx-qwen36-27b-optimized-quality"
    assert merged["small_model"] == "mtplx/mtplx-qwen36-27b-optimized-quality"


def test_write_opencode_config_backs_up_invalid_json(tmp_path, monkeypatch):
    path = tmp_path / "opencode.json"
    path.write_text("{bad json", encoding="utf-8")
    monkeypatch.setenv("MTPLX_OPENCODE_CONFIG", str(path))

    result = write_opencode_config(
        base_url="http://127.0.0.1:18083/v1",
        model_id="mtplx-qwen36-27b-optimized-quality",
    )

    assert result["written"] is True
    assert result["backup_path"]
    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["provider"]["mtplx"]["options"]["baseURL"] == "http://127.0.0.1:18083/v1"
