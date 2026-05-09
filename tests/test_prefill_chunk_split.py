"""Tests for the dense / repage prefill-chunk-size split.

PR #33 originally bumped a single `MTPLX_PREFILL_CHUNK_SIZE` knob from 2048 to
4096. Maintainer review on 2026-05-09 flagged that 4096 in the repage path
(contexts > 64k) regresses 128k TTFT, so the knob is split into:

    MTPLX_PREFILL_CHUNK_SIZE_DENSE   (dense path, contexts <= 64k)  default 4096
    MTPLX_PREFILL_CHUNK_SIZE_REPAGE  (repage path, contexts >  64k) default 2048

The dense cutoff is sourced from `MTPLX_SUSTAINED_DENSE_DECODE_MAX_CONTEXT`
and is locked at 65536 in the sustained profile config.

The legacy single-knob `MTPLX_PREFILL_CHUNK_SIZE` env stays as a back-compat
fallback: when set to a numeric value it overrides BOTH paths.
"""

from __future__ import annotations

import pytest

from mtplx.generation import (
    _prefill_chunk_size,
    _sustained_prefill_layout,
)
from mtplx.profiles import SUSTAINED_PREFILL_ENV


# ---------------------------------------------------------------------------
# Profile-level invariants


def test_sustained_profile_locks_dense_cutoff_at_64k() -> None:
    """The sustained profile must pin the dense cutoff at 64k so contexts
    above 64k take the repage path with the smaller chunk."""

    assert SUSTAINED_PREFILL_ENV["MTPLX_SUSTAINED_DENSE_DECODE_MAX_CONTEXT"] == "65536"


def test_sustained_profile_ships_split_chunk_defaults() -> None:
    assert SUSTAINED_PREFILL_ENV["MTPLX_PREFILL_CHUNK_SIZE"] == "auto"
    assert SUSTAINED_PREFILL_ENV["MTPLX_PREFILL_CHUNK_SIZE_DENSE"] == "4096"
    assert SUSTAINED_PREFILL_ENV["MTPLX_PREFILL_CHUNK_SIZE_REPAGE"] == "2048"


# ---------------------------------------------------------------------------
# Path selection at the 64k boundary


def _apply_split_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mirror the sustained-profile env relevant to chunk-size selection."""

    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL_LAYOUT", "auto")
    monkeypatch.setenv("MTPLX_SUSTAINED_DENSE_DECODE_MAX_CONTEXT", "65536")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "auto")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE_DENSE", "4096")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE_REPAGE", "2048")


@pytest.mark.parametrize("context_tokens", [32_768, 65_536])
def test_prefill_chunk_dense_uses_4096_at_le_64k(
    monkeypatch: pytest.MonkeyPatch, context_tokens: int
) -> None:
    """At <= 64k the dense layout is selected and the dense chunk size wins."""

    _apply_split_env(monkeypatch)
    monkeypatch.setenv("MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS", str(context_tokens))

    assert _sustained_prefill_layout() == "contiguous_dense_decode"
    assert _prefill_chunk_size() == 4096


@pytest.mark.parametrize("context_tokens", [80_000, 131_072])
def test_prefill_chunk_repage_uses_2048_at_gt_64k(
    monkeypatch: pytest.MonkeyPatch, context_tokens: int
) -> None:
    """Above 64k the repage layout is selected and the smaller chunk wins."""

    _apply_split_env(monkeypatch)
    monkeypatch.setenv("MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS", str(context_tokens))

    assert _sustained_prefill_layout() == "contiguous_then_repage"
    assert _prefill_chunk_size() == 2048


# ---------------------------------------------------------------------------
# Env-var honoring


def test_prefill_chunk_envs_honored(monkeypatch: pytest.MonkeyPatch) -> None:
    """The split envs must be respected end-to-end on each path."""

    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL_LAYOUT", "auto")
    monkeypatch.setenv("MTPLX_SUSTAINED_DENSE_DECODE_MAX_CONTEXT", "65536")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "auto")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE_DENSE", "1024")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE_REPAGE", "512")

    monkeypatch.setenv("MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS", "32768")
    assert _sustained_prefill_layout() == "contiguous_dense_decode"
    assert _prefill_chunk_size() == 1024

    monkeypatch.setenv("MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS", "131072")
    assert _sustained_prefill_layout() == "contiguous_then_repage"
    assert _prefill_chunk_size() == 512


def test_prefill_chunk_legacy_env_back_compat(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setting the legacy single-knob env to a numeric value must override
    BOTH the dense and repage paths so existing deployments keep working."""

    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL", "1")
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL_LAYOUT", "auto")
    monkeypatch.setenv("MTPLX_SUSTAINED_DENSE_DECODE_MAX_CONTEXT", "65536")
    # Legacy single-knob set to a non-default value; split envs are deliberately
    # left at fresh defaults so we can confirm the legacy knob actually wins.
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "1536")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE_DENSE", "4096")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE_REPAGE", "2048")

    monkeypatch.setenv("MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS", "32768")
    assert _prefill_chunk_size() == 1536

    monkeypatch.setenv("MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS", "131072")
    assert _prefill_chunk_size() == 1536
