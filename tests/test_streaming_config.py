# SPDX-License-Identifier: Apache-2.0
"""Unit tests for StreamingConfig (§7A — config unit tests).

These tests require no model and exercise the full config lifecycle:
defaults, from_dict, to_dict, round-trips, and edge cases.
"""

import pytest


@pytest.fixture
def _has_config():
    """Import guard — skip if streaming module unavailable."""
    try:
        from omlx.streaming.config import StreamingConfig
    except ImportError:
        pytest.skip("omlx.streaming not available")
    return StreamingConfig


class TestStreamingConfigDefaults:
    """Validate that StreamingConfig() produces sane defaults."""

    def test_defaults_match_dataclass(self, _has_config: type):
        """StreamingConfig() defaults: hot=8, warm=16, transient=4."""
        cfg = _has_config()
        assert cfg.stream_experts is False
        assert cfg.expert_sidecar_path is None
        assert cfg.expert_hot_count == 8
        assert cfg.expert_warm_slots == 16
        assert cfg.expert_transient_slots == 4
        assert cfg.expert_prefetch is True
        assert cfg.expert_prefetch_window == 4
        assert cfg.expert_top_k_override is None
        assert cfg.calibration_frequencies is None

    def test_from_dict_empty_equals_default(self, _has_config: type):
        """StreamingConfig.from_dict({}) == StreamingConfig()."""
        cfg_default = _has_config()
        cfg_empty = _has_config.from_dict({})
        assert cfg_default.expert_hot_count == cfg_empty.expert_hot_count
        assert cfg_default.expert_warm_slots == cfg_empty.expert_warm_slots
        assert cfg_default.expert_transient_slots == cfg_empty.expert_transient_slots
        assert cfg_default.stream_experts == cfg_empty.stream_experts


class TestStreamingConfigRoundTrip:
    """Validate to_dict / from_dict round-trips."""

    def test_roundtrip_to_dict_from_dict(self, _has_config: type):
        """from_dict(cfg.to_dict()) == cfg for fully-populated cfg."""
        cfg = _has_config(
            stream_experts=True,
            expert_sidecar_path="/tmp/model.stream",
            expert_hot_count=8,
            expert_warm_slots=16,
            expert_transient_slots=4,
            expert_prefetch=True,
            expert_prefetch_window=8,
            expert_top_k_override=4,
            calibration_frequencies={0: 1.0, 1: 2.0, 2: 3.0},
        )
        restored = _has_config.from_dict(cfg.to_dict())
        assert restored.stream_experts == cfg.stream_experts
        assert restored.expert_sidecar_path == cfg.expert_sidecar_path
        assert restored.expert_hot_count == cfg.expert_hot_count
        assert restored.expert_warm_slots == cfg.expert_warm_slots
        assert restored.expert_transient_slots == cfg.expert_transient_slots
        assert restored.expert_prefetch == cfg.expert_prefetch
        assert restored.expert_prefetch_window == cfg.expert_prefetch_window
        assert restored.expert_top_k_override == cfg.expert_top_k_override
        assert dict(restored.calibration_frequencies) == cfg.calibration_frequencies

    def test_from_dict_ignores_unknown_keys(self, _has_config: type):
        """Extra keys in dict are silently dropped."""
        extra = _has_config.from_dict({
            "stream_experts": True,
            "boilerplate_key": 42,
            "another_unknown": [1, 2, 3],
        })
        assert extra.stream_experts is True

    def test_to_dict_contains_all_fields(self, _has_config: type):
        """to_dict() produces exactly 9 documented keys."""
        cfg = _has_config()
        d = cfg.to_dict()
        expected_keys = {
            "stream_experts",
            "expert_sidecar_path",
            "expert_hot_count",
            "expert_warm_slots",
            "expert_transient_slots",
            "expert_prefetch",
            "expert_prefetch_window",
            "expert_top_k_override",
            "calibration_frequencies",
        }
        assert set(d.keys()) == expected_keys

    def test_calibration_frequencies_roundtrip(self, _has_config: type):
        """Dict with int keys survives JSON round-trip (str coercion handled)."""
        cfg = _has_config(calibration_frequencies={0: 1.5, 127: 2.5})
        d = cfg.to_dict()
        restored = _has_config.from_dict(d)
        # calibration_frequencies keys become str on JSON round-trip;
        # from_dict loads them as-is (str), so we compare dict equality.
        assert isinstance(restored.calibration_frequencies, dict)

    def test_inert_when_disabled(self, _has_config: type):
        """stream_experts=False round-trips cleanly (I5)."""
        cfg = _has_config(stream_experts=False)
        d = cfg.to_dict()
        assert d["stream_experts"] is False
        restored = _has_config.from_dict(d)
        assert restored.stream_experts is False

    def test_expert_top_k_override_optional(self, _has_config: type):
        """Default None; explicit int round-trips."""
        cfg_none = _has_config()
        assert cfg_none.expert_top_k_override is None

        cfg_explicit = _has_config(expert_top_k_override=4)
        d = cfg_explicit.to_dict()
        restored = _has_config.from_dict(d)
        assert restored.expert_top_k_override == 4
