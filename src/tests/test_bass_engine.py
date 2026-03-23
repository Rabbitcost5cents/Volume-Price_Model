"""
Unit tests for bass_engine.py
覆盖场景：bass_f 边界条件、参数缓存完整性（L3 已修复）、
         缓存校验拒绝越界参数、calculate_theoretical_sales 非负性。
"""
import os
import sys
import json
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bass_engine import bass_f, bass_S, BassEngine


# ---------------------------------------------------------------------------
# bass_f / bass_S
# ---------------------------------------------------------------------------

class TestBassF:
    def test_returns_zero_when_p_is_zero(self):
        """p=0 不应导致除零错误。"""
        result = bass_f(np.array([1.0, 2.0, 5.0]), p=0, q=0.4)
        assert np.all(result == 0)

    def test_returns_nonnegative_for_valid_params(self):
        t = np.arange(1, 30)
        result = bass_f(t, p=0.03, q=0.38)
        assert np.all(result >= 0)

    def test_curve_peaks_then_declines(self):
        """Bass 曲线应先升后降（单峰）。"""
        t = np.arange(1, 60)
        s = bass_f(t, p=0.03, q=0.38)
        peak_idx = int(np.argmax(s))
        assert 0 < peak_idx < len(t) - 1, "Peak should not be at boundary"
        assert s[peak_idx] > s[peak_idx + 5], "Sales should decline after peak"

    def test_bass_s_scales_by_m(self):
        """bass_S(t, p, q, m) 应等于 m * bass_f(t, p, q)。"""
        t = np.array([5.0])
        p, q, m = 0.03, 0.38, 100000
        assert abs(bass_S(t, p, q, m)[0] - m * bass_f(t, p, q)[0]) < 1e-9


# ---------------------------------------------------------------------------
# BassEngine
# ---------------------------------------------------------------------------

class TestBassEngineDefaults:
    def test_get_bass_params_unknown_series_returns_avg(self):
        engine = BassEngine()
        p, q, m = engine.get_bass_params("UNKNOWN_XYZ")
        assert p == engine.avg_p
        assert q == engine.avg_q

    def test_get_bass_params_known_series(self):
        engine = BassEngine()
        engine.params = {"V30": (0.02, 0.35, 50000.0)}
        p, q, m = engine.get_bass_params("V30")
        assert p == 0.02
        assert q == 0.35
        assert m == 50000.0

    def test_calculate_theoretical_sales_nonnegative(self):
        engine = BassEngine()
        engine.params = {"V30": (0.03, 0.38, 100000.0)}
        for month in [0, 1, 6, 12, 24]:
            val = engine.calculate_theoretical_sales("V30", month)
            assert val >= 0, f"Negative sales at month {month}"


# ---------------------------------------------------------------------------
# Bass 缓存完整性（L3 已修复验证）
# ---------------------------------------------------------------------------

class TestBassEngineCache:
    def test_save_and_load_roundtrip(self, tmp_path):
        """参数应能正确序列化再反序列化。"""
        engine = BassEngine()
        engine.params = {"V30": (0.02, 0.35, 50000.0), "V40": (0.03, 0.40, 80000.0)}
        engine.avg_p = 0.025
        engine.avg_q = 0.375

        cache_path = str(tmp_path / "bass_params.json")
        engine.save_to_cache(cache_path)

        engine2 = BassEngine()
        ok = engine2.load_from_cache(cache_path)

        assert ok is True
        assert "V30" in engine2.params
        assert "V40" in engine2.params
        p, q, m = engine2.params["V30"]
        assert abs(p - 0.02) < 1e-9
        assert abs(engine2.avg_p - 0.025) < 1e-9

    def test_load_rejects_p_out_of_range(self, tmp_path):
        """p > 1.0 时缓存应被拒绝，返回 False。"""
        bad = {
            "params": {"V30": [1.5, 0.35, 50000.0]},
            "avg_p": 0.03,
            "avg_q": 0.38,
        }
        cache_path = str(tmp_path / "bad_p.json")
        with open(cache_path, "w") as f:
            json.dump(bad, f)

        engine = BassEngine()
        result = engine.load_from_cache(cache_path)
        assert result is False

    def test_load_rejects_negative_m(self, tmp_path):
        """m <= 0 时缓存应被拒绝，返回 False。"""
        bad = {
            "params": {"V30": [0.03, 0.38, -1000.0]},
            "avg_p": 0.03,
            "avg_q": 0.38,
        }
        cache_path = str(tmp_path / "bad_m.json")
        with open(cache_path, "w") as f:
            json.dump(bad, f)

        engine = BassEngine()
        result = engine.load_from_cache(cache_path)
        assert result is False

    def test_load_rejects_invalid_avg_params(self, tmp_path):
        """avg_p 或 avg_q 越界时缓存应被拒绝。"""
        bad = {
            "params": {},
            "avg_p": 0.0,   # 0 不在 (0, 1] 范围内
            "avg_q": 0.38,
        }
        cache_path = str(tmp_path / "bad_avg.json")
        with open(cache_path, "w") as f:
            json.dump(bad, f)

        engine = BassEngine()
        result = engine.load_from_cache(cache_path)
        assert result is False

    def test_atomic_write_uses_tmp_file(self, tmp_path):
        """save_to_cache 应先写 .tmp 再原子替换，最终不留下 .tmp 文件。"""
        engine = BassEngine()
        engine.params = {"V30": (0.03, 0.38, 100000.0)}
        engine.avg_p = 0.03
        engine.avg_q = 0.38

        cache_path = str(tmp_path / "bass_params.json")
        engine.save_to_cache(cache_path)

        assert os.path.exists(cache_path)
        assert not os.path.exists(cache_path + ".tmp"), ".tmp file should be cleaned up"
