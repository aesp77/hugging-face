"""Tests for vol_math utilities."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from hf_volsurf.utils.vol_math import (
    TENOR_ORDER,
    bs_implied_vol,
    bs_price,
    check_butterfly_arbitrage,
    check_calendar_arbitrage,
    normalize_surface,
    denormalize_surface,
    tenor_to_years,
)


class TestTenorConversion:
    def test_all_tenors_convert(self):
        for t in TENOR_ORDER:
            assert tenor_to_years(t) > 0

    def test_known_values(self):
        assert tenor_to_years("1m") == pytest.approx(1 / 12)
        assert tenor_to_years("1y") == 1.0
        assert tenor_to_years("2y") == 2.0

    def test_unknown_tenor_raises(self):
        with pytest.raises(ValueError):
            tenor_to_years("5y")


class TestBlackScholes:
    def test_atm_call(self):
        price = bs_price(100, 100, 1.0, 0.05, 0.2, "C")
        assert 5 < price < 15  # ATM call ~10 for these params

    def test_put_call_parity(self):
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
        call = bs_price(S, K, T, r, sigma, "C")
        put = bs_price(S, K, T, r, sigma, "P")
        parity = call - put - (S - K * np.exp(-r * T))
        assert abs(parity) < 1e-10

    def test_implied_vol_roundtrip(self):
        S, K, T, r, sigma = 100, 105, 0.5, 0.03, 0.25
        price = bs_price(S, K, T, r, sigma, "C")
        recovered = bs_implied_vol(price, S, K, T, r, "C")
        assert recovered == pytest.approx(sigma, abs=1e-6)


class TestArbitrageChecks:
    def test_valid_surface_no_calendar_arb(self):
        # IVs increasing slightly with tenor (typical)
        tenors = TENOR_ORDER
        grid = np.array(
            [[0.20 + 0.01 * i] * 13 for i in range(8)]
        )
        assert check_calendar_arbitrage(grid, tenors) is True

    def test_calendar_arb_detected(self):
        tenors = TENOR_ORDER
        grid = np.array(
            [[0.30 - 0.03 * i] * 13 for i in range(8)]
        )
        # Decreasing vol with increasing tenor -> calendar arb
        assert check_calendar_arbitrage(grid, tenors) is False

    def test_valid_surface_no_butterfly_arb(self):
        strikes = [0.7 + 0.05 * i for i in range(13)]
        # Convex smile
        grid = np.array(
            [[0.3 + 0.01 * (k - 1.0) ** 2 for k in strikes]] * 8
        )
        assert check_butterfly_arbitrage(grid, strikes) is True


class TestNormalization:
    def test_roundtrip(self):
        grids = np.random.rand(100, 8, 13) * 0.3 + 0.1
        normed, stats = normalize_surface(grids)
        recovered = denormalize_surface(normed, stats)
        np.testing.assert_allclose(recovered, grids, atol=1e-10)

    def test_zero_mean_unit_std(self):
        grids = np.random.rand(100, 8, 13) * 0.3 + 0.1
        normed, _ = normalize_surface(grids)
        np.testing.assert_allclose(normed.mean(axis=0), 0, atol=1e-10)
        np.testing.assert_allclose(normed.std(axis=0), 1, atol=1e-2)
