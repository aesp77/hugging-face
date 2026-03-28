"""Evaluation metrics for vol surface models.

Extracted from notebooks/04 and 05.
"""

import numpy as np

from hf_volsurf.utils.vol_math import (
    STRIKE_GRID,
    TENOR_ORDER,
    check_butterfly_arbitrage,
    check_calendar_arbitrage,
)


def surface_rmse(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Root mean squared error across all grid points."""
    return float(np.sqrt(np.mean((predicted - actual) ** 2)))


def surface_mae(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Mean absolute error across all grid points."""
    return float(np.mean(np.abs(predicted - actual)))


def naive_baseline_rmse(surfaces: np.ndarray) -> float:
    """RMSE of predicting today = yesterday.

    Args:
        surfaces: (n_dates, 8, 13) array of consecutive surfaces.
    """
    yesterday = surfaces[:-1]
    today = surfaces[1:]
    return surface_rmse(yesterday, today)


def arbitrage_violation_rate(
    surfaces: np.ndarray,
) -> dict[str, float]:
    """Fraction of surfaces with calendar or butterfly arbitrage violations.

    Args:
        surfaces: (n_samples, 8, 13) array.

    Returns:
        Dict with 'calendar_free_pct' and 'butterfly_free_pct'.
    """
    n = len(surfaces)
    cal_ok = sum(
        check_calendar_arbitrage(surfaces[i], TENOR_ORDER) for i in range(n)
    )
    but_ok = sum(
        check_butterfly_arbitrage(surfaces[i], STRIKE_GRID) for i in range(n)
    )
    return {
        "calendar_free_pct": 100 * cal_ok / n,
        "butterfly_free_pct": 100 * but_ok / n,
        "calendar_free": cal_ok,
        "butterfly_free": but_ok,
        "total": n,
    }
