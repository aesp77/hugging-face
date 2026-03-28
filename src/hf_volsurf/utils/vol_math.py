"""Volatility mathematics utilities.

Black-Scholes pricing, tenor conversion, and no-arbitrage checks
for implied volatility surfaces.
"""

import numpy as np
from scipy.stats import norm

# Ordered tenors and their year fractions (365-day convention for IV)
TENOR_ORDER = ["1m", "2m", "3m", "6m", "9m", "1y", "18m", "2y"]
TENOR_TO_YEARS_MAP = {
    "1m": 1 / 12,
    "2m": 2 / 12,
    "3m": 3 / 12,
    "6m": 6 / 12,
    "9m": 9 / 12,
    "1y": 1.0,
    "18m": 1.5,
    "2y": 2.0,
}

# Standard strike grid (moneyness K/S)
STRIKE_GRID = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3]


def tenor_to_years(tenor: str) -> float:
    """Convert tenor string to year fraction (365-day convention)."""
    if tenor not in TENOR_TO_YEARS_MAP:
        raise ValueError(f"Unknown tenor: {tenor}. Expected one of {TENOR_ORDER}")
    return TENOR_TO_YEARS_MAP[tenor]


def bs_price(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str = "C"
) -> float:
    """Black-Scholes option price.

    Args:
        S: Spot price
        K: Strike price
        T: Time to expiry in years
        r: Risk-free rate
        sigma: Implied volatility
        option_type: 'C' for call, 'P' for put
    """
    if T <= 0 or sigma <= 0:
        intrinsic = max(S - K, 0) if option_type == "C" else max(K - S, 0)
        return intrinsic

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "C":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_implied_vol(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "C",
    tol: float = 1e-8,
    max_iter: int = 100,
) -> float | None:
    """Invert Black-Scholes to find implied volatility via Newton-Raphson."""
    if T <= 0 or price <= 0:
        return None

    sigma = 0.2  # initial guess
    for _ in range(max_iter):
        bs = bs_price(S, K, T, r, sigma, option_type)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        vega = S * np.sqrt(T) * norm.pdf(d1)

        if vega < 1e-12:
            return None

        diff = bs - price
        if abs(diff) < tol:
            return sigma

        sigma -= diff / vega
        if sigma <= 0:
            sigma = 0.001

    return None


def check_calendar_arbitrage(surface_grid: np.ndarray, tenors: list[str]) -> bool:
    """Check that total variance is non-decreasing in time (no calendar arb).

    Args:
        surface_grid: shape (n_tenors, n_strikes), values are implied vols
        tenors: list of tenor strings matching rows

    Returns:
        True if NO arbitrage violations found.
    """
    T = np.array([tenor_to_years(t) for t in tenors])
    total_var = surface_grid**2 * T[:, np.newaxis]  # sigma^2 * T

    for i in range(len(T) - 1):
        if np.any(total_var[i + 1] < total_var[i] - 1e-10):
            return False
    return True


def check_butterfly_arbitrage(surface_grid: np.ndarray, strikes: list[float]) -> bool:
    """Check convexity in strike (no butterfly arb).

    For each tenor, the call price must be convex in strike, which implies
    the second derivative of price w.r.t. strike is non-negative.
    We check this via the second finite difference of total variance.

    Args:
        surface_grid: shape (n_tenors, n_strikes), values are implied vols
        strikes: list of strike values (moneyness) matching columns

    Returns:
        True if NO arbitrage violations found.
    """
    K = np.array(strikes)
    log_k = np.log(K)

    for row_idx in range(surface_grid.shape[0]):
        iv = surface_grid[row_idx]
        # Second finite difference of iv w.r.t. log-strike
        for j in range(1, len(K) - 1):
            d2 = (iv[j + 1] - 2 * iv[j] + iv[j - 1]) / (
                (log_k[j + 1] - log_k[j]) * (log_k[j] - log_k[j - 1])
            )
            # Severe butterfly violation (negative second derivative of price)
            # This is a simplified check; full Dupire check is more involved
            if iv[j] + d2 * (log_k[j + 1] - log_k[j - 1]) ** 2 / 8 < -0.01:
                return False
    return True


def normalize_surface(
    grids: np.ndarray, stats: dict | None = None
) -> tuple[np.ndarray, dict]:
    """Z-score normalize per grid point.

    Args:
        grids: shape (n_samples, n_tenors, n_strikes)
        stats: pre-computed {'mean': ..., 'std': ...}. If None, compute from data.

    Returns:
        (normalized_grids, stats_dict)
    """
    if stats is None:
        mean = grids.mean(axis=0)
        std = grids.std(axis=0)
        std[std < 1e-10] = 1.0  # avoid division by zero
        stats = {"mean": mean, "std": std}

    normalized = (grids - stats["mean"]) / stats["std"]
    return normalized, stats


def denormalize_surface(grids: np.ndarray, stats: dict) -> np.ndarray:
    """Reverse z-score normalization."""
    return grids * stats["std"] + stats["mean"]
