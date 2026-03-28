"""Data loaders for SPX volatility surface data.

Reads from the local SQLite database and provides data in formats
suitable for analysis (DataFrames) and ML (numpy grids).
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from hf_volsurf.utils.vol_math import STRIKE_GRID, TENOR_ORDER

DB_PATH = Path(__file__).parent.parent.parent.parent / "data" / "db" / "hf_volsurf.db"


class VolSurfaceDataLoader:
    """Read-only loader for SPX vol surface data."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or DB_PATH
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"Database not found: {self.db_path}\n"
                "Run: poetry run python scripts/setup_database.py"
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def get_vol_surface(
        self, date: str, tenor: str | None = None
    ) -> pd.DataFrame:
        """Get vol surface for a specific date.

        Returns DataFrame with columns: date, strike, tenor, implied_vol
        """
        conn = self._connect()
        if tenor:
            df = pd.read_sql_query(
                "SELECT date, strike, tenor, implied_vol FROM spx_vol_surface "
                "WHERE date = ? AND tenor = ? ORDER BY strike",
                conn,
                params=[date, tenor],
            )
        else:
            df = pd.read_sql_query(
                "SELECT date, strike, tenor, implied_vol FROM spx_vol_surface "
                "WHERE date = ? ORDER BY tenor, strike",
                conn,
                params=[date],
            )
        conn.close()
        return df

    def get_vol_surfaces_range(
        self, start: str, end: str
    ) -> pd.DataFrame:
        """Get all vol surface data between two dates."""
        conn = self._connect()
        df = pd.read_sql_query(
            "SELECT date, strike, tenor, implied_vol FROM spx_vol_surface "
            "WHERE date BETWEEN ? AND ? ORDER BY date, tenor, strike",
            conn,
            params=[start, end],
        )
        conn.close()
        return df

    def get_surface_as_grid(self, date: str) -> np.ndarray | None:
        """Get a single day's vol surface as a 2D numpy array.

        Returns:
            Array of shape (8, 13) — (tenors, strikes) with IV values.
            None if date not found.
        """
        df = self.get_vol_surface(date)
        if df.empty:
            return None

        grid = np.full((len(TENOR_ORDER), len(STRIKE_GRID)), np.nan)
        tenor_idx = {t: i for i, t in enumerate(TENOR_ORDER)}
        strike_idx = {s: i for i, s in enumerate(STRIKE_GRID)}

        for _, row in df.iterrows():
            ti = tenor_idx.get(row["tenor"])
            si = strike_idx.get(row["strike"])
            if ti is not None and si is not None:
                grid[ti, si] = row["implied_vol"]

        if np.isnan(grid).any():
            return None
        return grid

    def get_all_surface_grids(self) -> tuple[np.ndarray, list[str]]:
        """Load all daily surfaces as a 3D array.

        Returns:
            (grids, dates) where grids has shape (n_dates, 8, 13)
            and dates is a list of date strings.
        """
        conn = self._connect()
        df = pd.read_sql_query(
            "SELECT date, strike, tenor, implied_vol FROM spx_vol_surface "
            "ORDER BY date, tenor, strike",
            conn,
        )
        conn.close()

        tenor_idx = {t: i for i, t in enumerate(TENOR_ORDER)}
        strike_idx = {s: i for i, s in enumerate(STRIKE_GRID)}

        dates = df["date"].unique().tolist()
        grids = np.full((len(dates), len(TENOR_ORDER), len(STRIKE_GRID)), np.nan)

        for date_i, date in enumerate(dates):
            day_df = df[df["date"] == date]
            for _, row in day_df.iterrows():
                ti = tenor_idx.get(row["tenor"])
                si = strike_idx.get(row["strike"])
                if ti is not None and si is not None:
                    grids[date_i, ti, si] = row["implied_vol"]

        # Filter out dates with incomplete grids
        valid_mask = ~np.isnan(grids).any(axis=(1, 2))
        grids = grids[valid_mask]
        dates = [d for d, v in zip(dates, valid_mask.tolist()) if v]

        return grids, dates

    def get_spot_prices(
        self, start: str | None = None, end: str | None = None
    ) -> pd.DataFrame:
        """Get SPX spot prices."""
        conn = self._connect()
        query = "SELECT date, open, high, low, close, volume, daily_return FROM spx_spot_prices"
        params: list = []
        if start and end:
            query += " WHERE date BETWEEN ? AND ?"
            params = [start, end]
        elif start:
            query += " WHERE date >= ?"
            params = [start]
        query += " ORDER BY date"
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df

    def get_vix_data(
        self, start: str | None = None, end: str | None = None
    ) -> pd.DataFrame:
        """Get VIX data with regime labels."""
        conn = self._connect()
        query = "SELECT * FROM vix_data"
        params: list = []
        if start and end:
            query += " WHERE date BETWEEN ? AND ?"
            params = [start, end]
        query += " ORDER BY date"
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df

    def get_data_summary(self) -> dict:
        """Get row counts and date ranges for all tables."""
        conn = self._connect()
        summary = {}
        for table in [
            "spx_vol_surface",
            "spx_spot_prices",
            "spx_dividend_yield",
            "ois_curve",
            "vix_data",
        ]:
            row = conn.execute(
                f"SELECT COUNT(*) as cnt, MIN(date) as min_d, MAX(date) as max_d FROM {table}"
            ).fetchone()
            summary[table] = {
                "rows": row["cnt"],
                "min_date": row["min_d"],
                "max_date": row["max_d"],
            }
        conn.close()
        return summary
