"""Tests for data loaders."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from hf_volsurf.data.loaders import VolSurfaceDataLoader

DB_PATH = Path(__file__).parent.parent.parent / "data" / "db" / "hf_volsurf.db"


@pytest.fixture
def loader():
    if not DB_PATH.exists():
        pytest.skip("Database not found — run scripts/setup_database.py first")
    return VolSurfaceDataLoader(DB_PATH)


class TestVolSurfaceDataLoader:
    def test_data_summary(self, loader):
        summary = loader.get_data_summary()
        assert summary["spx_vol_surface"]["rows"] == 429_936
        assert summary["spx_spot_prices"]["rows"] == 4_034

    def test_surface_grid_shape(self, loader):
        grid = loader.get_surface_as_grid("2020-03-16")  # COVID crash
        assert grid is not None
        assert grid.shape == (8, 13)

    def test_surface_grid_values_reasonable(self, loader):
        grid = loader.get_surface_as_grid("2020-03-16")
        assert grid is not None
        # IV should be between 0 and 2 (0% to 200%)
        assert np.all(grid > 0)
        assert np.all(grid < 2.0)

    def test_all_surface_grids(self, loader):
        grids, dates = loader.get_all_surface_grids()
        assert grids.shape[0] == len(dates)
        assert grids.shape[1] == 8   # tenors
        assert grids.shape[2] == 13  # strikes
        assert grids.shape[0] > 3000  # at least 3000 trading days

    def test_spot_prices(self, loader):
        df = loader.get_spot_prices("2020-01-01", "2020-12-31")
        assert len(df) > 200  # ~252 trading days
        assert "close" in df.columns

    def test_vix_data(self, loader):
        df = loader.get_vix_data("2020-01-01", "2020-12-31")
        assert len(df) > 200
        assert "vix" in df.columns

    def test_missing_date_returns_none(self, loader):
        grid = loader.get_surface_as_grid("1900-01-01")
        assert grid is None
