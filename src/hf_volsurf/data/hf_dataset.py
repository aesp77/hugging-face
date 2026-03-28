"""HuggingFace Dataset wrappers for vol surface data.

Converts SQLite data into HF datasets.Dataset objects for use
with Transformers Trainer API and other HF tools.
"""

from pathlib import Path

import pandas as pd
from datasets import Dataset

from hf_volsurf.data.loaders import VolSurfaceDataLoader
from hf_volsurf.utils.vol_math import TENOR_ORDER, tenor_to_years


def create_pointwise_dataset(
    db_path: Path | None = None,
    start: str | None = None,
    end: str | None = None,
) -> Dataset:
    """Create a pointwise regression dataset.

    Each row is one (date, moneyness, tenor_years, implied_vol) observation.
    ~429k rows total if no date filter.

    Optionally enriched with spot price and VIX if available.
    """
    loader = VolSurfaceDataLoader(db_path)

    # Load vol surface
    conn_str = str(loader.db_path)
    import sqlite3

    conn = sqlite3.connect(conn_str)
    query = "SELECT date, strike, tenor, implied_vol FROM spx_vol_surface"
    params: list = []
    if start and end:
        query += " WHERE date BETWEEN ? AND ?"
        params = [start, end]
    query += " ORDER BY date"
    vol_df = pd.read_sql_query(query, conn, params=params)

    # Load spot prices for enrichment
    spot_df = pd.read_sql_query(
        "SELECT date, close as spot FROM spx_spot_prices", conn
    )
    spot_map = dict(zip(spot_df["date"], spot_df["spot"]))

    # Load VIX for enrichment
    vix_df = pd.read_sql_query("SELECT date, vix FROM vix_data", conn)
    vix_map = dict(zip(vix_df["date"], vix_df["vix"]))
    conn.close()

    # Build dataset dict
    vol_df["tenor_years"] = vol_df["tenor"].map(
        {t: tenor_to_years(t) for t in TENOR_ORDER}
    )
    vol_df["spot"] = vol_df["date"].map(spot_map)
    vol_df["vix"] = vol_df["date"].map(vix_map)

    # Rename strike to moneyness for clarity
    vol_df = vol_df.rename(columns={"strike": "moneyness"})

    return Dataset.from_pandas(
        vol_df[["date", "moneyness", "tenor_years", "implied_vol", "spot", "vix"]],
        preserve_index=False,
    )


def create_image_dataset(
    db_path: Path | None = None,
) -> Dataset:
    """Create a dataset of vol surface 'images' for diffusion models.

    Each row is one trading day's surface as a (8, 13) grid.
    ~4,000 samples total.
    """
    loader = VolSurfaceDataLoader(db_path)
    grids, dates = loader.get_all_surface_grids()

    # Load spot + VIX for conditioning
    spot_df = loader.get_spot_prices()
    spot_map = dict(zip(spot_df["date"], spot_df["close"]))

    vix_df = loader.get_vix_data()
    vix_map = dict(zip(vix_df["date"], vix_df["vix"]))
    regime_map = dict(zip(vix_df["date"], vix_df["market_regime"]))

    data = {
        "date": dates,
        "surface": [grids[i].tolist() for i in range(len(dates))],
        "spot": [spot_map.get(d) for d in dates],
        "vix": [vix_map.get(d) for d in dates],
        "regime": [regime_map.get(d) for d in dates],
    }

    return Dataset.from_dict(data)
