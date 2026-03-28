"""Database schema for HF VolSurf project.

Reuses the rl_hedging_comparison schema (same table names) to enable
direct data copying without schema translation.
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "db" / "hf_volsurf.db"

SCHEMA_SQL = """
-- SPX spot prices (daily)
CREATE TABLE IF NOT EXISTS spx_spot_prices (
    date DATE PRIMARY KEY,
    open REAL,
    high REAL,
    low REAL,
    close REAL NOT NULL,
    volume BIGINT,
    daily_return REAL,
    data_source TEXT DEFAULT 'PSC',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_spot_date ON spx_spot_prices(date);

-- Volatility surface
CREATE TABLE IF NOT EXISTS spx_vol_surface (
    date DATE NOT NULL,
    strike REAL NOT NULL,
    tenor TEXT NOT NULL,
    implied_vol REAL NOT NULL,
    data_source TEXT DEFAULT 'MARQUEE',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date, strike, tenor)
);

CREATE INDEX IF NOT EXISTS idx_vol_surface_date ON spx_vol_surface(date);
CREATE INDEX IF NOT EXISTS idx_vol_surface_date_strike_tenor
    ON spx_vol_surface(date, strike, tenor);

-- Dividend yield
CREATE TABLE IF NOT EXISTS spx_dividend_yield (
    date DATE PRIMARY KEY,
    dividend_yield REAL NOT NULL,
    data_source TEXT DEFAULT 'BLOOMBERG',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- OIS curve (risk-free rates)
CREATE TABLE IF NOT EXISTS ois_curve (
    date DATE NOT NULL,
    tenor_years REAL NOT NULL,
    rate REAL NOT NULL,
    currency TEXT DEFAULT 'USD',
    data_source TEXT DEFAULT 'BLOOMBERG',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date, tenor_years)
);

CREATE INDEX IF NOT EXISTS idx_ois_date ON ois_curve(date);

-- VIX data with regime labels
CREATE TABLE IF NOT EXISTS vix_data (
    date DATE PRIMARY KEY,
    vix REAL,
    vx1 REAL,
    vx5 REAL,
    vix_term_structure REAL,
    vol_regime TEXT,
    term_structure_regime TEXT,
    market_regime TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Data update log
CREATE TABLE IF NOT EXISTS data_update_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    table_name TEXT NOT NULL,
    update_type TEXT NOT NULL,
    date_from DATE,
    date_to DATE,
    records_affected INTEGER,
    status TEXT,
    error_message TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def create_schema(db_path: Path | None = None) -> None:
    """Create all tables and indexes."""
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.executescript(SCHEMA_SQL)
    conn.close()
