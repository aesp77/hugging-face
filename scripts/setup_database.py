"""Copy SPX market data from rl_hedging_comparison into this project's database.

Uses SQLite ATTACH to copy tables directly. Idempotent via INSERT OR IGNORE.

Usage:
    poetry run python scripts/setup_database.py
"""

import sqlite3
import sys
from pathlib import Path

# Add scripts dir to path for database_schema import
sys.path.insert(0, str(Path(__file__).parent))
from database_schema import DB_PATH, create_schema

SOURCE_DB = Path(__file__).parent.parent.parent.parent / (
    "rl_hedging_comparison/data/db/rl_hedging_data.db"
)

# Tables to copy: (table_name, columns_to_select)
TABLES = {
    "spx_vol_surface": "date, strike, tenor, implied_vol, data_source",
    "spx_spot_prices": "date, open, high, low, close, volume, daily_return, data_source",
    "spx_dividend_yield": "date, dividend_yield, data_source",
    "ois_curve": "date, tenor_years, rate, currency, data_source",
    "vix_data": "date, vix, vx1, vx5, vix_term_structure, vol_regime, term_structure_regime, market_regime",
}


def copy_data() -> dict[str, int]:
    """Copy tables from source DB. Returns row counts per table."""
    if not SOURCE_DB.exists():
        print(f"ERROR: Source database not found: {SOURCE_DB}")
        sys.exit(1)

    # Create schema in target
    create_schema(DB_PATH)

    conn = sqlite3.connect(str(DB_PATH))
    conn.execute(f"ATTACH DATABASE ? AS source", (str(SOURCE_DB),))

    counts = {}
    for table, columns in TABLES.items():
        conn.execute(
            f"INSERT OR IGNORE INTO {table} ({columns}) "
            f"SELECT {columns} FROM source.{table}"
        )
        conn.commit()

        row_count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        counts[table] = row_count

        # Log the copy operation
        date_range = conn.execute(
            f"SELECT MIN(date), MAX(date) FROM {table}"
        ).fetchone()
        conn.execute(
            "INSERT INTO data_update_log (table_name, update_type, date_from, date_to, records_affected, status) "
            "VALUES (?, 'copy_from_rl_hedging', ?, ?, ?, 'success')",
            (table, date_range[0], date_range[1], row_count),
        )
        conn.commit()

    conn.execute("DETACH DATABASE source")
    conn.close()
    return counts


def main():
    print(f"Source: {SOURCE_DB}")
    print(f"Target: {DB_PATH}")
    print(f"Source exists: {SOURCE_DB.exists()}")
    print()

    counts = copy_data()

    print("=== Copy complete ===")
    for table, count in counts.items():
        print(f"  {table:25s} {count:>10,} rows")
    print(f"\n  Database: {DB_PATH}")
    print(f"  Size: {DB_PATH.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
