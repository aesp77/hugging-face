"""Shared configuration for the Streamlit app."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "db" / "hf_volsurf.db"
OUTPUT_DIR = PROJECT_ROOT / "output"
