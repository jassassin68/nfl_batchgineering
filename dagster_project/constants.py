"""Shared paths and configuration constants for the Dagster project."""

from pathlib import Path

# Project root is the parent of dagster_project/
PROJECT_ROOT = Path(__file__).parent.parent

# dbt project directory
DBT_PROJECT_DIR = PROJECT_ROOT / "dbt_project"

# Trained model directory (where XGBoost model artifacts live)
MODEL_DIR = PROJECT_ROOT / "src" / "ml" / "models" / "ensemble"

# Data output directory
DATA_DIR = PROJECT_ROOT / "data"
