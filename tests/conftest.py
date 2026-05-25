"""Shared pytest fixtures and test configuration for the NFL prediction suite."""

import matplotlib

# Force a headless backend before any module imports pyplot, so plot smoke
# tests never try to open a window on CI / Windows consoles.
matplotlib.use("Agg")

import numpy as np
import polars as pl
import pytest


@pytest.fixture
def rng():
    """Deterministic random generator."""
    return np.random.RandomState(42)


@pytest.fixture
def seasons_dataframe():
    """Polars DataFrame spanning 8 seasons (2015-2022), 4 games per season."""
    rows = []
    for season in range(2015, 2023):
        for week in range(1, 5):
            rows.append(
                {"season": season, "week": week, "value": float(season + week)}
            )
    return pl.DataFrame(rows)
