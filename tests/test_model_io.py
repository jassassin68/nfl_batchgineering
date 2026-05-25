"""Model persistence round-trip tests.

A model that does not reload to identical predictions is a silent-failure
risk: training looks fine, but the deployed artifact behaves differently.
These tests fit tiny models on synthetic data, save, reload, and assert the
predictions are bit-for-bit reproducible.
"""

import numpy as np
import polars as pl

from src.ml.models.elo_model import EloModel
from src.ml.models.spread_predictor import SpreadPredictor


def test_spread_predictor_save_load_roundtrip(tmp_path, rng):
    X = rng.randn(120, 4)
    y = X[:, 0] * 3.0 - X[:, 1] * 1.5 + rng.randn(120) * 0.1
    feature_names = [f"feat_{i}" for i in range(4)]

    params = {
        "objective": "reg:squarederror",
        "max_depth": 3,
        "eta": 0.3,
        "n_estimators": 30,
        "tree_method": "hist",
        "verbosity": 0,
    }

    model = SpreadPredictor(params=params)
    model.train(X, y, feature_names=feature_names, verbose=False)
    preds_before = model.predict(X)

    model.save_model(str(tmp_path), model_name="test_model")

    reloaded = SpreadPredictor()
    reloaded.load_model(str(tmp_path / "test_model.json"))
    preds_after = reloaded.predict(X)

    assert reloaded.feature_names == feature_names
    np.testing.assert_allclose(preds_before, preds_after, rtol=1e-6)


def test_elo_model_save_load_roundtrip(tmp_path):
    games = pl.DataFrame(
        {
            "game_id": ["g1", "g2", "g3", "g4", "g5", "g6"],
            "season": [2023, 2023, 2023, 2023, 2023, 2023],
            "week": [1, 1, 2, 2, 3, 3],
            "home_team": ["AAA", "CCC", "BBB", "DDD", "AAA", "DDD"],
            "away_team": ["BBB", "DDD", "AAA", "CCC", "CCC", "BBB"],
            "home_score": [24, 17, 31, 10, 28, 21],
            "away_score": [20, 27, 14, 13, 24, 17],
        }
    )

    model = EloModel()
    model.fit(games)
    spread_before = model.predict_spread("AAA", "BBB")
    ratings_before = dict(model.ratings)

    saved_path = model.save_model(str(tmp_path), model_name="test_elo")

    reloaded = EloModel()
    reloaded.load_model(str(saved_path))
    spread_after = reloaded.predict_spread("AAA", "BBB")

    assert reloaded.ratings == ratings_before
    assert spread_before == spread_after


def test_spread_predictor_predict_before_load_raises():
    model = SpreadPredictor()
    try:
        model.predict(np.zeros((1, 4)))
    except ValueError:
        pass
    else:  # pragma: no cover - guard must fail loudly
        raise AssertionError("predict() must raise before a model is loaded")
