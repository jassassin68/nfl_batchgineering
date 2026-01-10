"""
Elo Rating System for NFL game predictions.

Based on FiveThirtyEight's NFL Elo methodology:
- K-factor: 20 (standard for NFL)
- Home advantage: 48 Elo points (~2.5 point spread)
- Margin of victory multiplier: log(abs(point_diff) + 1)
- Season regression: 1/3 toward mean (1505)

References:
- Silver, N. (2014). FiveThirtyEight NFL Elo Ratings
- Glickman & Stern (1998). JASA Dynamic Rating System
"""

from typing import Dict, List, Tuple, Optional
import polars as pl
import numpy as np
from pathlib import Path
import json


class EloModel:
    """
    Time-decayed Elo rating system for NFL teams.

    Attributes:
        k_factor: Learning rate (20 for NFL standard)
        home_advantage: Elo points for home team (~48 = 2.5 pts spread)
        initial_rating: Starting Elo for new teams (1500)
        regression_factor: Fraction to regress toward mean between seasons (0.33)
        mean_rating: Target for regression (1505 for NFL)
    """

    def __init__(
        self,
        k_factor: float = 20.0,
        home_advantage: float = 48.0,
        initial_rating: float = 1500.0,
        regression_factor: float = 0.33,
        mean_rating: float = 1505.0
    ):
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.initial_rating = initial_rating
        self.regression_factor = regression_factor
        self.mean_rating = mean_rating

        # Track ratings over time
        self.ratings: Dict[str, float] = {}  # team -> current rating
        self.rating_history: List[Dict] = []  # Historical ratings by game

    def get_rating(self, team: str) -> float:
        """Get current Elo rating for team."""
        return self.ratings.get(team, self.initial_rating)

    def _expected_score(self, rating_a: float, rating_b: float) -> float:
        """
        Calculate expected score for team A vs team B.
        Returns probability between 0 and 1.
        """
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def _margin_of_victory_multiplier(self, point_diff: int, elo_diff: float) -> float:
        """
        Adjust K-factor based on margin of victory.
        Blowouts should matter less than close games.
        """
        # Log transform dampens blowouts
        mov = np.log(abs(point_diff) + 1) * 2.2

        # Correlation factor: expected blowouts matter less
        if elo_diff > 0 and point_diff > 0:
            correlation = 2.2 / (elo_diff * 0.001 + 2.2)
        else:
            correlation = 1.0

        return mov * correlation

    def update_ratings(
        self,
        home_team: str,
        away_team: str,
        home_score: int,
        away_score: int,
        game_id: Optional[str] = None,
        season: Optional[int] = None,
        week: Optional[int] = None
    ) -> Tuple[float, float]:
        """
        Update Elo ratings after a game.

        Returns:
            (home_rating_change, away_rating_change)
        """
        # Get current ratings
        home_elo = self.get_rating(home_team)
        away_elo = self.get_rating(away_team)

        # Adjust for home advantage
        home_elo_adj = home_elo + self.home_advantage

        # Expected scores (probability of winning)
        home_expected = self._expected_score(home_elo_adj, away_elo)
        away_expected = 1.0 - home_expected

        # Actual result (1 for win, 0.5 for tie, 0 for loss)
        if home_score > away_score:
            home_actual, away_actual = 1.0, 0.0
        elif away_score > home_score:
            home_actual, away_actual = 0.0, 1.0
        else:
            home_actual, away_actual = 0.5, 0.5

        # Margin of victory adjustment
        point_diff = home_score - away_score
        elo_diff = home_elo_adj - away_elo
        mov_mult = self._margin_of_victory_multiplier(point_diff, elo_diff)

        # Update ratings
        home_change = self.k_factor * mov_mult * (home_actual - home_expected)
        away_change = self.k_factor * mov_mult * (away_actual - away_expected)

        self.ratings[home_team] = home_elo + home_change
        self.ratings[away_team] = away_elo + away_change

        # Track history
        self.rating_history.append({
            'game_id': game_id,
            'season': season,
            'week': week,
            'home_team': home_team,
            'away_team': away_team,
            'home_elo_pre': home_elo,
            'away_elo_pre': away_elo,
            'home_elo_post': self.ratings[home_team],
            'away_elo_post': self.ratings[away_team],
            'home_score': home_score,
            'away_score': away_score
        })

        return home_change, away_change

    def predict_spread(self, home_team: str, away_team: str) -> float:
        """
        Predict point spread (home - away) using Elo ratings.

        Returns:
            Predicted spread (positive = home favored)
        """
        home_elo = self.get_rating(home_team)
        away_elo = self.get_rating(away_team)

        # Adjust for home advantage
        elo_diff = (home_elo + self.home_advantage) - away_elo

        # Convert Elo difference to point spread
        # Rule of thumb: 25 Elo points ≈ 1 point spread
        predicted_spread = elo_diff / 25.0

        return predicted_spread

    def predict_win_probability(self, home_team: str, away_team: str) -> float:
        """
        Predict win probability for home team.

        Returns:
            Probability between 0 and 1
        """
        home_elo = self.get_rating(home_team)
        away_elo = self.get_rating(away_team)

        # Adjust for home advantage
        home_elo_adj = home_elo + self.home_advantage

        return self._expected_score(home_elo_adj, away_elo)

    def regress_to_mean(self, season: int):
        """
        Regress all team ratings toward mean between seasons.
        Prevents ratings from drifting too far over time.
        """
        for team in self.ratings:
            current = self.ratings[team]
            self.ratings[team] = (
                current * (1 - self.regression_factor) +
                self.mean_rating * self.regression_factor
            )

        print(f"Regressed ratings for season {season} (factor: {self.regression_factor})")

    def fit(self, games_df: pl.DataFrame):
        """
        Train Elo model on historical games.
        Games must be in chronological order (CRITICAL for time-series).

        Args:
            games_df: Polars DataFrame with columns:
                - game_id, season, week
                - home_team, away_team
                - home_score, away_score
        """
        # Ensure chronological order
        games_df = games_df.sort(['season', 'week', 'game_id'])

        prev_season = None

        for row in games_df.iter_rows(named=True):
            # Regress to mean at start of new season
            if prev_season is not None and row['season'] != prev_season:
                self.regress_to_mean(row['season'])

            # Update ratings based on game result
            self.update_ratings(
                home_team=row['home_team'],
                away_team=row['away_team'],
                home_score=row['home_score'],
                away_score=row['away_score'],
                game_id=row['game_id'],
                season=row['season'],
                week=row['week']
            )

            prev_season = row['season']

        print(f"Elo model trained on {len(games_df)} games")

    def predict(self, games_df: pl.DataFrame) -> pl.DataFrame:
        """
        Predict spreads for future games using current Elo ratings.

        Returns:
            DataFrame with predictions added
        """
        predictions = []
        win_probs = []

        for row in games_df.iter_rows(named=True):
            pred_spread = self.predict_spread(row['home_team'], row['away_team'])
            win_prob = self.predict_win_probability(row['home_team'], row['away_team'])
            predictions.append(pred_spread)
            win_probs.append(win_prob)

        return games_df.with_columns([
            pl.Series('elo_predicted_spread', predictions),
            pl.Series('elo_home_win_prob', win_probs)
        ])

    def get_current_ratings(self) -> pl.DataFrame:
        """Get current Elo ratings for all teams as DataFrame."""
        return pl.DataFrame({
            'team': list(self.ratings.keys()),
            'elo_rating': list(self.ratings.values())
        }).sort('elo_rating', descending=True)

    def get_rating_history(self) -> pl.DataFrame:
        """Get historical Elo ratings as DataFrame."""
        if not self.rating_history:
            return pl.DataFrame()
        return pl.DataFrame(self.rating_history)

    def save_model(self, output_dir: str, model_name: str = 'elo_model'):
        """Save Elo ratings and history to JSON."""
        output_path = Path(output_dir) / f"{model_name}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'hyperparameters': {
                'k_factor': self.k_factor,
                'home_advantage': self.home_advantage,
                'initial_rating': self.initial_rating,
                'regression_factor': self.regression_factor,
                'mean_rating': self.mean_rating
            },
            'current_ratings': self.ratings,
            'rating_history': self.rating_history
        }

        with open(output_path, 'w') as f:
            json.dump(model_data, f, indent=2)

        print(f"Elo model saved to: {output_path}")
        return output_path

    def load_model(self, model_path: str):
        """Load Elo ratings and history from JSON."""
        with open(model_path, 'r') as f:
            model_data = json.load(f)

        # Restore hyperparameters
        params = model_data['hyperparameters']
        self.k_factor = params['k_factor']
        self.home_advantage = params['home_advantage']
        self.initial_rating = params['initial_rating']
        self.regression_factor = params['regression_factor']
        self.mean_rating = params['mean_rating']

        # Restore ratings
        self.ratings = model_data['current_ratings']
        self.rating_history = model_data['rating_history']

        print(f"Elo model loaded from: {model_path}")

    def __repr__(self) -> str:
        return (
            f"EloModel(k_factor={self.k_factor}, "
            f"home_advantage={self.home_advantage}, "
            f"teams={len(self.ratings)})"
        )
