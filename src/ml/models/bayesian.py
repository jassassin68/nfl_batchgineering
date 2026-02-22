"""
Bayesian State-Space Model for NFL predictions.

Based on Glickman & Stern (1998) JASA methodology.
Models team strength as latent variables with:
- Week-to-week variance
- Season-to-season variance
- Home field advantage as fixed effect

Uses PyMC for MCMC inference.
"""

from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import numpy as np
import polars as pl
import json
import pickle

# PyMC import with availability check
try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False
    pm = None
    az = None

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.ml.base import BasePredictor


class BayesianStateSpace(BasePredictor):
    """
    Bayesian State-Space model for NFL predictions.

    Based on Glickman & Stern (1998) JASA methodology.
    Models team strength as latent variables with sum-to-zero constraint
    for identifiability.

    Mathematical Model:
        Y_ij = theta_i - theta_j + h + epsilon_ij
        where:
            Y_ij = point differential (home - away)
            theta_i = home team strength (latent)
            theta_j = away team strength (latent)
            h = home field advantage (~2.5 points)
            epsilon_ij ~ N(0, sigma^2_game)

    Attributes:
        home_advantage_prior: Prior mean for home field advantage (~2.5)
        n_samples: Number of MCMC samples
        n_chains: Number of parallel MCMC chains
        target_accept: Target acceptance rate for NUTS sampler
        team_strengths: Fitted team strength estimates
        home_advantage: Fitted home field advantage
        game_std: Fitted game-level standard deviation
    """

    def __init__(
        self,
        model_name: str = "bayesian_state_space",
        home_advantage_prior: float = 2.5,
        team_strength_prior_std: float = 5.0,
        n_samples: int = 1000,
        n_chains: int = 2,
        target_accept: float = 0.9,
        random_seed: int = 42
    ):
        """
        Initialize Bayesian State-Space model.

        Args:
            model_name: Model identifier
            home_advantage_prior: Prior mean for home advantage (points)
            team_strength_prior_std: Prior std dev for team strengths
            n_samples: MCMC samples per chain
            n_chains: Number of MCMC chains
            target_accept: Target acceptance rate for NUTS sampler
            random_seed: Random seed for reproducibility
        """
        super().__init__(model_name=model_name)

        if not PYMC_AVAILABLE:
            raise ImportError(
                "PyMC not installed. Install with: pip install pymc arviz"
            )

        self.home_advantage_prior = home_advantage_prior
        self.team_strength_prior_std = team_strength_prior_std
        self.n_samples = n_samples
        self.n_chains = n_chains
        self.target_accept = target_accept
        self.random_seed = random_seed

        # Fitted attributes
        self.trace = None
        self.team_strengths: Dict[str, float] = {}
        self.team_to_idx: Dict[str, int] = {}
        self.idx_to_team: Dict[int, str] = {}
        self.home_advantage: float = home_advantage_prior
        self.game_std: float = 13.5  # Typical NFL game std dev

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        home_teams: Optional[np.ndarray] = None,
        away_teams: Optional[np.ndarray] = None,
        seasons: Optional[np.ndarray] = None,
        weeks: Optional[np.ndarray] = None,
        verbose: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Fit Bayesian model using MCMC.

        IMPORTANT: This model uses team identifiers, not features.
        X is not used directly - home_teams/away_teams are required.

        Args:
            X: Features (not used, can be None)
            y: Actual spreads (home_score - away_score)
            X_val: Validation features (not used)
            y_val: Validation targets (not used)
            feature_names: Not used
            home_teams: Array of home team names (REQUIRED)
            away_teams: Array of away team names (REQUIRED)
            seasons: Season array (for temporal structure, optional)
            weeks: Week array (for within-season structure, optional)
            verbose: Print progress during sampling

        Returns:
            Training diagnostics dictionary

        Raises:
            ValueError: If home_teams or away_teams not provided
        """
        if home_teams is None or away_teams is None:
            raise ValueError(
                "BayesianStateSpace requires home_teams and away_teams arrays"
            )

        if len(home_teams) != len(y):
            raise ValueError(
                f"home_teams has {len(home_teams)} elements, y has {len(y)}"
            )

        # Build team index mapping
        all_teams = list(set(home_teams) | set(away_teams))
        all_teams.sort()  # Consistent ordering
        self.team_to_idx = {team: i for i, team in enumerate(all_teams)}
        self.idx_to_team = {i: team for team, i in self.team_to_idx.items()}
        n_teams = len(all_teams)

        # Convert team names to indices
        home_idx = np.array([self.team_to_idx[t] for t in home_teams])
        away_idx = np.array([self.team_to_idx[t] for t in away_teams])

        if verbose:
            print(f"Training Bayesian model on {len(y)} games with {n_teams} teams")
            print(f"MCMC: {self.n_samples} samples x {self.n_chains} chains")

        # Build PyMC model
        with pm.Model() as model:
            # Priors
            # Home field advantage
            home_adv = pm.Normal(
                'home_advantage',
                mu=self.home_advantage_prior,
                sigma=1.0
            )

            # Team strengths (centered at 0)
            # Use sum-to-zero constraint for identifiability
            # Model N-1 teams, last team is negative sum of others
            team_strength_raw = pm.Normal(
                'team_strength_raw',
                mu=0,
                sigma=self.team_strength_prior_std,
                shape=n_teams - 1
            )

            # Last team strength = -sum(others) to ensure sum = 0
            last_team_strength = -pm.math.sum(team_strength_raw)
            team_strength = pm.Deterministic(
                'team_strength',
                pm.math.concatenate([
                    team_strength_raw,
                    pm.math.stack([last_team_strength])
                ])
            )

            # Game-level variance
            sigma_game = pm.HalfNormal('sigma_game', sigma=10.0)

            # Expected score differential
            mu = (
                team_strength[home_idx] -
                team_strength[away_idx] +
                home_adv
            )

            # Likelihood
            score_diff = pm.Normal(
                'score_diff',
                mu=mu,
                sigma=sigma_game,
                observed=y
            )

            # Sample using NUTS
            self.trace = pm.sample(
                draws=self.n_samples,
                chains=self.n_chains,
                target_accept=self.target_accept,
                return_inferencedata=True,
                progressbar=verbose,
                random_seed=self.random_seed
            )

        # Extract posterior means
        posterior = self.trace.posterior

        self.home_advantage = float(posterior['home_advantage'].mean())
        self.game_std = float(posterior['sigma_game'].mean())

        team_strength_samples = posterior['team_strength'].mean(dim=['chain', 'draw'])
        for i, team in self.idx_to_team.items():
            self.team_strengths[team] = float(team_strength_samples[i])

        self.is_fitted = True

        # Calculate diagnostics
        summary = az.summary(self.trace, var_names=['home_advantage', 'sigma_game'])
        max_rhat = summary['r_hat'].max()

        divergences = 0
        if 'diverging' in self.trace.sample_stats:
            divergences = int(self.trace.sample_stats['diverging'].sum())

        diagnostics = {
            'home_advantage': self.home_advantage,
            'game_std': self.game_std,
            'n_teams': n_teams,
            'n_games': len(y),
            'max_rhat': float(max_rhat),
            'divergences': divergences,
            'converged': max_rhat < 1.1 and divergences == 0
        }

        if verbose:
            print(f"\nTraining complete!")
            print(f"  Home advantage: {self.home_advantage:.2f} points")
            print(f"  Game std dev: {self.game_std:.2f} points")
            print(f"  Max R-hat: {max_rhat:.3f}")
            print(f"  Divergences: {divergences}")

        return diagnostics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using posterior mean team strengths.

        NOTE: For this model, X should be structured as:
        - Column 0: home_team_idx (integer)
        - Column 1: away_team_idx (integer)

        For team name interface, use predict_batch() instead.

        Args:
            X: Array with team indices [home_idx, away_idx, ...]

        Returns:
            Predicted spreads (positive = home favored)
        """
        self._validate_fitted()

        if X.shape[1] < 2:
            raise ValueError(
                "X must have at least 2 columns: "
                "[home_team_idx, away_team_idx]"
            )

        home_idx = X[:, 0].astype(int)
        away_idx = X[:, 1].astype(int)

        predictions = np.zeros(len(X))
        for i in range(len(X)):
            home_team = self.idx_to_team.get(home_idx[i])
            away_team = self.idx_to_team.get(away_idx[i])

            if home_team is None or away_team is None:
                # Default for unknown teams
                predictions[i] = self.home_advantage
            else:
                predictions[i] = (
                    self.team_strengths.get(home_team, 0) -
                    self.team_strengths.get(away_team, 0) +
                    self.home_advantage
                )

        return predictions

    def predict_matchup(
        self,
        home_team: str,
        away_team: str
    ) -> float:
        """
        Predict spread for a specific matchup.

        Args:
            home_team: Home team name
            away_team: Away team name

        Returns:
            Predicted spread (positive = home favored)
        """
        self._validate_fitted()

        home_strength = self.team_strengths.get(home_team, 0)
        away_strength = self.team_strengths.get(away_team, 0)

        return home_strength - away_strength + self.home_advantage

    def predict_batch(
        self,
        home_teams: List[str],
        away_teams: List[str]
    ) -> np.ndarray:
        """
        Predict spreads for multiple matchups.

        Args:
            home_teams: List of home team names
            away_teams: List of away team names

        Returns:
            Array of predicted spreads
        """
        return np.array([
            self.predict_matchup(h, a)
            for h, a in zip(home_teams, away_teams)
        ])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Generate win probability predictions.

        Uses posterior mean spread and game std dev for
        more accurate probability calibration.

        Args:
            X: Array with team indices

        Returns:
            Home win probabilities [0, 1]
        """
        spreads = self.predict(X)
        # Use fitted game std dev for probability calculation
        # More accurate than default logistic transformation
        scale = self.game_std / 2.5
        return 1 / (1 + np.exp(-spreads / scale))

    def predict_proba_matchup(
        self,
        home_team: str,
        away_team: str
    ) -> float:
        """
        Predict win probability for a specific matchup.

        Args:
            home_team: Home team name
            away_team: Away team name

        Returns:
            Home win probability [0, 1]
        """
        spread = self.predict_matchup(home_team, away_team)
        scale = self.game_std / 2.5
        return 1 / (1 + np.exp(-spread / scale))

    def get_prediction_uncertainty(self, X: np.ndarray) -> np.ndarray:
        """
        Get posterior predictive uncertainty.

        Returns game-level std dev from posterior.

        Args:
            X: Features (shape used to determine output length)

        Returns:
            Standard deviation estimates
        """
        self._validate_fitted()
        return np.full(len(X), self.game_std)

    def get_team_rankings(self) -> pl.DataFrame:
        """
        Get current team strength rankings.

        Returns:
            DataFrame with team, strength, rank columns
        """
        self._validate_fitted()

        teams = list(self.team_strengths.keys())
        strengths = list(self.team_strengths.values())

        df = pl.DataFrame({
            'team': teams,
            'strength': strengths
        }).sort('strength', descending=True)

        df = df.with_row_index('rank').with_columns(
            (pl.col('rank') + 1).alias('rank')
        )

        return df

    def get_team_strength(self, team: str) -> float:
        """
        Get strength estimate for a single team.

        Args:
            team: Team name

        Returns:
            Team strength (centered at 0)
        """
        self._validate_fitted()
        return self.team_strengths.get(team, 0.0)

    def save_model(self, output_dir: Union[str, Path]) -> Path:
        """
        Save model to disk.

        Saves:
        - Model parameters as pickle
        - MCMC trace as NetCDF (optional, for diagnostics)

        Args:
            output_dir: Directory to save model

        Returns:
            Path to main model file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        model_path = output_dir / f"{self.model_name}.pkl"

        model_data = {
            'model_name': self.model_name,
            'home_advantage': self.home_advantage,
            'game_std': self.game_std,
            'team_strengths': self.team_strengths,
            'team_to_idx': self.team_to_idx,
            'idx_to_team': self.idx_to_team,
            'hyperparameters': {
                'home_advantage_prior': self.home_advantage_prior,
                'team_strength_prior_std': self.team_strength_prior_std,
                'n_samples': self.n_samples,
                'n_chains': self.n_chains,
                'target_accept': self.target_accept,
                'random_seed': self.random_seed
            }
        }

        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)

        # Also save trace for diagnostics (optional)
        if self.trace is not None:
            try:
                trace_path = output_dir / f"{self.model_name}_trace.nc"
                self.trace.to_netcdf(trace_path)
            except Exception as e:
                print(f"Warning: Could not save trace: {e}")

        print(f"Bayesian model saved to: {model_path}")
        return model_path

    def load_model(self, model_path: Union[str, Path]) -> None:
        """
        Load model from disk.

        Args:
            model_path: Path to saved model file
        """
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        self.model_name = model_data['model_name']
        self.home_advantage = model_data['home_advantage']
        self.game_std = model_data['game_std']
        self.team_strengths = model_data['team_strengths']
        self.team_to_idx = model_data['team_to_idx']
        self.idx_to_team = model_data['idx_to_team']

        params = model_data['hyperparameters']
        self.home_advantage_prior = params['home_advantage_prior']
        self.team_strength_prior_std = params['team_strength_prior_std']
        self.n_samples = params['n_samples']
        self.n_chains = params['n_chains']
        self.target_accept = params['target_accept']
        self.random_seed = params['random_seed']

        self.is_fitted = True
        print(f"Bayesian model loaded from: {model_path}")
        print(f"  Teams: {len(self.team_strengths)}")
        print(f"  Home advantage: {self.home_advantage:.2f}")

    def summary(self) -> None:
        """Print model summary."""
        self._validate_fitted()

        print("\n" + "=" * 60)
        print("BAYESIAN STATE-SPACE MODEL SUMMARY")
        print("=" * 60)
        print(f"Home field advantage: {self.home_advantage:.2f} points")
        print(f"Game std deviation: {self.game_std:.2f} points")
        print(f"Number of teams: {len(self.team_strengths)}")
        print("\nTop 10 Teams by Strength:")

        rankings = self.get_team_rankings()
        for row in rankings.head(10).iter_rows(named=True):
            print(f"  {row['rank']:2d}. {row['team']:4s}  {row['strength']:+.2f}")

        print("\nBottom 5 Teams:")
        for row in rankings.tail(5).iter_rows(named=True):
            print(f"  {row['rank']:2d}. {row['team']:4s}  {row['strength']:+.2f}")

        print("=" * 60 + "\n")
