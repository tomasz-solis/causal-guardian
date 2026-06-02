"""Rolling-window causal monitoring on a time-series panel.

This module turns the static drift demo (generate A, generate B, compare)
into a proper time-series setup: a panel dataset with a structural break
at a known timestep, and a rolling estimator that must detect the break
from the data stream alone.

Classes:
    TimeSeriesPanel - generates a panel with a known break point.
    RollingCausalEstimator - fits the causal model over a sliding window.

The CUSUM detector (causal_guardian.detection.cusum) is applied to the
rolling estimate sequence to identify when the relationship shifts.
The ROC analysis script (scripts/roc_analysis.py) sweeps effect-size
and noise parameters to characterise detector sensitivity.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit

from causal_guardian.config import DriftConfig, StreamingConfig, SyntheticDGPConfig
from causal_guardian.dag import CausalDAG
from causal_guardian.detection.cusum import CUSUMDetector
from causal_guardian.schemas import TimeSeriesPanelSchema

logger = logging.getLogger(__name__)


@dataclass
class TimeSeriesPanel:
    """Synthetic panel dataset with a structural break at a known timestep.

    Generates ``n_timesteps`` periods, each with ``n_units_per_step`` companies.
    Before ``break_at``, the causal effect of ``treatment`` on ``outcome`` is
    ``pre_break_effect``. From ``break_at`` onward it switches to
    ``post_break_effect``.

    Args:
        dag: Causal DAG shared across all timesteps.
        pre_break_effect: True treatment coefficient before the break.
        post_break_effect: True treatment coefficient after the break.
        n_timesteps: Total number of time periods.
        n_units_per_step: Companies per period.
        break_at: Timestep index (0-based) where the structural break occurs.
        noise_std: Gaussian noise on the outcome logit.
        random_seed: NumPy seed.
    """

    dag: CausalDAG
    pre_break_effect: float = -0.003   # ATE on probability scale
    post_break_effect: float = 0.0     # Drift: usage effect disappears
    n_timesteps: int = 60
    n_units_per_step: int = 300
    break_at: int = 30
    noise_std: float = 0.02
    random_seed: int = 42

    _data: pd.DataFrame = field(default=None, init=False, repr=False)  # type: ignore[assignment]

    def generate(self) -> pd.DataFrame:
        """Generate the full panel and return it.

        The panel uses a linear probability DGP rather than the logistic form
        of the static synthetic generators. This is intentional: the streaming
        module exists to demonstrate CUSUM-based drift detection, and a clean
        linear signal makes the detector's behaviour interpretable. Production
        deployments should use the logistic DGP via the static monitor or
        replace OLS in the rolling estimator with a logistic regression.

        Returns:
            DataFrame satisfying TimeSeriesPanelSchema with a ``timestep``
            column identifying each period.
        """
        rng = np.random.default_rng(self.random_seed)
        chunks: list[pd.DataFrame] = []

        for t in range(self.n_timesteps):
            n = self.n_units_per_step
            effect = self.pre_break_effect if t < self.break_at else self.post_break_effect

            marketing_spend = rng.uniform(1_000, 10_000, n)
            friction = rng.uniform(0, 10, n)
            plan_tier = rng.integers(0, 4, n)

            # card_usage with strong residual variation so the rolling estimator
            # has a real signal to work with after backdoor adjustment.
            card_usage = (
                0.05 * marketing_spend
                - 3.0 * friction
                + 8.0 * plan_tier
                + 50.0
                + rng.normal(0, 15.0, n)
            )
            card_usage = np.maximum(card_usage, 0.0)

            # Linear probability of churn - clean LPM that the rolling OLS
            # recovers exactly under the correct adjustment set. Usage is
            # centred at its mean (~300) to keep probabilities in [0, 1]
            # without clipping at realistic effect sizes.
            base_rate = 0.4
            usage_mean = 300.0
            churn_prob = (
                base_rate
                + effect * (card_usage - usage_mean)
                - 0.02 * plan_tier
                + rng.normal(0, self.noise_std, n)
            )
            churn_prob = np.clip(churn_prob, 0.0, 1.0)
            churn = rng.binomial(1, churn_prob)

            chunk = pd.DataFrame(
                {
                    "timestep": t,
                    "marketing_spend": marketing_spend,
                    "onboarding_friction_score": friction,
                    "card_usage": card_usage,
                    "churn": churn,
                    "plan_tier": plan_tier.astype(int),
                }
            )
            chunks.append(chunk)

        self._data = pd.concat(chunks, ignore_index=True)
        TimeSeriesPanelSchema.validate(self._data)
        logger.info(
            "Generated panel: %d timesteps × %d units, break at t=%d.",
            self.n_timesteps,
            self.n_units_per_step,
            self.break_at,
        )
        return self._data

    @property
    def data(self) -> pd.DataFrame:
        """The generated panel; calls generate() if not yet done."""
        if self._data is None:
            self.generate()
        return self._data


@dataclass
class RollingCausalEstimator:
    """Estimates a causal effect over a sliding window of timesteps.

    At each step, fits the backdoor-adjusted OLS on the window of data
    and records the point estimate and its standard error.

    Args:
        dag: Causal DAG for all windows.
        config: StreamingConfig controlling window size and step size.
    """

    dag: CausalDAG
    config: StreamingConfig = field(default_factory=StreamingConfig)

    def fit(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Run the rolling estimator over the panel.

        Args:
            panel: Time-series panel with a ``timestep`` column.

        Returns:
            DataFrame with columns: timestep, ate, std_error, n_obs.
            One row per window ending at that timestep.
        """
        TimeSeriesPanelSchema.validate(panel)
        timesteps = sorted(panel["timestep"].unique())
        regressors = list(self.dag.confounders) + [self.dag.treatment]

        records = []
        window = self.config.window_size
        step = self.config.step_size

        for i in range(window, len(timesteps) + 1, step):
            window_steps = timesteps[i - window : i]
            mask = panel["timestep"].isin(window_steps)
            sub = panel[mask]

            X = sm.add_constant(sub[regressors])
            ols = sm.OLS(sub[self.dag.outcome], X).fit()

            records.append(
                {
                    "timestep": timesteps[i - 1],
                    "ate": float(ols.params[self.dag.treatment]),
                    "std_error": float(ols.bse[self.dag.treatment]),
                    "n_obs": len(sub),
                }
            )

        return pd.DataFrame(records)


def detect_drift_streaming(
    panel: pd.DataFrame,
    dag: CausalDAG,
    streaming_cfg: StreamingConfig | None = None,
    drift_cfg: DriftConfig | None = None,
) -> tuple[pd.DataFrame, list[int]]:
    """Run the full streaming drift detection pipeline on a panel.

    Combines the rolling estimator and the CUSUM detector. Uses the first
    ``burn_in_periods`` windows to calibrate the CUSUM baseline.

    Args:
        panel: Time-series panel from TimeSeriesPanel.generate().
        dag: Causal DAG used for rolling estimation.
        streaming_cfg: StreamingConfig for window/step/CUSUM params.
        drift_cfg: DriftConfig (used only for random_seed).

    Returns:
        Tuple of:
            - rolling_results: DataFrame from RollingCausalEstimator.
            - alert_timesteps: List of timesteps where CUSUM fired.
    """
    scfg = streaming_cfg or StreamingConfig()
    dcfg = drift_cfg or DriftConfig()

    estimator = RollingCausalEstimator(dag=dag, config=scfg)
    rolling = estimator.fit(panel)

    detector = CUSUMDetector(config=scfg)
    burn_in = scfg.burn_in_periods
    burn_in_values = rolling["ate"].to_numpy()[:burn_in]
    detector.fit(burn_in_values)

    alerts: list[int] = []
    for _, row in rolling.iloc[burn_in:].iterrows():
        fired = detector.update(int(row["timestep"]), float(row["ate"]))
        if fired:
            alerts.append(int(row["timestep"]))

    logger.info(
        "Streaming detection complete: %d alerts at timesteps %s.",
        len(alerts),
        alerts[:5],
    )
    return rolling, alerts
