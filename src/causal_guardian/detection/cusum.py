"""CUSUM change-point detector.

The CUSUM (Cumulative Sum) detector monitors a sequence of observations
for a shift in the mean. It's well-suited for detecting when a causal
effect estimate has moved away from its baseline level.

Algorithm:
    Given observations X_1, X_2, ..., normalised by the baseline mean μ₀
    and standard deviation σ₀, the CUSUM statistic at time t is:

        S_t = max(0, S_{t-1} + (X_t - μ₀)/σ₀ - k)

    where k is the slack parameter (typically 0.5 × expected shift in units
    of σ₀). An alert fires when S_t > h.

References:
    Page, E. S. (1954). Continuous inspection schemes.
    Biometrika, 41(1/2), 100 - 115.

    Montgomery, D. C. (2020). Introduction to Statistical Quality Control
    (8th ed.), Chapter 9.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from causal_guardian.config import StreamingConfig

logger = logging.getLogger(__name__)


@dataclass
class CUSUMDetector:
    """Two-sided CUSUM detector for changes in a monitored time series.

    Maintains separate statistics for upward shifts (S_pos) and downward
    shifts (S_neg). An alert fires when either exceeds the threshold.

    Args:
        config: StreamingConfig controlling slack and threshold.
        baseline_mean: Baseline mean μ₀. Estimated from the burn-in period
            by ``fit()``.
        baseline_std: Baseline standard deviation σ₀. Estimated from burn-in.
    """

    config: StreamingConfig = field(default_factory=StreamingConfig)
    baseline_mean: float = 0.0
    baseline_std: float = 1.0

    # Internal CUSUM state
    _s_pos: float = field(default=0.0, init=False, repr=False)
    _s_neg: float = field(default=0.0, init=False, repr=False)
    _fitted: bool = field(default=False, init=False, repr=False)
    _history: list[tuple[int, float, float]] = field(
        default_factory=list, init=False, repr=False
    )

    def fit(self, burn_in_values: np.ndarray) -> "CUSUMDetector":
        """Estimate μ₀ and σ₀ from a burn-in sequence and reset state.

        Args:
            burn_in_values: 1-D array of observations during the stable
                baseline period (before monitoring begins).

        Returns:
            self, for chaining.
        """
        if len(burn_in_values) < 5:
            raise ValueError(
                f"Need at least 5 burn-in observations, got {len(burn_in_values)}."
            )
        self.baseline_mean = float(np.mean(burn_in_values))
        self.baseline_std = float(np.std(burn_in_values, ddof=1))
        if self.baseline_std < 1e-8:
            logger.warning(
                "Burn-in std is near zero (%.2e). "
                "CUSUM may false-positive on numerical noise.",
                self.baseline_std,
            )
            self.baseline_std = 1e-8
        self._s_pos = 0.0
        self._s_neg = 0.0
        self._fitted = True
        self._history = []
        logger.info(
            "CUSUM fitted: μ₀=%.4f, σ₀=%.4f from %d observations.",
            self.baseline_mean,
            self.baseline_std,
            len(burn_in_values),
        )
        return self

    def update(self, timestep: int, value: float) -> bool:
        """Incorporate one new observation and return whether an alert fires.

        Args:
            timestep: Integer timestep label (used for reporting only).
            value: The observed statistic (e.g. current ATE estimate).

        Returns:
            True if CUSUM has exceeded the threshold (drift detected).

        Raises:
            RuntimeError: If ``fit()`` has not been called.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before update().")

        k = self.config.cusum_slack
        h = self.config.cusum_threshold
        z = (value - self.baseline_mean) / self.baseline_std

        self._s_pos = max(0.0, self._s_pos + z - k)
        self._s_neg = max(0.0, self._s_neg - z - k)

        alert = self._s_pos > h or self._s_neg > h
        self._history.append((timestep, self._s_pos, self._s_neg))

        if alert:
            direction = "upward" if self._s_pos > h else "downward"
            logger.warning(
                "CUSUM alert at t=%d: %s shift detected "
                "(S_pos=%.2f, S_neg=%.2f, threshold=%.1f).",
                timestep,
                direction,
                self._s_pos,
                self._s_neg,
                h,
            )

        return alert

    def reset(self) -> None:
        """Reset CUSUM statistics to zero without clearing baseline."""
        self._s_pos = 0.0
        self._s_neg = 0.0
        self._history = []

    @property
    def statistics(self) -> np.ndarray:
        """CUSUM statistics history as (timestep, S_pos, S_neg) array."""
        return np.array(self._history)

    @property
    def current_s_pos(self) -> float:
        """Current upward CUSUM statistic."""
        return self._s_pos

    @property
    def current_s_neg(self) -> float:
        """Current downward CUSUM statistic."""
        return self._s_neg
