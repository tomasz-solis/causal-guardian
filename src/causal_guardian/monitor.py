"""Causal drift monitor.

Compares the causal effect estimated on a baseline dataset against a
current dataset and reports whether the relationship has drifted.

The monitor is decoupled from the synthetic data generators - it accepts
any DataFrame that satisfies CompanyChurnSchema. For production use, swap
the synthetic generators for real data loaders and the rest is unchanged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from causal_guardian.config import EPSILON, DriftConfig
from causal_guardian.dag import CausalDAG
from causal_guardian.estimation.backdoor import EffectEstimate, estimate_ate
from causal_guardian.schemas import CompanyChurnSchema

import pandas as pd

logger = logging.getLogger(__name__)


class Severity(str, Enum):
    """Drift severity levels."""

    NONE = "NONE"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class DriftReport:
    """Result of a single drift check.

    Args:
        drift_detected: True if any drift criterion triggered.
        severity: Worst severity level across all triggered criteria.
        reasons: Human-readable explanation of each triggered criterion.
        baseline_estimate: The effect estimate from the reference period.
        current_estimate: The effect estimate from the current period.
    """

    drift_detected: bool
    severity: Severity
    reasons: tuple[str, ...]
    baseline_estimate: EffectEstimate
    current_estimate: EffectEstimate

    @property
    def summary(self) -> str:
        if not self.drift_detected:
            return (
                f"No drift. current ATE={self.current_estimate.ate:+.5f} "
                f"(baseline={self.baseline_estimate.ate:+.5f})."
            )
        return (
            f"Drift [{self.severity.value}]: "
            + "; ".join(self.reasons)
        )


class CausalDriftMonitor:
    """Detects drift in a causal relationship between two time windows.

    Establishes a baseline from one dataset, then compares subsequent
    datasets against that baseline on three criteria:
        1. Refutation failure - the effect is no longer distinguishable
           from noise (effect / std(placebo) < threshold).
        2. Effect size drift - the effect magnitude changed by more than
           ``config.effect_change_threshold``.
        3. Sign flip - the effect direction reversed.

    Args:
        dag: Causal DAG specifying the relationship to monitor.
        config: DriftConfig controlling thresholds and refutation params.
    """

    def __init__(self, dag: CausalDAG, config: DriftConfig | None = None) -> None:
        self.dag = dag
        self.config = config or DriftConfig()
        self._baseline: EffectEstimate | None = None

    def establish_baseline(self, df: pd.DataFrame) -> EffectEstimate:
        """Estimate the causal effect on baseline data and store it.

        Args:
            df: Baseline observational data satisfying CompanyChurnSchema.

        Returns:
            The baseline EffectEstimate.
        """
        CompanyChurnSchema.validate(df)
        self._baseline = estimate_ate(df, self.dag)
        logger.info("Baseline established: %s", self._baseline.summary)
        return self._baseline

    def check_drift(self, df: pd.DataFrame) -> DriftReport:
        """Check whether the causal relationship has drifted in new data.

        Args:
            df: Current-period data satisfying CompanyChurnSchema.

        Returns:
            DriftReport with drift status, severity, and reasons.

        Raises:
            RuntimeError: If ``establish_baseline()`` has not been called.
        """
        if self._baseline is None:
            raise RuntimeError(
                "Call establish_baseline() before check_drift()."
            )

        CompanyChurnSchema.validate(df)
        current = estimate_ate(df, self.dag)

        reasons: list[str] = []
        severity = Severity.NONE

        # Criterion 1: refutation failure via permutation test
        effect_ratio = _effect_to_placebo_ratio(df, self.dag, current.ate, self.config)
        if effect_ratio < self.config.refutation_threshold:
            reasons.append(
                f"Effect no longer distinguishable from noise "
                f"(ratio={effect_ratio:.2f}x, threshold={self.config.refutation_threshold}x)."
            )
            severity = Severity.CRITICAL

        # Criterion 2: effect size drift
        baseline_ate = self._baseline.ate
        if abs(baseline_ate) > EPSILON:
            pct_change = abs(current.ate - baseline_ate) / abs(baseline_ate)
            if pct_change > self.config.effect_change_threshold:
                reasons.append(
                    f"Effect magnitude changed by {pct_change:.1%} "
                    f"(baseline={baseline_ate:+.5f}, current={current.ate:+.5f})."
                )
                severity = max(severity, Severity.WARNING, key=_severity_rank)

        # Criterion 3: sign flip
        if baseline_ate != 0.0 and np.sign(baseline_ate) != np.sign(current.ate):
            reasons.append(
                f"Effect direction reversed "
                f"(baseline={baseline_ate:+.5f}, current={current.ate:+.5f})."
            )
            severity = Severity.CRITICAL

        report = DriftReport(
            drift_detected=bool(reasons),
            severity=severity,
            reasons=tuple(reasons),
            baseline_estimate=self._baseline,
            current_estimate=current,
        )
        logger.info(report.summary)
        return report

    @property
    def baseline(self) -> EffectEstimate | None:
        """The stored baseline estimate, or None if not yet established."""
        return self._baseline


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _effect_to_placebo_ratio(
    df: pd.DataFrame,
    dag: CausalDAG,
    true_effect: float,
    config: DriftConfig,
) -> float:
    """Compute |true_effect| / std(placebo distribution).

    Kept here (rather than in refutation.py) because the monitor needs
    a lightweight single-metric test, not the full refutation suite.
    """
    import statsmodels.api as sm

    rng = np.random.default_rng(config.random_seed)
    regressors = list(dag.confounders) + [dag.treatment]

    placebo_effects: list[float] = []
    for _ in range(config.n_permutations):
        df_perm = df.copy()
        df_perm[dag.treatment] = rng.permutation(df_perm[dag.treatment].to_numpy())
        X = sm.add_constant(df_perm[regressors])
        ols = sm.OLS(df_perm[dag.outcome], X).fit()
        placebo_effects.append(float(ols.params[dag.treatment]))

    std_placebo = float(np.std(placebo_effects))
    return abs(true_effect) / (std_placebo + EPSILON)


_SEVERITY_ORDER = {Severity.NONE: 0, Severity.WARNING: 1, Severity.CRITICAL: 2}


def _severity_rank(s: Severity) -> int:
    return _SEVERITY_ORDER[s]
