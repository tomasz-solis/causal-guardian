"""Configuration dataclasses for the Causal Guardian.

All numeric tunables live here. Nothing in the core logic should
contain a magic number - import from this module instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

# Avoid divide-by-zero when computing effect ratios.
EPSILON: Final[float] = 1e-10


@dataclass(frozen=True)
class DriftConfig:
    """Tunables for the drift detector and refutation suite.

    Args:
        refutation_threshold: Minimum effect-to-placebo-std ratio for
            the permutation test to pass. Higher = stricter.
        placebo_bias_threshold: Maximum allowed mean(placebo) / std(placebo).
            Keeps the placebo distribution centred near zero.
        effect_change_threshold: Fraction of baseline effect that triggers
            a WARNING severity. 0.5 means a 50% change in effect size.
        n_permutations: Number of permutations for the placebo refuter.
        n_bootstrap: Replications for the bootstrap refuter.
        n_subset_reps: Replications for the data-subset refuter.
        n_random_cc: Replications for the random-common-cause refuter.
        subset_fraction: Fraction of data used per data-subset replication.
        random_seed: Seed passed to all random operations for reproducibility.
    """

    refutation_threshold: float = 3.0
    placebo_bias_threshold: float = 0.5
    effect_change_threshold: float = 0.5
    n_permutations: int = 200
    n_bootstrap: int = 200
    n_subset_reps: int = 100
    n_random_cc: int = 100
    subset_fraction: float = 0.8
    random_seed: int = 42


@dataclass(frozen=True)
class StreamingConfig:
    """Tunables for the rolling-window drift monitor.

    Args:
        window_size: Number of timesteps in each estimation window.
        step_size: How many timesteps to advance per evaluation.
        cusum_slack: CUSUM slack parameter k (in units of baseline std).
            Typical value: 0.5 * expected_shift_magnitude / sigma.
        cusum_threshold: CUSUM alert threshold h (in units of baseline std).
            Common values: 4 - 5 for ~5% false-positive rate per 1000 steps.
        burn_in_periods: Timesteps used to estimate baseline μ and σ before
            monitoring begins.
    """

    window_size: int = 10
    step_size: int = 1
    cusum_slack: float = 0.5
    cusum_threshold: float = 4.0
    burn_in_periods: int = 20


# Reasonable defaults for the synthetic DGP.
@dataclass(frozen=True)
class SyntheticDGPConfig:
    """Parameters for synthetic data generation.

    Keeping these explicit makes it easy to run sensitivity sweeps
    by replacing individual fields.
    """

    n_companies: int = 5000
    random_seed: int = 42

    # Structural coefficients - baseline regime
    marketing_to_usage: float = 0.05
    friction_to_usage: float = -3.0
    usage_baseline: float = 50.0
    usage_noise_std: float = 5.0

    churn_intercept_logit: float = 2.0   # logit scale; maps to ~88% base churn
    usage_to_churn_logit: float = -0.02  # negative: more usage → less churn
    plan_to_churn_logit: float = -0.3    # higher plan tier → lower churn

    churn_noise_std: float = 0.3         # logit-scale noise


@dataclass(frozen=True)
class DriftedDGPConfig:
    """Parameters for the post-drift synthetic DGP.

    The drift: friction now directly drives churn; usage has no effect.
    """

    n_companies: int = 5000
    random_seed: int = 100

    # Upstream equations unchanged from baseline
    marketing_to_usage: float = 0.05
    friction_to_usage: float = -3.0
    usage_baseline: float = 50.0
    usage_noise_std: float = 5.0

    churn_intercept_logit: float = -0.4  # logit scale; lower baseline churn
    friction_to_churn_logit: float = 0.4  # friction now directly causes churn
    plan_to_churn_logit: float = -0.3
    churn_noise_std: float = 0.3
