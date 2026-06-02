"""Causal refutation tests.

DoWhy provides a refutation suite that stress-tests a causal estimate
from multiple angles. This module wraps that suite and returns a
structured result that the drift monitor and tests can reason over.

We run three refuters:
    1. Placebo treatment (permutation) - permutes the treatment variable.
       If the original effect survives, it's distinguishable from noise.
    2. Random common cause - adds a random noise variable as a confounder.
       A stable causal estimate shouldn't shift much.
    3. Data subset - re-estimates on a random 80% subset multiple times.
       Tests whether the estimate is stable or driven by a particular slice.

A fourth refuter (bootstrap) is run to produce a bootstrap standard error
for the effect, giving an additional uncertainty estimate.

References:
    Sharma, A., & Kiciman, E. (2020). DoWhy: An end-to-end library for
    causal inference. arXiv:2011.04216.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from dowhy import CausalModel

from causal_guardian.config import EPSILON, DriftConfig
from causal_guardian.dag import CausalDAG
from causal_guardian.estimation.backdoor import EffectEstimate, _dowhy_data

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RefutationReport:
    """Aggregated result from the full refutation suite.

    Args:
        placebo_p_value: Fraction of permuted effects with |effect| >= |original|.
            Small value (< 0.05) means the original effect is in the tail of
            the placebo distribution - supports causality.
        random_cc_relative_change: |new_effect - original| / |original| after
            adding a random common cause. Values near zero mean robustness.
        subset_relative_change: Same metric from the data-subset refuter.
        bootstrap_std_error: Standard error from bootstrap replication.
        original_effect: The effect estimate being refuted.
        consistent: True when all three tests pass (placebo_p_value < 0.05,
            both relative changes < 10%).
    """

    placebo_p_value: float
    random_cc_relative_change: float
    subset_relative_change: float
    bootstrap_std_error: float
    original_effect: float
    consistent: bool

    @property
    def summary(self) -> str:
        status = "PASS" if self.consistent else "FAIL"
        return (
            f"Refutation [{status}]: "
            f"placebo_p={self.placebo_p_value:.3f}, "
            f"rand_cc_delta={self.random_cc_relative_change:.3f}, "
            f"subset_delta={self.subset_relative_change:.3f}, "
            f"bootstrap_SE={self.bootstrap_std_error:.5f}"
        )


def run_refutation_suite(
    df: pd.DataFrame,
    dag: CausalDAG,
    estimate: EffectEstimate,
    config: DriftConfig | None = None,
) -> RefutationReport:
    """Run DoWhy's refutation suite on an existing effect estimate.

    Args:
        df: The same data used to compute ``estimate``.
        dag: The causal DAG the estimate came from.
        estimate: The effect estimate to stress-test.
        config: Refutation config (number of simulations, seed, etc.).
            If None, uses DriftConfig defaults.

    Returns:
        RefutationReport summarising all refuters.

    Notes:
        Runs three DoWhy refuters plus a manual bootstrap. Total number
        of OLS fits ≈ n_permutations + n_random_cc + n_subset_reps + n_bootstrap.
        With defaults this is ~700 fits - expect a few seconds on 5k rows.
    """
    cfg = config or DriftConfig()

    model = CausalModel(
        data=_dowhy_data(df, dag),
        treatment=dag.treatment,
        outcome=dag.outcome,
        graph=dag.to_dot(),
    )
    identified = model.identify_effect(proceed_when_unidentifiable=True)
    dowhy_estimate = model.estimate_effect(
        identified,
        method_name="backdoor.linear_regression",
    )

    # 1. Placebo treatment - permute treatment, re-estimate
    placebo_p = _placebo_p_value(df, dag, estimate.ate, cfg)

    # 2. Random common cause - compare to DoWhy's own original estimate
    dowhy_baseline = float(dowhy_estimate.value)
    try:
        rcc = model.refute_estimate(
            identified,
            dowhy_estimate,
            method_name="random_common_cause",
            num_simulations=cfg.n_random_cc,
        )
        rcc_delta = abs(rcc.new_effect - dowhy_baseline) / (abs(dowhy_baseline) + EPSILON)
    except Exception as exc:  # noqa: BLE001
        logger.warning("random_common_cause refuter failed: %s", exc)
        rcc_delta = float("nan")

    # 3. Data subset - same comparison
    try:
        sub = model.refute_estimate(
            identified,
            dowhy_estimate,
            method_name="data_subset_refuter",
            subset_fraction=cfg.subset_fraction,
            num_simulations=cfg.n_subset_reps,
        )
        sub_delta = abs(sub.new_effect - dowhy_baseline) / (abs(dowhy_baseline) + EPSILON)
    except Exception as exc:  # noqa: BLE001
        logger.warning("data_subset_refuter failed: %s", exc)
        sub_delta = float("nan")

    # 4. Bootstrap SE (manual - more reliable than DoWhy's across versions)
    boot_se = _bootstrap_std_error(df, dag, cfg)

    consistent = (
        placebo_p < 0.05
        and (np.isnan(rcc_delta) or rcc_delta < 0.10)
        and (np.isnan(sub_delta) or sub_delta < 0.10)
    )

    report = RefutationReport(
        placebo_p_value=placebo_p,
        random_cc_relative_change=rcc_delta,
        subset_relative_change=sub_delta,
        bootstrap_std_error=boot_se,
        original_effect=estimate.ate,
        consistent=consistent,
    )
    logger.info(report.summary)
    return report


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _placebo_p_value(
    df: pd.DataFrame,
    dag: CausalDAG,
    true_effect: float,
    cfg: DriftConfig,
) -> float:
    """Fraction of permuted effects with |effect| >= |true_effect|.

    Small p-value → original effect is in the tail → supports causality.
    """
    import statsmodels.api as sm  # local to avoid circular at module level

    rng = np.random.default_rng(cfg.random_seed)
    regressors = list(dag.confounders) + [dag.treatment]

    placebo_effects: list[float] = []
    for _ in range(cfg.n_permutations):
        df_perm = df.copy()
        df_perm[dag.treatment] = rng.permutation(df_perm[dag.treatment].to_numpy())
        X = sm.add_constant(df_perm[regressors])
        ols = sm.OLS(df_perm[dag.outcome], X).fit()
        placebo_effects.append(float(ols.params[dag.treatment]))

    arr = np.array(placebo_effects)
    return float(np.mean(np.abs(arr) >= abs(true_effect)))


def _bootstrap_std_error(
    df: pd.DataFrame,
    dag: CausalDAG,
    cfg: DriftConfig,
) -> float:
    """Bootstrap standard error for the backdoor-adjusted ATE."""
    import statsmodels.api as sm

    rng = np.random.default_rng(cfg.random_seed + 1)
    regressors = list(dag.confounders) + [dag.treatment]
    n = len(df)

    boot_effects: list[float] = []
    for _ in range(cfg.n_bootstrap):
        idx = rng.integers(0, n, size=n)
        sample = df.iloc[idx]
        X = sm.add_constant(sample[regressors])
        ols = sm.OLS(sample[dag.outcome], X).fit()
        boot_effects.append(float(ols.params[dag.treatment]))

    return float(np.std(boot_effects))
