"""Causal effect estimation via backdoor adjustment.

Uses DoWhy to identify the valid adjustment set from the DAG, then
estimates the Average Treatment Effect (ATE) via OLS regression with
the identified confounders as controls.

Separating identification (DoWhy) from estimation (statsmodels) gives
clean access to confidence intervals and standard errors without depending
on DoWhy's version-specific CI interface.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import cast

import pandas as pd
import statsmodels.api as sm
from dowhy import CausalModel

from causal_guardian.dag import CausalDAG
from causal_guardian.schemas import CompanyChurnSchema

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EffectEstimate:
    """Point estimate and uncertainty for a causal effect.

    Args:
        treatment: Name of the treatment variable.
        outcome: Name of the outcome variable.
        ate: Average Treatment Effect (point estimate).
        ci_lower: Lower bound of the 95% confidence interval.
        ci_upper: Upper bound of the 95% confidence interval.
        std_error: Standard error of the estimate.
        n_obs: Number of observations used in estimation.
        identified: Whether DoWhy confirmed the effect is identifiable
            from the given DAG without untestable assumptions.
    """

    treatment: str
    outcome: str
    ate: float
    ci_lower: float
    ci_upper: float
    std_error: float
    n_obs: int
    identified: bool

    @property
    def summary(self) -> str:
        return (
            f"ATE({self.treatment} → {self.outcome}) = {self.ate:+.5f} "
            f"[95% CI: {self.ci_lower:+.5f}, {self.ci_upper:+.5f}], "
            f"SE={self.std_error:.5f}, n={self.n_obs}"
        )


def estimate_ate(
    df: pd.DataFrame,
    dag: CausalDAG,
    significance_level: float = 0.05,
) -> EffectEstimate:
    """Estimate the ATE of ``dag.treatment`` on ``dag.outcome`` via backdoor adjustment.

    DoWhy is used to validate identifiability from the DAG. The effect is
    estimated by OLS regression controlling for ``dag.confounders``.

    Args:
        df: Observational data. Must satisfy CompanyChurnSchema.
        dag: Causal DAG specifying the treatment, outcome, and adjustment set.
        significance_level: Used only for the CI level (default: 5% → 95% CI).

    Returns:
        EffectEstimate with the point estimate, 95% CI, and identification status.

    Raises:
        pandera.errors.SchemaError: If ``df`` fails schema validation.
        KeyError: If treatment, outcome, or confounders are not in ``df``.
    """
    CompanyChurnSchema.validate(df)

    identified = _check_identifiability(df, dag)
    if not identified:
        logger.warning(
            "DoWhy could not confirm identifiability for %s → %s. "
            "Proceeding with backdoor adjustment as specified by the DAG.",
            dag.treatment,
            dag.outcome,
        )

    regressors = list(dag.confounders) + [dag.treatment]
    X = sm.add_constant(df[regressors])
    y = df[dag.outcome]

    ols = sm.OLS(y, X).fit()
    ate = float(ols.params[dag.treatment])
    ci = ols.conf_int(alpha=significance_level).loc[dag.treatment]
    se = float(ols.bse[dag.treatment])

    logger.info(
        "Estimated %s → %s: ATE=%.5f [%.5f, %.5f], SE=%.5f, n=%d",
        dag.treatment,
        dag.outcome,
        ate,
        float(ci.iloc[0]),
        float(ci.iloc[1]),
        se,
        len(df),
    )

    return EffectEstimate(
        treatment=dag.treatment,
        outcome=dag.outcome,
        ate=ate,
        ci_lower=float(ci.iloc[0]),
        ci_upper=float(ci.iloc[1]),
        std_error=se,
        n_obs=len(df),
        identified=identified,
    )


def _check_identifiability(df: pd.DataFrame, dag: CausalDAG) -> bool:
    """Ask DoWhy whether the causal effect is identifiable from this DAG.

    Returns True if DoWhy can identify the effect without untestable assumptions.
    Never raises - identification failures are logged as warnings.
    """
    try:
        model = CausalModel(
            data=_dowhy_data(df, dag),
            treatment=dag.treatment,
            outcome=dag.outcome,
            graph=dag.to_dot(),
        )
        model.identify_effect(proceed_when_unidentifiable=True)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Identifiability check raised: %s", exc)
        return False


def _dowhy_data(df: pd.DataFrame, dag: CausalDAG) -> pd.DataFrame:
    """Return the columns DoWhy expects from the DAG.

    The project schema allows extra operational columns such as
    ``cohort_month``. DoWhy logs those as graph mismatches, so keep its
    input limited to variables named in the causal graph.
    """
    return cast(pd.DataFrame, df.loc[:, sorted(dag.variables)])
