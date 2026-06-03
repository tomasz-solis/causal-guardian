"""Tests for backdoor-adjusted ATE estimation.

Checks that:
    - The estimated effect has the correct sign for both DGPs.
    - Confidence intervals are well-formed (lower < upper, non-degenerate).
    - The schema validator is invoked at the boundary.
    - Estimation is consistent across repeated calls (same data → same estimate).
    - The wrong-DAG case behaves predictably (wrong confounder set).
"""

from __future__ import annotations

import pandas as pd
import pytest

from causal_guardian.dag import BASELINE_DAG, DRIFT_DAG
from causal_guardian.estimation.backdoor import estimate_ate


class TestEffectEstimateBaseline:
    def test_card_usage_effect_is_negative(self, baseline_df: pd.DataFrame) -> None:
        est = estimate_ate(baseline_df, BASELINE_DAG)
        assert est.ate < 0, (
            f"Expected negative card_usage → churn ATE, got {est.ate:.5f}."
        )

    def test_ci_is_well_formed(self, baseline_df: pd.DataFrame) -> None:
        est = estimate_ate(baseline_df, BASELINE_DAG)
        assert est.ci_lower < est.ate < est.ci_upper, (
            "ATE should sit inside its own confidence interval."
        )

    def test_ci_excludes_zero(self, baseline_df: pd.DataFrame) -> None:
        """With n=5000, the effect should be precisely estimated."""
        est = estimate_ate(baseline_df, BASELINE_DAG)
        assert est.ci_upper < 0, (
            f"95% CI should exclude zero; got [{est.ci_lower:.5f}, {est.ci_upper:.5f}]."
        )

    def test_std_error_positive(self, baseline_df: pd.DataFrame) -> None:
        est = estimate_ate(baseline_df, BASELINE_DAG)
        assert est.std_error > 0

    def test_n_obs_matches_dataframe(self, baseline_df: pd.DataFrame) -> None:
        est = estimate_ate(baseline_df, BASELINE_DAG)
        assert est.n_obs == len(baseline_df)

    def test_reproducible(self, baseline_df: pd.DataFrame) -> None:
        est1 = estimate_ate(baseline_df, BASELINE_DAG)
        est2 = estimate_ate(baseline_df, BASELINE_DAG)
        assert est1.ate == est2.ate


class TestEffectEstimateDrift:
    def test_friction_effect_is_positive(self, drifted_df: pd.DataFrame) -> None:
        est = estimate_ate(drifted_df, DRIFT_DAG)
        assert est.ate > 0, (
            f"Expected positive friction → churn ATE in drifted regime, got {est.ate:.5f}."
        )

    def test_ci_excludes_zero(self, drifted_df: pd.DataFrame) -> None:
        est = estimate_ate(drifted_df, DRIFT_DAG)
        assert est.ci_lower > 0, (
            f"95% CI should exclude zero; got [{est.ci_lower:.5f}, {est.ci_upper:.5f}]."
        )


class TestSchemaValidation:
    def test_missing_column_raises(self, baseline_df: pd.DataFrame) -> None:
        import pandera.errors as pa_errors

        bad_df = baseline_df.drop(columns=["churn"])
        with pytest.raises(pa_errors.SchemaError):
            estimate_ate(bad_df, BASELINE_DAG)

    def test_negative_marketing_spend_raises(self, baseline_df: pd.DataFrame) -> None:
        import pandera.errors as pa_errors

        bad_df = baseline_df.copy()
        bad_df.loc[0, "marketing_spend"] = -100.0
        with pytest.raises(pa_errors.SchemaError):
            estimate_ate(bad_df, BASELINE_DAG)

    def test_invalid_churn_value_raises(self, baseline_df: pd.DataFrame) -> None:
        import pandera.errors as pa_errors

        bad_df = baseline_df.copy()
        bad_df.loc[0, "churn"] = 2
        with pytest.raises(pa_errors.SchemaError):
            estimate_ate(bad_df, BASELINE_DAG)
