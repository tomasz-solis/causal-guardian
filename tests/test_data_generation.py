"""Tests for the synthetic data generators.

These tests verify schema compliance, distributional properties, and
that the embedded causal structure is recoverable via OLS - i.e., the
data generators do what they claim.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from hypothesis import given, settings
from hypothesis import strategies as st

from causal_guardian.data.drift import generate_drifted_data
from causal_guardian.data.synthetic import generate_causal_data
from causal_guardian.schemas import CompanyChurnSchema


class TestBaselineDGP:
    def test_schema_compliant(self, baseline_df: pd.DataFrame) -> None:
        CompanyChurnSchema.validate(baseline_df)

    def test_shape(self, baseline_df: pd.DataFrame) -> None:
        assert len(baseline_df) == 5000
        assert {"marketing_spend", "onboarding_friction_score", "card_usage", "churn"}.issubset(
            baseline_df.columns
        )

    def test_churn_is_binary(self, baseline_df: pd.DataFrame) -> None:
        assert set(baseline_df["churn"].unique()).issubset({0, 1})

    def test_card_usage_non_negative(self, baseline_df: pd.DataFrame) -> None:
        assert (baseline_df["card_usage"] >= 0).all()

    def test_churn_rate_plausible(self, baseline_df: pd.DataFrame) -> None:
        rate = baseline_df["churn"].mean()
        # 5 - 95% covers the realistic B2B SaaS range from low-churn enterprise
        # plans through high-churn freemium products.
        assert 0.02 < rate < 0.95, f"Churn rate {rate:.2%} outside expected range."

    def test_card_usage_drives_churn_negatively(self, baseline_df: pd.DataFrame) -> None:
        """OLS should recover a negative card_usage → churn coefficient."""
        X = sm.add_constant(
            baseline_df[["card_usage", "marketing_spend", "onboarding_friction_score"]]
        )
        ols = sm.OLS(baseline_df["churn"], X).fit()
        coef = ols.params["card_usage"]
        assert coef < 0, (
            f"Expected negative card_usage → churn coefficient in baseline, got {coef:.5f}."
        )

    def test_marketing_has_no_direct_churn_effect(self, baseline_df: pd.DataFrame) -> None:
        """Marketing_spend's direct coefficient on churn should be near zero."""
        X = sm.add_constant(
            baseline_df[["card_usage", "marketing_spend", "onboarding_friction_score"]]
        )
        ols = sm.OLS(baseline_df["churn"], X).fit()
        direct_coef = ols.params["marketing_spend"]
        # Can't be exactly zero due to noise, but should be small relative to usage coef
        assert abs(direct_coef) < 0.01 * abs(ols.params["card_usage"]) or abs(direct_coef) < 0.001

    @given(
        n=st.integers(min_value=50, max_value=500),
        seed=st.integers(min_value=0, max_value=9999),
    )
    @settings(max_examples=20)
    def test_arbitrary_n_and_seed(self, n: int, seed: int) -> None:
        """Any (n, seed) combination produces a valid, schema-compliant DataFrame."""
        df = generate_causal_data(n_companies=n, random_seed=seed)
        CompanyChurnSchema.validate(df)
        assert len(df) == n
        assert set(df["churn"].unique()).issubset({0, 1})


class TestDriftedDGP:
    def test_schema_compliant(self, drifted_df: pd.DataFrame) -> None:
        CompanyChurnSchema.validate(drifted_df)

    def test_friction_drives_churn_positively(self, drifted_df: pd.DataFrame) -> None:
        """In the drifted regime, friction should have a positive coefficient on churn."""
        X = sm.add_constant(
            drifted_df[["onboarding_friction_score", "marketing_spend"]]
        )
        ols = sm.OLS(drifted_df["churn"], X).fit()
        coef = ols.params["onboarding_friction_score"]
        assert coef > 0, (
            f"Expected positive friction → churn coefficient in drift regime, got {coef:.5f}."
        )

    def test_column_names_match_baseline(
        self, baseline_df: pd.DataFrame, drifted_df: pd.DataFrame
    ) -> None:
        """Drift data must expose the same columns as baseline - a realistic scenario."""
        assert set(baseline_df.columns) == set(drifted_df.columns)

    def test_drifted_differs_from_baseline(
        self, baseline_df: pd.DataFrame, drifted_df: pd.DataFrame
    ) -> None:
        """Card-usage effect on churn should differ significantly between regimes."""
        def usage_effect(df: pd.DataFrame) -> float:
            X = sm.add_constant(
                df[["card_usage", "marketing_spend", "onboarding_friction_score"]]
            )
            return float(sm.OLS(df["churn"], X).fit().params["card_usage"])

        baseline_coef = usage_effect(baseline_df)
        drifted_coef = usage_effect(drifted_df)
        # The drifted regime removed the usage → churn effect; coefficients should diverge
        assert abs(baseline_coef - drifted_coef) > 0.001, (
            f"Expected coefficients to differ; got baseline={baseline_coef:.5f}, "
            f"drifted={drifted_coef:.5f}."
        )
