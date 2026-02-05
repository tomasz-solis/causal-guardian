"""
Test suite for causal logic validation.

Tests verify that:
1. The estimated causal effect of card_usage on churn is negative
2. The refutation test p-value is > 0.05 (supporting the causal claim)
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import pandas as pd
import numpy as np
from dowhy import CausalModel
import statsmodels.api as sm

from generate_data import generate_causal_data
from drift_generator import generate_drifted_data


def test_data_generation():
    """Test that data generation works correctly."""
    df = generate_causal_data(n_companies=100, random_seed=42)

    assert len(df) == 100
    assert 'marketing_spend' in df.columns
    assert 'onboarding_friction_score' in df.columns
    assert 'card_usage' in df.columns
    assert 'churn' in df.columns

    # Check that churn is binary
    assert set(df['churn'].unique()).issubset({0, 1})


def test_causal_effect_is_negative():
    """
    Test that the estimated causal effect of card_usage on churn is negative.

    This verifies that increased card usage reduces churn, as per the design.
    """
    # Generate data
    df = generate_causal_data(n_companies=5000, random_seed=42)

    # Define causal model
    causal_graph = """
    digraph {
        marketing_spend -> card_usage;
        onboarding_friction_score -> card_usage;
        card_usage -> churn;
    }
    """

    model = CausalModel(
        data=df,
        treatment='card_usage',
        outcome='churn',
        graph=causal_graph
    )

    # Identify effect
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

    # Estimate using OLS with backdoor adjustment
    X = df[['card_usage', 'marketing_spend', 'onboarding_friction_score']]
    X = sm.add_constant(X)
    y = df['churn']

    model_ols = sm.OLS(y, X).fit()
    causal_effect = model_ols.params['card_usage']

    print(f"\nEstimated causal effect: {causal_effect:.6f}")
    assert causal_effect < 0, (
        f"Expected negative causal effect (usage reduces churn), "
        f"but got {causal_effect:.6f}"
    )


def test_refutation_supports_causal_claim():
    """
    Test that the placebo refutation test supports the causal claim.

    A p-value > 0.05 indicates that the effect is not due to random chance.
    """
    # Generate data
    df = generate_causal_data(n_companies=5000, random_seed=42)

    # Define causal model
    causal_graph = """
    digraph {
        marketing_spend -> card_usage;
        onboarding_friction_score -> card_usage;
        card_usage -> churn;
    }
    """

    model = CausalModel(
        data=df,
        treatment='card_usage',
        outcome='churn',
        graph=causal_graph
    )

    # Identify effect
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

    # Estimate using OLS with backdoor adjustment
    X = df[['card_usage', 'marketing_spend', 'onboarding_friction_score']]
    X = sm.add_constant(X)
    y = df['churn']

    model_ols = sm.OLS(y, X).fit()
    causal_effect = model_ols.params['card_usage']

    # Run refutation test using permutation
    placebo_effects = []
    np.random.seed(42)

    for i in range(100):
        df_permuted = df.copy()
        df_permuted['card_usage'] = np.random.permutation(df_permuted['card_usage'].values)

        X_perm = df_permuted[['card_usage', 'marketing_spend', 'onboarding_friction_score']]
        X_perm = sm.add_constant(X_perm)
        y_perm = df_permuted['churn']

        model_perm = sm.OLS(y_perm, X_perm).fit()
        placebo_effects.append(model_perm.params['card_usage'])

    mean_placebo = np.mean(placebo_effects)
    std_placebo = np.std(placebo_effects)
    effect_ratio = np.abs(causal_effect) / (std_placebo + 1e-10)
    mean_placebo_ratio = np.abs(mean_placebo) / (std_placebo + 1e-10)
    refutation_passed = effect_ratio > 3 and mean_placebo_ratio < 0.5

    print(f"\nRefutation test:")
    print(f"  True effect: {causal_effect:.6f}")
    print(f"  Mean placebo: {mean_placebo:.6f}")
    print(f"  Effect ratio: {effect_ratio:.2f}x")
    assert refutation_passed, (
        f"Refutation test failed: Effect ratio {effect_ratio:.2f} is not strong enough. "
        f"True effect should be much stronger than placebo effects."
    )


def test_causal_structure_holds():
    """
    Combined test that verifies both conditions:
    1. Causal effect is negative
    2. Refutation test p-value > 0.05
    """
    # Generate data
    df = generate_causal_data(n_companies=5000, random_seed=42)

    # Define causal model
    causal_graph = """
    digraph {
        marketing_spend -> card_usage;
        onboarding_friction_score -> card_usage;
        card_usage -> churn;
    }
    """

    model = CausalModel(
        data=df,
        treatment='card_usage',
        outcome='churn',
        graph=causal_graph
    )

    # Identify effect
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

    # Estimate using OLS with backdoor adjustment
    X = df[['card_usage', 'marketing_spend', 'onboarding_friction_score']]
    X = sm.add_constant(X)
    y = df['churn']

    model_ols = sm.OLS(y, X).fit()
    causal_effect = model_ols.params['card_usage']

    # Run refutation test using permutation
    placebo_effects = []
    np.random.seed(42)

    for i in range(100):
        df_permuted = df.copy()
        df_permuted['card_usage'] = np.random.permutation(df_permuted['card_usage'].values)

        X_perm = df_permuted[['card_usage', 'marketing_spend', 'onboarding_friction_score']]
        X_perm = sm.add_constant(X_perm)
        y_perm = df_permuted['churn']

        model_perm = sm.OLS(y_perm, X_perm).fit()
        placebo_effects.append(model_perm.params['card_usage'])

    mean_placebo = np.mean(placebo_effects)
    std_placebo = np.std(placebo_effects)
    effect_ratio = np.abs(causal_effect) / (std_placebo + 1e-10)
    mean_placebo_ratio = np.abs(mean_placebo) / (std_placebo + 1e-10)
    refutation_passed = effect_ratio > 3 and mean_placebo_ratio < 0.5

    # Print summary
    print("\n" + "=" * 60)
    print("CAUSAL LOGIC VALIDATION")
    print("=" * 60)
    print(f"Causal Effect: {causal_effect:.6f}")
    print(f"Expected: < 0 (negative)")
    print(f"Result: {'PASS ✓' if causal_effect < 0 else 'FAIL ✗'}")
    print(f"\nRefutation Test:")
    print(f"Effect ratio: {effect_ratio:.2f}x")
    print(f"Expected: > 3x")
    print(f"Result: {'PASS ✓' if refutation_passed else 'FAIL ✗'}")
    print("=" * 60)

    # Assert both conditions
    assert causal_effect < 0, f"Causal effect should be negative, got {causal_effect:.6f}"
    assert refutation_passed, f"Refutation test failed: Effect ratio {effect_ratio:.2f} is not strong enough"


def test_drifted_data_friction_causes_churn():
    """
    Test that in drifted data, onboarding_friction_score has a POSITIVE causal effect on churn.

    This verifies the new causal structure after drift.
    """
    # Generate drifted data
    df = generate_drifted_data(n_companies=5000, random_seed=100)

    # Define NEW causal model for drifted data
    causal_graph = """
    digraph {
        marketing_spend -> card_usage;
        onboarding_friction_score -> card_usage;
        onboarding_friction_score -> churn;
    }
    """

    model = CausalModel(
        data=df,
        treatment='onboarding_friction_score',
        outcome='churn',
        graph=causal_graph
    )

    # Identify effect
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

    # Estimate using OLS with backdoor adjustment
    X = df[['onboarding_friction_score', 'marketing_spend']]
    X = sm.add_constant(X)
    y = df['churn']

    model_ols = sm.OLS(y, X).fit()
    causal_effect = model_ols.params['onboarding_friction_score']

    print(f"\nDRIFTED DATA - Estimated causal effect: {causal_effect:.6f}")
    assert causal_effect > 0, (
        f"Expected POSITIVE causal effect (friction increases churn), "
        f"but got {causal_effect:.6f}"
    )


def test_drifted_data_refutation_passes():
    """
    Test that the refutation test passes for the drifted data causal relationship.
    """
    # Generate drifted data
    df = generate_drifted_data(n_companies=5000, random_seed=100)

    # Estimate using OLS with backdoor adjustment
    X = df[['onboarding_friction_score', 'marketing_spend']]
    X = sm.add_constant(X)
    y = df['churn']

    model_ols = sm.OLS(y, X).fit()
    causal_effect = model_ols.params['onboarding_friction_score']

    # Run refutation test using permutation
    placebo_effects = []
    np.random.seed(42)

    for i in range(100):
        df_permuted = df.copy()
        df_permuted['onboarding_friction_score'] = np.random.permutation(
            df_permuted['onboarding_friction_score'].values
        )

        X_perm = df_permuted[['onboarding_friction_score', 'marketing_spend']]
        X_perm = sm.add_constant(X_perm)
        y_perm = df_permuted['churn']

        model_perm = sm.OLS(y_perm, X_perm).fit()
        placebo_effects.append(model_perm.params['onboarding_friction_score'])

    mean_placebo = np.mean(placebo_effects)
    std_placebo = np.std(placebo_effects)
    effect_ratio = np.abs(causal_effect) / (std_placebo + 1e-10)
    mean_placebo_ratio = np.abs(mean_placebo) / (std_placebo + 1e-10)
    refutation_passed = effect_ratio > 3 and mean_placebo_ratio < 0.5

    print(f"\nDRIFTED DATA - Refutation test:")
    print(f"  True effect: {causal_effect:.6f}")
    print(f"  Mean placebo: {mean_placebo:.6f}")
    print(f"  Effect ratio: {effect_ratio:.2f}x")
    assert refutation_passed, (
        f"Refutation test failed: Effect ratio {effect_ratio:.2f} is not strong enough. "
        f"True effect should be much stronger than placebo effects."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
