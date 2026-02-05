"""
Causal Discovery using DoWhy library.

This script:
1. Loads synthetic data
2. Defines a causal model
3. Identifies the causal effect of card_usage on churn
4. Runs refutation tests to validate the findings
"""

import pandas as pd
import numpy as np
from dowhy import CausalModel
import matplotlib.pyplot as plt
import statsmodels.api as sm

from generate_data import generate_causal_data


def run_causal_analysis(use_drifted_data=False):
    """
    Run causal analysis to identify the effect of onboarding_friction_score on churn.

    Args:
        use_drifted_data: If True, load drifted data. If False, generate fresh data.
    """
    print("=" * 80)
    print("THE CAUSAL GUARDIAN - Causal Discovery Analysis")
    print("=" * 80)

    # Load data
    print("\n1. Loading data...")
    if use_drifted_data:
        df = pd.read_csv('drifted_data.csv')
        print(f"   Loaded DRIFTED data with {len(df)} companies")
    else:
        df = generate_causal_data(n_companies=5000, random_seed=42)
        print(f"   Generated {len(df)} companies")
    print(f"   Churn rate: {df['churn'].mean():.2%}")

    # Define the causal graph - NEW HYPOTHESIS: Friction -> Churn
    print("\n2. Defining causal model...")
    causal_graph = """
    digraph {
        marketing_spend -> card_usage;
        onboarding_friction_score -> card_usage;
        onboarding_friction_score -> churn;
    }
    """
    print("   Causal Graph:")
    print("   - marketing_spend -> card_usage")
    print("   - onboarding_friction_score -> card_usage")
    print("   - onboarding_friction_score -> churn [NEW!]")

    # Create the causal model
    model = CausalModel(
        data=df,
        treatment='onboarding_friction_score',
        outcome='churn',
        graph=causal_graph
    )

    print("\n3. Identifying causal effect...")
    # Identify the causal effect
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    print(identified_estimand)

    # Estimate the causal effect using statsmodels OLS
    print("\n4. Estimating causal effect using backdoor adjustment...")
    # Control for confounders: marketing_spend
    X = df[['onboarding_friction_score', 'marketing_spend']]
    X = sm.add_constant(X)  # Add intercept
    y = df['churn']

    model_ols = sm.OLS(y, X).fit()
    print(model_ols.summary())

    causal_effect = model_ols.params['onboarding_friction_score']
    print(f"\n   >>> Estimated ATE (Average Treatment Effect): {causal_effect:.6f}")
    print(f"   >>> Interpretation: Each unit increase in onboarding_friction_score changes churn by {causal_effect:.6f}")

    # Refutation test: Placebo treatment using permutation
    print("\n5. Running refutation test (Placebo Treatment)...")
    print("   Permuting onboarding_friction_score and re-estimating effect 100 times...")

    placebo_effects = []
    np.random.seed(42)

    for i in range(100):
        df_permuted = df.copy()
        df_permuted['onboarding_friction_score'] = np.random.permutation(df_permuted['onboarding_friction_score'].values)

        X_perm = df_permuted[['onboarding_friction_score', 'marketing_spend']]
        X_perm = sm.add_constant(X_perm)
        y_perm = df_permuted['churn']

        model_perm = sm.OLS(y_perm, X_perm).fit()
        placebo_effects.append(model_perm.params['onboarding_friction_score'])

    # Calculate refutation success: placebo effects should be near zero and much smaller than true effect
    mean_placebo = np.mean(placebo_effects)
    std_placebo = np.std(placebo_effects)

    # The refutation passes if:
    # 1. The true effect is much larger than placebo effects (measured in std devs)
    # 2. Placebo effects are centered near zero
    effect_ratio = np.abs(causal_effect) / (std_placebo + 1e-10)  # Avoid division by zero
    mean_placebo_ratio = np.abs(mean_placebo) / (std_placebo + 1e-10)

    refutation_passed = effect_ratio > 3 and mean_placebo_ratio < 0.5

    print(f"\n   >>> Mean placebo effect: {mean_placebo:.6f}")
    print(f"   >>> Std placebo effect: {std_placebo:.6f}")
    print(f"   >>> True effect: {causal_effect:.6f}")
    print(f"   >>> Effect magnitude (in std devs): {effect_ratio:.2f}")

    if refutation_passed:
        print("   >>> PASS: True effect is much stronger than placebo effects")
    else:
        print("   >>> FAIL: True effect is similar to placebo effects")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Causal Effect (onboarding_friction_score -> churn): {causal_effect:.6f}")
    print(f"Expected: Positive (more friction increases churn)")
    print(f"Actual: {'POSITIVE ✓' if causal_effect > 0 else 'NEGATIVE ✗'}")
    print(f"\nRefutation Test: {'PASS ✓' if refutation_passed else 'FAIL ✗'}")
    print(f"True effect is {effect_ratio:.1f}x stronger than placebo effects")
    print("=" * 80)

    return {
        'estimate': causal_effect,
        'refutation_passed': refutation_passed,
        'effect_ratio': effect_ratio,
        'model': model,
        'identified_estimand': identified_estimand,
        'ols_model': model_ols
    }


if __name__ == "__main__":
    results = run_causal_analysis(use_drifted_data=True)
