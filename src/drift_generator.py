"""
Generate synthetic data with DRIFTED causal structure.

This represents a scenario where the causal relationships have changed over time.

NEW Causal Structure (DRIFT):
- marketing_spend -> card_usage (positive effect) [SAME]
- onboarding_friction_score -> card_usage (negative effect) [SAME]
- onboarding_friction_score -> churn (positive direct effect) [NEW!]
- card_usage -/-> churn (NO effect) [CHANGED!]

The key drift: onboarding_friction now DIRECTLY causes churn,
and card_usage NO LONGER affects churn at all.
"""

import numpy as np
import pandas as pd


def generate_drifted_data(n_companies: int = 5000, random_seed: int = 100) -> pd.DataFrame:
    """
    Generate synthetic data with a DRIFTED causal structure.

    In this scenario, the business has changed:
    - Product usage (card_usage) no longer reduces churn
    - Onboarding friction now directly drives churn
    - This could happen if competitors improved, or if the product lost its edge

    Args:
        n_companies: Number of companies to generate
        random_seed: Random seed for reproducibility

    Returns:
        DataFrame with columns: marketing_spend, onboarding_friction_score, card_usage, churn
    """
    np.random.seed(random_seed)

    # Exogenous variables
    marketing_spend = np.random.uniform(1000, 10000, n_companies)
    onboarding_friction_score = np.random.uniform(0, 10, n_companies)

    # Endogenous variable: card_usage
    # Still influenced by marketing_spend (positive) and onboarding_friction (negative)
    card_usage_noise = np.random.normal(0, 5, n_companies)
    card_usage = (
        0.05 * marketing_spend  # Marketing increases usage
        - 3.0 * onboarding_friction_score  # Friction decreases usage
        + 50  # Baseline usage
        + card_usage_noise
    )
    card_usage = np.maximum(card_usage, 0)  # Can't be negative

    # Endogenous variable: churn (DRIFTED RELATIONSHIPS)
    # NOW: Onboarding friction DIRECTLY causes churn
    # AND: Card usage has NO effect on churn
    churn_noise = np.random.normal(0, 0.05, n_companies)
    churn_probability = (
        0.3  # Lower baseline churn
        + 0.08 * onboarding_friction_score  # Friction DIRECTLY increases churn (NEW!)
        # REMOVED: card_usage no longer affects churn at all
        + churn_noise
    )
    churn_probability = np.clip(churn_probability, 0, 1)
    churn = (np.random.random(n_companies) < churn_probability).astype(int)

    # Create DataFrame
    df = pd.DataFrame({
        'marketing_spend': marketing_spend,
        'onboarding_friction_score': onboarding_friction_score,
        'card_usage': card_usage,
        'churn': churn
    })

    return df


if __name__ == "__main__":
    df = generate_drifted_data()

    # Save to CSV
    df.to_csv('drifted_data.csv', index=False)
    print("Saved drifted data to 'drifted_data.csv'")

    print("\nGenerated DRIFTED dataset:")
    print(df.head(10))
    print(f"\nDataset shape: {df.shape}")
    print(f"\nChurn rate: {df['churn'].mean():.2%}")
    print(f"\nDescriptive statistics:")
    print(df.describe())

    print("\n" + "=" * 80)
    print("CAUSAL DRIFT SUMMARY")
    print("=" * 80)
    print("NEW Causal Structure:")
    print("  - onboarding_friction_score -> churn (DIRECT, STRONG) [NEW!]")
    print("  - card_usage -/-> churn (NO EFFECT) [CHANGED FROM STRONG!]")
    print("  - marketing_spend -> card_usage (same)")
    print("  - onboarding_friction_score -> card_usage (same)")
    print("\nIMPLICATION:")
    print("  The old model assumed card_usage protects against churn.")
    print("  In this NEW reality, card_usage has NO effect on churn.")
    print("  Onboarding friction is now the dominant driver of churn.")
    print("  The Guardian should detect this drift!")
    print("=" * 80)
