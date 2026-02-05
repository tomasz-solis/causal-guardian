"""
Generate synthetic Pleo-like data with hidden causal structure.

Causal Structure:
- marketing_spend -> card_usage (positive effect)
- onboarding_friction_score -> card_usage (negative effect)
- card_usage -> churn (negative effect, i.e., more usage reduces churn)
- marketing_spend -/-> churn (no direct effect)
"""

import numpy as np
import pandas as pd


def generate_causal_data(n_companies: int = 5000, random_seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic data for companies with a hidden causal structure.

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
    # Influenced by marketing_spend (positive) and onboarding_friction (negative)
    card_usage_noise = np.random.normal(0, 5, n_companies)
    card_usage = (
        0.05 * marketing_spend  # Marketing increases usage
        - 3.0 * onboarding_friction_score  # Friction decreases usage
        + 50  # Baseline usage
        + card_usage_noise
    )
    card_usage = np.maximum(card_usage, 0)  # Can't be negative

    # Endogenous variable: churn
    # Influenced ONLY by card_usage (negative effect)
    # Marketing has NO direct effect on churn
    churn_noise = np.random.normal(0, 0.05, n_companies)
    churn_probability = (
        0.8  # High baseline churn
        - 0.005 * card_usage  # More usage reduces churn
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
    df = generate_causal_data()
    print("Generated dataset:")
    print(df.head(10))
    print(f"\nDataset shape: {df.shape}")
    print(f"\nChurn rate: {df['churn'].mean():.2%}")
    print(f"\nDescriptive statistics:")
    print(df.describe())
