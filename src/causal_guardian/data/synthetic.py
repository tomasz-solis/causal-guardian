"""Synthetic data generator for the baseline (pre-drift) causal regime.

Causal structure:
    marketing_spend → card_usage
    onboarding_friction_score → card_usage
    card_usage → churn       (negative: more usage reduces churn)
    plan_tier → churn         (confounder: higher tier → lower churn)

marketing_spend has no direct effect on churn - only through card_usage.

The churn model uses a logistic (sigmoid) function rather than a linear
probability clipped to [0, 1]. This is more realistic and avoids the
boundary artefacts of the linear-clip approach.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit  # sigmoid

from causal_guardian.config import SyntheticDGPConfig


def generate_causal_data(
    n_companies: int = 5000,
    random_seed: int = 42,
    config: SyntheticDGPConfig | None = None,
) -> pd.DataFrame:
    """Generate synthetic company-level data under the baseline causal regime.

    Args:
        n_companies: Number of companies to simulate.
        random_seed: NumPy seed for reproducibility.
        config: Optional DGP config overriding per-coefficient values.
            If None, uses SyntheticDGPConfig defaults.

    Returns:
        DataFrame with columns: marketing_spend, onboarding_friction_score,
        card_usage, churn, plan_tier, cohort_month.

    Notes:
        ``plan_tier`` is an ordinal confounder (0=free, 1=starter,
        2=business, 3=enterprise). It affects both card_usage and churn
        but is not part of the primary causal pathway. It's included to
        demonstrate that naive regression without backdoor adjustment
        would be confounded.
    """
    cfg = config or SyntheticDGPConfig(n_companies=n_companies, random_seed=random_seed)
    rng = np.random.default_rng(cfg.random_seed)

    # Exogenous variables
    marketing_spend = rng.uniform(1_000, 10_000, cfg.n_companies)
    onboarding_friction_score = rng.uniform(0, 10, cfg.n_companies)

    # Plan tier: confounder affecting both usage and churn.
    # Higher-tier customers onboard more deliberately (lower friction impact)
    # and have stronger relationships with their CSMs (lower churn).
    plan_tier = rng.integers(0, 4, cfg.n_companies)  # 0 - 3

    # Cohort month (0 - 11): newer cohorts show slightly higher friction sensitivity
    cohort_month = rng.integers(0, 12, cfg.n_companies)

    # card_usage: depends on marketing, friction, and plan tier
    # Higher-tier plans tend to have dedicated onboarding → friction effect is dampened
    tier_usage_bonus = plan_tier * 8.0
    usage_noise = rng.normal(0, cfg.usage_noise_std, cfg.n_companies)
    card_usage = (
        cfg.marketing_to_usage * marketing_spend
        + cfg.friction_to_usage * onboarding_friction_score
        + tier_usage_bonus
        + cfg.usage_baseline
        + usage_noise
    )
    card_usage = np.maximum(card_usage, 0.0)

    # churn: logistic model on card_usage and plan_tier
    # Seasonality: companies signed in Q4 (months 9 - 11) churn slightly more
    # in the following Q1 (month_effect > 0 for high cohort_month)
    month_effect = 0.15 * np.sin(2 * np.pi * cohort_month / 12)
    logit_churn = (
        cfg.churn_intercept_logit
        + cfg.usage_to_churn_logit * card_usage
        + cfg.plan_to_churn_logit * plan_tier
        + month_effect
        + rng.normal(0, cfg.churn_noise_std, cfg.n_companies)
    )
    churn_prob = expit(logit_churn)
    churn = rng.binomial(1, churn_prob)

    return pd.DataFrame(
        {
            "marketing_spend": marketing_spend,
            "onboarding_friction_score": onboarding_friction_score,
            "card_usage": card_usage,
            "churn": churn,
            "plan_tier": plan_tier.astype(int),
            "cohort_month": cohort_month.astype(int),
        }
    )
