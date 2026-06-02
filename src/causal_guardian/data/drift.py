"""Synthetic data generator for the post-drift causal regime.

What changed relative to the baseline:
    - card_usage no longer affects churn at all.
    - onboarding_friction_score now has a direct positive effect on churn.

This represents a market shift: product usage no longer differentiates
retention, but the quality of the first experience does.

The upstream equations (marketing → usage, friction → usage) are unchanged,
so a naive monitoring approach that only tracks usage metrics would miss
the structural shift entirely.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit

from causal_guardian.config import DriftedDGPConfig


def generate_drifted_data(
    n_companies: int = 5000,
    random_seed: int = 100,
    config: DriftedDGPConfig | None = None,
) -> pd.DataFrame:
    """Generate synthetic company data under the post-drift causal regime.

    Args:
        n_companies: Number of companies to simulate.
        random_seed: NumPy seed for reproducibility.
        config: Optional DGP config. If None, uses DriftedDGPConfig defaults.

    Returns:
        DataFrame with the same schema as the baseline generator.
        The causal mechanism driving churn has changed, but column names
        and ranges are the same - a realistic scenario where the data
        surface looks stable while the underlying structure has shifted.
    """
    cfg = config or DriftedDGPConfig(n_companies=n_companies, random_seed=random_seed)
    rng = np.random.default_rng(cfg.random_seed)

    # Exogenous variables - same distributions as baseline
    marketing_spend = rng.uniform(1_000, 10_000, cfg.n_companies)
    onboarding_friction_score = rng.uniform(0, 10, cfg.n_companies)
    plan_tier = rng.integers(0, 4, cfg.n_companies)
    cohort_month = rng.integers(0, 12, cfg.n_companies)

    # card_usage: upstream structure unchanged
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

    # churn: friction now directly drives churn; card_usage has no effect
    month_effect = 0.15 * np.sin(2 * np.pi * cohort_month / 12)
    logit_churn = (
        cfg.churn_intercept_logit
        + cfg.friction_to_churn_logit * onboarding_friction_score
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
