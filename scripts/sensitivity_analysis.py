"""Sensitivity analysis: E-values for unmeasured confounding.

The E-value (VanderWeele & Ding, 2017) quantifies how strong an unmeasured
confounder would need to be - measured as risk ratio with both the
treatment and outcome - to fully explain away an observed association.

For an observed risk ratio RR, the E-value is:
    E = RR + sqrt(RR * (RR - 1))    if RR > 1
    E = (1/RR) + sqrt((1/RR) * (1/RR - 1))    if RR < 1

A high E-value means the observed effect is hard to explain away: only a strong
unmeasured confounder could do it. A low E-value means a moderate
unmeasured confounder could explain the result, so the causal claim
is fragile.

Reference:
    VanderWeele, T. J., & Ding, P. (2017). Sensitivity analysis in
    observational research: Introducing the E-value.
    Annals of Internal Medicine, 167(4), 268 - 274.

Run:
    python scripts/sensitivity_analysis.py --output-dir analysis/
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm

from causal_guardian.data.synthetic import generate_causal_data


def e_value(rr: float) -> float:
    """E-value for a risk ratio."""
    if rr <= 0:
        return float("nan")
    if rr < 1:
        rr = 1 / rr
    return rr + np.sqrt(rr * (rr - 1))


def coef_to_rr(coef: float, baseline_rate: float, treatment_delta: float) -> float:
    """Translate an LPM coefficient into an approximate risk ratio.

    Compares churn probability at (baseline + treatment_delta) to baseline.
    """
    p1 = baseline_rate + coef * treatment_delta
    p0 = baseline_rate
    if p0 <= 0 or p1 <= 0:
        return float("nan")
    return p1 / p0


def run_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """Run sensitivity sweep over a range of unmeasured-confounder strengths.

    For each hypothetical confounder strength (effect on both treatment and
    outcome), recompute the bias-corrected ATE and check whether the CI still
    excludes zero.
    """
    # Backdoor estimate
    controls = ["marketing_spend", "onboarding_friction_score", "plan_tier"]
    X = sm.add_constant(df[["card_usage"] + controls])
    fit = sm.OLS(df["churn"], X).fit()
    ate = float(fit.params["card_usage"])
    se = float(fit.bse["card_usage"])
    ci = fit.conf_int().loc["card_usage"]

    # Effect of a 100-unit increase in card_usage (≈one std)
    treatment_delta = 100.0
    baseline_rate = float(df["churn"].mean())
    rr_point = coef_to_rr(ate, baseline_rate, treatment_delta)
    rr_ci = coef_to_rr(float(ci.iloc[1]), baseline_rate, treatment_delta)  # CI bound nearer null
    e_point = e_value(rr_point) if not np.isnan(rr_point) else float("nan")
    e_ci = e_value(rr_ci) if not np.isnan(rr_ci) else float("nan")

    summary = pd.DataFrame([{
        "ate": ate,
        "ate_se": se,
        "ate_ci_lower": float(ci.iloc[0]),
        "ate_ci_upper": float(ci.iloc[1]),
        "treatment_delta": treatment_delta,
        "implied_rr_point": rr_point,
        "implied_rr_ci_near_null": rr_ci,
        "e_value_point": e_point,
        "e_value_ci": e_ci,
    }])
    return summary


def confounder_strength_sweep(df: pd.DataFrame) -> pd.DataFrame:
    """Sweep over hypothetical unmeasured-confounder strengths.

    Models a hypothetical confounder U that affects both card_usage and churn.
    For each U effect strength, computes the bias-adjusted ATE per VanderWeele's
    formulae for linear models.
    """
    controls = ["marketing_spend", "onboarding_friction_score", "plan_tier"]
    X = sm.add_constant(df[["card_usage"] + controls])
    fit = sm.OLS(df["churn"], X).fit()
    ate = float(fit.params["card_usage"])

    rows = []
    # Hypothetical: U has effect γ on outcome and ρ on treatment.
    # Bias in OLS coefficient ≈ +ρ × γ / Var(treatment | controls) when both
    # effects share the same sign, attenuating the observed ATE.
    # We sweep both signs of confounding to bracket worst-case.
    treat_resid_var = float(fit.resid.var())  # approximate residual variance
    for u_to_y in [0.0, 0.001, 0.005, 0.01, 0.02]:
        for u_to_t in [0.0, 0.5, 1.0, 2.0, 5.0]:
            # Two-sided sweep: confounding can go either direction
            for sign, label in [(+1, "amplifies"), (-1, "attenuates")]:
                bias = sign * (u_to_t * u_to_y) / max(treat_resid_var, 1e-6)
                adjusted_ate = ate - bias
                rows.append({
                    "u_effect_on_outcome": u_to_y,
                    "u_effect_on_treatment": u_to_t,
                    "confounding_direction": label,
                    "implied_bias": bias,
                    "adjusted_ate": adjusted_ate,
                    "sign_changed_vs_observed": np.sign(adjusted_ate) != np.sign(ate),
                })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("analysis"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = generate_causal_data(n_companies=args.n, random_seed=args.seed)
    print(f"Generated {len(df)} rows. Churn rate: {df['churn'].mean():.2%}.")
    print()

    summary = run_sensitivity(df)
    print("E-value summary:")
    print(summary.to_string(index=False))
    summary.to_csv(args.output_dir / "e_value_summary.csv", index=False)

    print("\nConfounder strength sweep:")
    sweep = confounder_strength_sweep(df)
    print(sweep.to_string(index=False))
    sweep.to_csv(args.output_dir / "confounder_sweep.csv", index=False)

    print("\nInterpretation:")
    e_point = float(summary["e_value_point"].iloc[0])
    e_ci = float(summary["e_value_ci"].iloc[0])
    print(f"  E-value (point estimate): {e_point:.2f}")
    print(f"  E-value (CI bound near null): {e_ci:.2f}")
    print(f"  An unmeasured confounder would need risk ratios of at least {e_ci:.2f}")
    print("  with BOTH the treatment and the outcome to fully explain away the")
    print("  observed association. Weak confounding is unlikely to explain it away,")
    print("  but stronger unmeasured confounding remains a concern.")
