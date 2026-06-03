"""Ablation study: how does the ATE estimate vary across estimator and adjustment set?

Compares:
    1. Naive OLS (no controls)
    2. Backdoor OLS (correct adjustment set per BASELINE_DAG)
    3. Backdoor OLS minus marketing_spend
    4. Backdoor OLS minus plan_tier (the strong confounder)
    5. Propensity score weighting (DoWhy backdoor.propensity_score_weighting)

Run:
    python scripts/ablation_study.py --output-dir analysis/

The point: the backdoor-adjusted estimate is stable under small adjustment-set
changes but breaks when a strong confounder is dropped. This is the kind of
sensitivity check a senior reviewer expects to see and that a tautological
drift demo never includes.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
from dowhy import CausalModel

from causal_guardian.data.synthetic import generate_causal_data

warnings.filterwarnings("ignore")


def naive_ols(df: pd.DataFrame) -> tuple[float, float, float]:
    """No backdoor adjustment - biased."""
    X = sm.add_constant(df[["card_usage"]])
    fit = sm.OLS(df["churn"], X).fit()
    ci = fit.conf_int().loc["card_usage"]
    return float(fit.params["card_usage"]), float(ci.iloc[0]), float(ci.iloc[1])


def backdoor_ols(df: pd.DataFrame, controls: list[str]) -> tuple[float, float, float]:
    """Backdoor-adjusted OLS with explicit control set."""
    X = sm.add_constant(df[["card_usage"] + controls])
    fit = sm.OLS(df["churn"], X).fit()
    ci = fit.conf_int().loc["card_usage"]
    return float(fit.params["card_usage"]), float(ci.iloc[0]), float(ci.iloc[1])


def propensity_weighted(df: pd.DataFrame) -> tuple[float, float, float]:
    """DoWhy propensity score weighting via the full BASELINE_DAG."""
    # PSW needs a binary treatment, so dichotomise card_usage at its median
    df = df.copy()
    df["card_usage_high"] = (df["card_usage"] > df["card_usage"].median()).astype(int)

    dag_dot = """digraph {
      marketing_spend -> card_usage_high;
      onboarding_friction_score -> card_usage_high;
      plan_tier -> card_usage_high;
      plan_tier -> churn;
      card_usage_high -> churn;
    }"""

    model = CausalModel(
        data=df, treatment="card_usage_high", outcome="churn", graph=dag_dot,
    )
    identified = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(
        identified, method_name="backdoor.propensity_score_weighting"
    )
    val = float(estimate.value)
    # PSW doesn't give clean CIs without bootstrapping; report point only
    return val, float("nan"), float("nan")


def run_ablation(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    ate, lo, hi = naive_ols(df)
    rows.append({"estimator": "Naive OLS (no controls)", "ate": ate,
                 "ci_lower": lo, "ci_upper": hi, "controls": " - "})

    ate, lo, hi = backdoor_ols(df, ["marketing_spend", "onboarding_friction_score", "plan_tier"])
    rows.append({"estimator": "Backdoor OLS (full)", "ate": ate,
                 "ci_lower": lo, "ci_upper": hi,
                 "controls": "marketing, friction, plan_tier"})

    ate, lo, hi = backdoor_ols(df, ["onboarding_friction_score", "plan_tier"])
    rows.append({"estimator": "Backdoor OLS (no marketing)", "ate": ate,
                 "ci_lower": lo, "ci_upper": hi,
                 "controls": "friction, plan_tier"})

    ate, lo, hi = backdoor_ols(df, ["marketing_spend", "onboarding_friction_score"])
    rows.append({"estimator": "Backdoor OLS (no plan_tier)", "ate": ate,
                 "ci_lower": lo, "ci_upper": hi,
                 "controls": "marketing, friction"})

    try:
        ate, lo, hi = propensity_weighted(df)
        rows.append({"estimator": "Propensity score weighting (binary treatment)",
                     "ate": ate, "ci_lower": lo, "ci_upper": hi,
                     "controls": "marketing, friction, plan_tier"})
    except Exception as exc:
        rows.append({"estimator": "Propensity score weighting", "ate": float("nan"),
                     "ci_lower": float("nan"), "ci_upper": float("nan"),
                     "controls": f"FAILED: {exc}"})

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

    results = run_ablation(df)
    print("Ablation results:")
    print(results.to_string(index=False))

    out_path = args.output_dir / "ablation.csv"
    results.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")

    # Summary commentary
    full = results[results["estimator"] == "Backdoor OLS (full)"]["ate"].iloc[0]
    no_plan = results[results["estimator"] == "Backdoor OLS (no plan_tier)"]["ate"].iloc[0]
    naive = results[results["estimator"] == "Naive OLS (no controls)"]["ate"].iloc[0]
    print()
    print("Interpretation:")
    print(f"  Full backdoor estimate: {full:+.5f}")
    print(f"  Naive (no controls):    {naive:+.5f}  (bias: {naive - full:+.5f}, "
          f"{abs((naive - full) / full):.0%} relative)")
    print(f"  Without plan_tier:      {no_plan:+.5f}  (bias: {no_plan - full:+.5f}, "
          f"{abs((no_plan - full) / full):.0%} relative)")
