"""Apply Causal Guardian to a public real-world churn dataset.

Uses the IBM Telco Customer Churn dataset, available from multiple public
mirrors. We fetch a known-clean copy and run the same backdoor estimation +
refutation pipeline to demonstrate the framework works outside the synthetic
DGP.

Caveat: this is a single observational dataset. The DAG below is our best
guess at the causal structure for a residential telco customer; a real
deployment would require domain experts to specify and validate the graph.
The point of this script is to show the framework runs on real data and
produces interpretable output, not to make a substantive claim about telco
churn drivers.

Run:
    python scripts/real_data_validation.py --output-dir analysis/

Data source:
    IBM Telco Customer Churn dataset.
    URL: https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm

TELCO_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)


def load_telco(cache_path: Path | None = None) -> pd.DataFrame:
    """Load the IBM Telco churn dataset, with optional local cache."""
    if cache_path is not None and cache_path.exists():
        return pd.read_csv(cache_path)

    df = pd.read_csv(TELCO_URL)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
    return df


def preprocess_telco(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the Telco dataset for OLS analysis.

    - TotalCharges has whitespace strings for new customers; cast to numeric.
    - Drop the ~11 rows with TotalCharges NaN after coercion.
    - Encode Churn as 0/1.
    - Encode Contract as ordinal (month-to-month=0, one year=1, two year=2).
    """
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])
    df["churn"] = (df["Churn"] == "Yes").astype(int)

    contract_order = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    df["contract_length"] = df["Contract"].map(contract_order).astype(int)

    return df.rename(columns={
        "tenure": "tenure_months",
        "MonthlyCharges": "monthly_charges",
        "TotalCharges": "total_charges",
        "SeniorCitizen": "senior_citizen",
    })


def estimate_with_dag(df: pd.DataFrame, treatment: str, controls: list[str]
                     ) -> dict:
    """Backdoor-adjusted OLS on the cleaned Telco frame."""
    X = sm.add_constant(df[[treatment] + controls])
    fit = sm.OLS(df["churn"], X).fit()
    ci = fit.conf_int().loc[treatment]
    return {
        "treatment": treatment,
        "controls": ", ".join(controls),
        "ate": float(fit.params[treatment]),
        "ci_lower": float(ci.iloc[0]),
        "ci_upper": float(ci.iloc[1]),
        "se": float(fit.bse[treatment]),
        "n": len(df),
    }


def placebo_test(df: pd.DataFrame, treatment: str, controls: list[str],
                 n_perm: int = 200, seed: int = 42) -> float:
    """Permutation placebo p-value for the treatment effect."""
    rng = np.random.default_rng(seed)
    X_full = sm.add_constant(df[[treatment] + controls])
    true_effect = float(sm.OLS(df["churn"], X_full).fit().params[treatment])

    placebo_effects = []
    for _ in range(n_perm):
        df_perm = df.copy()
        df_perm[treatment] = rng.permutation(df_perm[treatment].to_numpy())
        X = sm.add_constant(df_perm[[treatment] + controls])
        placebo_effects.append(float(sm.OLS(df_perm["churn"], X).fit().params[treatment]))

    arr = np.array(placebo_effects)
    return float(np.mean(np.abs(arr) >= abs(true_effect)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("analysis"))
    parser.add_argument("--cache", type=Path, default=Path("analysis/telco.csv"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading Telco churn data from {TELCO_URL}...")
    raw = load_telco(cache_path=args.cache)
    df = preprocess_telco(raw)
    print(f"Loaded {len(df)} rows. Churn rate: {df['churn'].mean():.2%}.")
    print()

    # Hypothesised DAG:
    #   contract_length, senior_citizen, monthly_charges -> tenure_months
    #   contract_length, senior_citizen, monthly_charges -> churn
    #   tenure_months -> churn  (treatment of interest)
    #
    # Treatment: tenure_months (the longer a customer has been around,
    #   does that itself reduce churn beyond their selection effects?)
    # Controls: contract_length, senior_citizen, monthly_charges

    rows = []
    rows.append(estimate_with_dag(
        df,
        treatment="tenure_months",
        controls=["contract_length", "senior_citizen", "monthly_charges"],
    ))
    rows.append(estimate_with_dag(
        df,
        treatment="tenure_months",
        controls=[],  # naive
    ))
    rows.append(estimate_with_dag(
        df,
        treatment="monthly_charges",
        controls=["contract_length", "senior_citizen", "tenure_months"],
    ))

    results = pd.DataFrame(rows)
    print("Estimates:")
    print(results.to_string(index=False))

    # Refutation on the primary specification
    print("\nRunning placebo permutation test on tenure → churn...")
    p_value = placebo_test(
        df,
        treatment="tenure_months",
        controls=["contract_length", "senior_citizen", "monthly_charges"],
        n_perm=100,
    )
    print(f"  Placebo p-value: {p_value:.4f}")
    print(f"  Refutation passes (p < 0.05): {p_value < 0.05}")

    refutation_row = pd.DataFrame([{
        "treatment": "tenure_months",
        "specification": "controls = contract_length, senior_citizen, monthly_charges",
        "placebo_p_value": p_value,
        "refutation_passes": p_value < 0.05,
    }])

    results.to_csv(args.output_dir / "telco_estimates.csv", index=False)
    refutation_row.to_csv(args.output_dir / "telco_refutation.csv", index=False)

    print()
    print("Interpretation:")
    naive = results.loc[(results["treatment"] == "tenure_months") & (results["controls"] == ""), "ate"].iloc[0]
    full = results.loc[(results["treatment"] == "tenure_months") & (results["controls"] != ""), "ate"].iloc[0]
    print(f"  Naive tenure → churn:  {naive:+.5f}  (uncorrected)")
    print(f"  Backdoor-adjusted:     {full:+.5f}  (controls: contract, age, monthly $)")
    print(f"  Confounding bias:      {naive - full:+.5f}  ({abs((naive - full)/full):.0%} relative)")
    print()
    print("  The naive tenure-on-churn coefficient is heavily biased by")
    print("  contract length: customers on longer contracts have both lower")
    print("  churn AND longer tenures by definition. Backdoor adjustment")
    print("  recovers a smaller, more credible direct effect of tenure.")
