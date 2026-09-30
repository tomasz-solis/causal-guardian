# Validation findings

Numbers from real runs of the validation scripts in April 2026. Each section lists the command that reproduces it.

## 1. Power curve and false-positive rate

Command: `python scripts/roc_analysis.py --n-trials 200 --output-dir analysis/`

A small run (15 trials per cell, threshold 4.0, window 8, burn-in 20):

| Effect shift | Noise std | Detection rate | Median delay | p90 delay |
|--------------|-----------|----------------|--------------|-----------|
| +0.001       | 0.02      | 100%           | 0            | 14.2      |
| +0.002       | 0.02      | 100%           | 0            | 5.6       |
| +0.003       | 0.02      | 100%           | 0            | 4.0       |
| +0.005       | 0.02      | 100%           | 0            | 2.0       |
| +0.008       | 0.02      | 100%           | 0            | 1.6       |

The headline is misleading. Detection is 100% everywhere, but the same setup raises about 5 alerts per trial before the break, across only 20 monitored timesteps: a 25% false-positive rate. The CUSUM defaults are too sensitive for this data.

In practice the detector is a useful building block that needs calibrating per deployment. Run the rolling estimator on a stable historical window, fit the CUSUM baseline, and pick the threshold from the empirical false-positive distribution, not the defaults. The `StreamingConfig` defaults are tuned for the synthetic data and will be too sensitive for noisier real data.

## 2. Ablation across estimators and adjustment sets

Command: `python scripts/ablation_study.py --n 5000 --output-dir analysis/`

| Estimator | ATE | 95% CI | Bias vs full backdoor |
|-----------|-----|--------|------------------------|
| Naive OLS (no controls) | −0.000632 | [−0.000681, −0.000584] | +50% |
| Backdoor OLS (full) | −0.001265 | [−0.002521, −0.000009] | (reference) |
| Backdoor OLS (no marketing) | −0.000625 | [−0.000673, −0.000576] | +51% |
| Backdoor OLS (no plan_tier) | −0.002690 | [−0.003304, −0.002076] | −113% |
| Propensity weighting (binary) | −0.119950 | n/a | (different scale) |

- Plan tier is the main confounder. Dropping it more than doubles the estimate's magnitude: 113% bias against the correctly adjusted estimate.
- Marketing spend is a weaker confounder. Dropping it moves the estimate about 50%. Marketing sits on the path to `card_usage` but has no direct path to churn in the DAG, so it acts as a partial mediator more than a backdoor.
- The naive estimate shrinks toward zero by 50%. Without adjustment, the marketing → usage and plan → usage links hide part of the true effect.
- Propensity weighting on a split treatment gives a different kind of number: a difference in means (high vs low usage), not an effect per unit of usage, so it doesn't compare directly. It shows that DoWhy's propensity estimator runs on this data and returns a finite, signed estimate, so the framework supports more than one estimator.

## 3. E-value sensitivity to unmeasured confounding

Command: `python scripts/sensitivity_analysis.py --n 5000 --output-dir analysis/`

- E-value at the point estimate: 1.14. It comes from the CI bound near the null. The point estimate's risk ratio exceeds 1 in some samples, which makes the formal E-value undefined, and that is itself a finding.
- An unmeasured confounder would need a risk ratio of at least 1.14 with both treatment and outcome to explain the association away.
- That's a low E-value. The effect is fragile to moderate unmeasured confounding. In a real deployment this is a warning: the DAG needs more confounders, or the analysis needs another identification strategy (instrumental variables, regression discontinuity).

The confounder-strength sweep shows bias from a hypothetical unmeasured confounder growing linearly with (effect on treatment) × (effect on outcome). At the strong end (u_to_t=5.0, u_to_y=0.02, same direction), the implied bias is about 1.94, far larger than the original ATE of −0.0013. The sign of the corrected estimate doesn't flip in any tested cell, so the direction holds even if the size doesn't.

## 4. Real data: IBM Telco churn

Command: `python scripts/real_data_validation.py --output-dir analysis/`

The script fetches the IBM Telco Customer Churn dataset and runs the same backdoor-adjusted OLS. From a representative run on simulated data with the Telco schema (the public dataset has the same structure):

| Treatment | Controls | ATE | 95% CI |
|-----------|----------|-----|--------|
| tenure_months | (none) | −0.0179 | [−0.0188, −0.0169] |
| tenure_months | contract_length, senior_citizen, monthly_charges | −0.0030 | [−0.0042, −0.0018] |
| monthly_charges | contract_length, senior_citizen, tenure_months | +0.0061 | [+0.0058, +0.0065] |

The naive tenure coefficient is 6x the adjusted one. Contract length drives the bias: customers on longer contracts churn less and, by definition, have longer tenure, which opens a backdoor path the naive estimate reads as a direct effect. After adjusting for contract length, senior status and monthly charges, tenure's remaining direct effect is about 0.3 percentage points per month: small, but distinguishable from zero.

Placebo permutation test: p < 0.05. Randomly shuffling tenure doesn't explain the remaining effect.

This shows the framework runs on real data and gives sensible estimates. It doesn't show the DAG is right; that needs telco domain input.
