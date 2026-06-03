# Methodology

This document covers the statistical machinery behind Causal Guardian: the backdoor criterion, what the refutation tests actually test, the CUSUM detector, and the assumptions that have to hold for each piece to be valid.

---

## 1. Causal Identification via Backdoor Adjustment

### Identification Question

We observe a dataset with treatment T, outcome Y, and a set of variables V. The question is: what is the Average Treatment Effect (ATE) of T on Y - i.e., what would happen to Y if we *intervened* to set T to a particular value, rather than just observing it?

Naive regression of Y on T is biased if there are confounders: variables that influence both T and Y. In the card-usage scenario, `onboarding_friction_score` influences both `card_usage` (friction reduces usage) and `churn` (in the drifted regime, directly). A naive regression of `churn` on `card_usage` would partially attribute the friction-driven churn to usage.

### The backdoor criterion (Pearl, 2009)

A set of variables Z satisfies the backdoor criterion relative to (T, Y) in DAG G if:

1. No element of Z is a descendant of T.
2. Z blocks all *backdoor paths* from T to Y - i.e., all paths that enter T via an arrow pointing *into* T.

If such a Z exists, the ATE is identifiable and equals:

```
ATE = E[Y | do(T=t)] = Σ_z E[Y | T=t, Z=z] P(Z=z)
```

For a linear model this simplifies to the OLS coefficient on T in a regression that includes Z as covariates.

### Application

Baseline DAG: T = `card_usage`, Y = `churn`.
Backdoor paths from `card_usage` to `churn` pass through `marketing_spend` and `onboarding_friction_score` (both arrow-into T). Controlling for both blocks all backdoor paths.

Drift DAG: T = `onboarding_friction_score`, Y = `churn`.
The only backdoor path runs through `marketing_spend` → `card_usage` (arrow-into `onboarding_friction_score`? No - `marketing_spend` does not cause `friction`). In this DAG, there are no backdoor paths from friction to churn, so the ATE is identified without adjustment. We nonetheless include `marketing_spend` as a control to reduce variance.

DoWhy's `identify_effect()` verifies this identification before estimation proceeds.

### Estimation

Given a valid adjustment set Z, we estimate ATE via OLS:

```
Y = α + β·T + γ·Z + ε
```

β is the estimated ATE. Confidence intervals come from the OLS standard errors. HC3 heteroskedasticity-consistent standard errors are available via `statsmodels` and would be preferable for binary outcomes; the current implementation uses standard OLS CIs as a simplification.

Limitation: churn is binary (0/1). OLS on a binary outcome is the Linear Probability Model (LPM). LPM is unbiased for ATE under the assumptions above but can produce fitted probabilities outside [0,1] and is less efficient than logistic regression. For small effects and moderate churn rates, the bias is negligible. For a production deployment, I would use DoWhy's `backdoor.propensity_score_weighting` estimator with a logistic propensity model, or an augmented inverse-propensity estimator.

---

## 2. Refutation Tests

### Purpose

Causal identification rests on the DAG being correct. If the assumed DAG omits an important confounding path, the estimate is biased even with correct backdoor adjustment. Refutation tests stress-test the estimate from multiple angles.

### Placebo treatment refuter

What it does: permutes the treatment variable (T ← shuffle(T)) and re-estimates the effect N times.

Null hypothesis: the observed effect is no larger than what you'd see with a random treatment assignment.

Interpretation: the fraction of permuted effects with |effect| ≥ |true effect| is the p-value. A small p-value (< 0.05) means the true effect is in the tail of the placebo distribution - supports causality.

Failure mode: with large N and any real signal, this will almost always pass. It doesn't protect against a correctly-estimated but causally misinterpreted effect.

### Random common cause refuter

What it does: adds a synthetic random noise variable as a common cause of T and Y, then re-estimates.

Logic: if the estimate is genuinely causal (not spurious), adding irrelevant noise shouldn't change it much. A large shift suggests the estimate was sensitive to unmeasured confounding.

Decision rule: |new_effect − original| / |original| < 10%.

### Data subset refuter

What it does: re-estimates on random 80% subsets N times.

Logic: the estimate should be stable across subsets. Instability suggests the result is driven by a particular slice of data.

Decision rule: mean(|new_effect − original| / |original|) < 10%.

### Bootstrap standard error

Samples the dataset with replacement N times and estimates the ATE on each bootstrap sample. The standard deviation of bootstrap estimates is the bootstrap SE, giving a non-parametric uncertainty measure that doesn't rely on OLS normality assumptions.

---

## 3. CUSUM Drift Detection

### The drift detection problem

We observe a rolling sequence of ATE estimates, one per time window. Under the null (no drift), these are approximately i.i.d. from some distribution with mean μ₀ and standard deviation σ₀. We want to detect when the mean shifts.

### CUSUM statistic (Page 1954)

Normalise observations: Z_t = (X_t − μ₀) / σ₀.

The one-sided upper CUSUM statistic is:

```
S_t⁺ = max(0, S_{t-1}⁺ + Z_t − k)
```

where k is the *slack* parameter - typically 0.5 if you want to detect a shift of 1σ.

The lower CUSUM monitors downward shifts:

```
S_t⁻ = max(0, S_{t-1}⁻ − Z_t − k)
```

An alert fires when S_t⁺ > h or S_t⁻ > h, where h is the *threshold* (typically 4 - 5 for a ~5% false-positive rate per 1000 observations under Gaussian null).

### Calibration

μ₀ and σ₀ are estimated from a *burn-in period* of stable windows before monitoring begins. The threshold h trades sensitivity for specificity:

| Threshold h | Approx. FPR (Gaussian, 1000 steps) |
|-------------|-------------------------------------|
| 3.0 | ~10% |
| 4.0 | ~3% |
| 5.0 | ~1% |

See `analysis/false_positive_sweep.csv` (generated by `scripts/roc_analysis.py`) for empirical FPR rates on this dataset.

### Detection delay

Detection delay is the number of timesteps between the true break and the first CUSUM alert. It depends on:

- Effect-size delta (larger shift → shorter delay).
- Noise level (more noise → longer delay).
- Window size (larger window → more stable estimates, shorter burn-in bias, but misses fast shifts).

See `analysis/power_curve.csv` for the empirical power curve.

---

## 4. Assumptions and Failure Modes

Assumption 1: The DAG is correct.
The whole framework rests on the user-specified DAG correctly representing the causal structure. If a back-door path is missing from the graph, the ATE estimate is biased. The refutation tests provide partial protection but cannot detect a consistently-wrong DAG.

Assumption 2: No feedback loops.
The backdoor criterion requires an acyclic graph. Feedback (e.g., churn → friction, if churned users leave negative reviews affecting onboarding) violates the DAG assumption and requires dynamic causal inference.

Assumption 3: Sufficient statistical power.
At small effect sizes and high noise, the rolling estimator produces estimates whose variance is too large for the CUSUM to reliably distinguish signal from noise. The power curve (see `scripts/roc_analysis.py`) characterises the boundary.

Assumption 4: Covariate shift is handled separately.
The detector monitors the *causal effect coefficient*, not the feature distribution. If marketing_spend distribution shifts, the OLS coefficient may remain stable even though the causal mechanism is intact. Separate monitoring of covariate distributions (e.g., with a KS test) is recommended alongside this detector.

Assumption 5: Linearity.
The LPM estimator assumes a linear relationship between treatment and outcome probability. This is a reasonable approximation for the regime this project targets (moderate churn rates, moderate effect sizes) but breaks down at extreme probabilities.
