# Worked example: detecting causal drift on synthetic data

This document walks through what the detector does and what its output looks like, using entirely synthetic data. The scenario is illustrative - a fictional B2B fintech company where churn relationships change over time.

What this is not: a real business case. The coefficients, customer counts, and timelines are fabricated. The point is to show the methodology, not to claim a specific business result.

---

## The scenario

A fintech company (let's call it Pleo-like) tracks three signals:

- `card_usage`: how often B2B customers use the product.
- `onboarding_friction_score`: difficulty of the signup and onboarding flow (0=frictionless, 10=painful).
- `marketing_spend`: monthly marketing budget.

The data science team believes the causal structure is:

```
marketing_spend ──────┐
                      ├──> card_usage ──> churn
onboarding_friction ──┘
```

In words: marketing and onboarding experience determine how much customers use the product; usage then determines whether they churn. Marketing has no *direct* effect on churn - it only matters if it translates into usage.

The team has been optimising engagement. Churn stays high. Something changed.

---

## Historical baseline

On the historical dataset, the Guardian estimates:

```
ATE(card_usage → churn) = -0.00105 [95% CI: -0.00112, -0.00098]
```

Interpretation: each unit increase in `card_usage` reduces churn probability by ~0.1 percentage points. Refutation suite passes (placebo p=0.002, consistent=True). The causal claim holds in historical data.

---

## Current period: drift detected

When the detector is run on new data:

```
ATE(card_usage → churn) = -0.00003 [95% CI: -0.00011, +0.00005]
```

The effect has dropped by 97% and the CI now straddles zero. The refutation test fails (placebo p=0.38 - the true effect is indistinguishable from permuted noise). The drift report:

```json
{
  "drift_detected": true,
  "severity": "CRITICAL",
  "reasons": [
    "Effect no longer distinguishable from noise (ratio=1.2x, threshold=3.0x).",
    "Effect magnitude changed by 97.1% (baseline=-0.00105, current=-0.00003)."
  ]
}
```

---

## Discovery under the drift DAG

With card_usage no longer causal, the team tests an alternative hypothesis: friction now drives churn directly.

```
ATE(onboarding_friction → churn) = +0.078 [95% CI: +0.071, +0.085]
```

Refutation suite passes. The new causal structure is:

```
marketing_spend ──────┐
                      ├──> card_usage
onboarding_friction ──┴──> churn  [direct]
```

The causal mechanism shifted. In the old model, customers who used the product stayed. In the new model, customers who had a poor onboarding experience are leaving regardless of how much they use the product afterward.

---

## Streaming detection on a time-series panel

The static comparison above (dataset A vs dataset B) tells us that something changed, but not *when*. The streaming module generates a panel with a known break at timestep 40, then the CUSUM detector monitors rolling ATE estimates:

- Before t=40: mean ATE ≈ −0.002, CUSUM stable.
- After t=40: mean ATE drifts toward zero, CUSUM statistic rises, alert at t=47 (7-step lag at noise_std=0.30).

The detection delay of 7 timesteps is characterised across many noise levels in `analysis/power_curve.csv`.

---

## What this does not prove

- This is synthetic data. The DGP is designed so the drift is detectable. A real production dataset has unmeasured confounders, missing data, and distributional shifts in covariates that the DGP does not replicate.
- The 7-step detection lag on synthetic data is a lower bound on real-world lag, where noise is higher and data volume per period is lower.
- The dollar impact claims from earlier versions of this document have been removed. Those numbers came from multiplying a synthetic coefficient by a made-up customer count. That is not analysis.

For a discussion of when the detector fails and why, see [docs/methodology.md](methodology.md#4-assumptions-and-failure-modes).
