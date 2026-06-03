# ADR 0001: DoWhy for identification, statsmodels for estimation

Status: accepted
Date: 2026-04

---

## Context

Causal Guardian needs to estimate Average Treatment Effects from observational data. Two approaches were considered:

Option A: Use DoWhy for the full path: `CausalModel.identify_effect()` then `estimate_effect(method_name="backdoor.linear_regression")`. DoWhy internally calls statsmodels OLS and returns a `CausalEstimate` object.

Option B: Use DoWhy only for `identify_effect()` to validate the backdoor criterion against the DAG, then run statsmodels OLS directly for estimation and confidence intervals.

---

## Decision

We use Option B: DoWhy for identification and validation, statsmodels for estimation.

---

## Reasoning

1. Confidence interval reliability. DoWhy's `CausalEstimate.get_confidence_intervals()` API has changed across versions (0.9 → 0.11), and its availability depends on the estimation method and internal hooks. Statsmodels OLS CIs are stable, well-documented, and directly support HC3 heteroskedasticity-consistent standard errors if needed. We need CIs to be reliable because they appear in the run artifact and inform the drift report.

2. Refutation independence. We use DoWhy for the refutation suite (`refute_estimate` with multiple refuters). Keeping estimation in statsmodels means we can refute the DoWhy estimate and check that our statsmodels OLS gives the same ATE.

3. Inspectability. `ols.summary()` from statsmodels gives a full regression table that is straightforward to log and audit. DoWhy's `CausalEstimate` object is less transparent.

4. DoWhy's role is identification, not estimation. The hard part of causal inference is knowing what to control for. That is what `identify_effect()` does. Once the adjustment set is established, OLS is OLS regardless of which library runs it. DoWhy does not add correctness to the OLS step; it validates the identification.

---

## Consequences

- We run `identify_effect()` on every estimation call to catch DAG misspecifications early. This adds a small overhead (~50ms on typical datasets) that is acceptable.
- The refutation suite calls `estimate_effect()` internally (DoWhy needs an estimate to refute). The DoWhy internal estimate will be numerically identical to our OLS result when both use the same adjustment set.
- If DoWhy drops or renames `identify_effect()` in a future version, only `estimation/backdoor.py:_check_identifiability()` needs updating. The estimation path is unaffected.
