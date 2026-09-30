# ADR 0001: DoWhy for identification, statsmodels for estimation

Status: accepted. Date: 2026-04.

## Context

Causal Guardian estimates average treatment effects from observational data. Two options:

- Option A: DoWhy end to end. `CausalModel.identify_effect()`, then `estimate_effect(method_name="backdoor.linear_regression")`. DoWhy calls statsmodels OLS internally and returns a `CausalEstimate`.
- Option B: DoWhy only for `identify_effect()`, to check the backdoor criterion against the DAG, then statsmodels OLS directly for the estimate and confidence intervals.

## Decision

Option B: DoWhy for identification and validation, statsmodels for estimation.

## Why

1. Reliable confidence intervals. DoWhy's `CausalEstimate.get_confidence_intervals()` changed between versions (0.9 to 0.11), and whether it works depends on the method and internal hooks. Statsmodels OLS intervals are stable, documented, and support HC3 errors if needed. The intervals go into the run artifact and the drift report, so they must be reliable.
2. Independent refutation. DoWhy runs the refutation suite (`refute_estimate` with several refuters). Keeping estimation in statsmodels lets us refute DoWhy's estimate and check that our OLS gives the same ATE.
3. Easier to inspect. `ols.summary()` gives a full regression table that is easy to log and audit. DoWhy's `CausalEstimate` is less transparent.
4. DoWhy's job is identification. The hard part of causal inference is knowing what to control for, which is what `identify_effect()` checks. Once the adjustment set is fixed, OLS is OLS whichever library runs it.

## Consequences

- `identify_effect()` runs on every estimation call to catch DAG mistakes early. That adds about 50ms on typical datasets, which is acceptable.
- The refutation suite calls `estimate_effect()` internally, because DoWhy needs an estimate to refute. With the same adjustment set it matches our OLS result exactly.
- If a future DoWhy drops or renames `identify_effect()`, only `estimation/backdoor.py:_check_identifiability()` changes. Estimation is unaffected.
