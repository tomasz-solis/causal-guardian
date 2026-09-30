# Causal Guardian: drift detection and recalibration run

One run of the framework on synthetic data. The effects are injected and the drift is engineered, to check that the monitor catches a relationship breaking and that recalibration finds a model that passes refutation. The numbers show the mechanism on data where the true answer is known. They aren't a business result.

## Starting model

```text
marketing_spend ──────┐
                      ├──> card_usage ──> churn
onboarding_friction ──┘
```

Three assumed relationships: marketing spend raises usage, onboarding friction lowers usage, and usage lowers churn. The estimated usage-to-churn effect is small and negative (about -0.001 per unit), consistent with the injected baseline, and passes refutation against the placebo distribution.

In this world usage protects against churn, and the playbook is the familiar one: drive engagement.

## Drift detected

After the drift event, the monitor flags that the usage-to-churn effect no longer passes refutation: the estimate can't be told apart from the placebo distribution. The monitor reports the breakdown, not a new model. Finding the replacement is a separate step.

## Recalibration

The new hypothesis: onboarding friction now drives churn directly, not only through usage. The updated DAG adds a direct edge:

```text
marketing_spend ──────┐
                      ├──> card_usage
onboarding_friction ──┴──> churn
```

With `onboarding_friction_score` as the treatment, the estimated effect is about +0.078 per friction point, passing refutation at roughly 34x the placebo spread. Tests lock in the new structure (`test_friction_drives_churn_positively` in `tests/test_data_generation.py` and `test_friction_effect_is_positive` in `tests/test_estimation.py`).

## What the run shows

Friction is now the main driver in this synthetic world: usage has dropped out and friction is where the lever is. The exact size, and any translation into churn rates or revenue, only means something here because the effect was injected. On real data the same workflow produces the estimate, but you don't know the answer in advance, and the refutation and drift checks carry real weight.

The points it makes:

- Causal models drift. A relationship that held last quarter can stop, and metric dashboards won't show it because they track metrics, not links.
- Refutation against a placebo distribution separates a supported effect from a spurious one better than a p-value does.
- Recalibration is fast when the monitor says early and specifically which relationship broke.

## Run details

DoWhy with backdoor adjustment, statsmodels OLS for estimation, permutation-based refutation. The figures come from one synthetic run. Regenerate them with `python -m causal_guardian.runner` instead of quoting this file.
