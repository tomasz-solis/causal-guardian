# Causal Guardian: drift detection and recalibration run
 
This documents one run of the framework on synthetic data. The data has known effects
injected, and the drift is engineered deliberately - the point is to check that the
monitor catches a relationship breaking and that recalibration recovers a model that
passes the refutation tests. Treat the numbers as a demonstration of the mechanism on
data where the true answer is known, not as a measured business result.
 
## Starting model
 
The initial DAG:
 
```
marketing_spend ──────┐
                      ├──> card_usage ──> churn
onboarding_friction ──┘
```
 
Three assumed relationships: marketing spend raises usage, onboarding friction lowers
usage, and usage lowers churn. The estimated usage → churn effect is small and
negative (about -0.001 per unit), consistent with the injected baseline, and it passes
the refutation test against the placebo distribution.
 
In this starting world, usage is the protective factor and the implied playbook is the
familiar one: drive engagement to hold down churn.
 
## Detected drift
 
After the drift event, `monitor.py` flags that the usage → churn effect no longer
passes refutation - the estimated effect becomes indistinguishable from the placebo
distribution. The monitor reports the breakdown rather than a clean new model; finding
the replacement is the next step, not an automatic one.
 
## Recalibration
 
Given the drift signal, the new hypothesis is that onboarding friction now drives churn
directly rather than only through usage. The updated DAG adds a direct edge:
 
```
marketing_spend ──────┐
                      ├──> card_usage
onboarding_friction ──┴──> churn
```
 
Re-running discovery with `onboarding_friction_score` as the treatment estimates a
positive effect of about +0.078 per friction point, which passes refutation at roughly
34x the placebo spread. Regression tests were added to lock in the new structure:
 
- `test_drifted_data_friction_causes_churn` - checks the positive effect holds
- `test_drifted_data_refutation_passes` - checks it stays above the noise threshold
```bash
$ pytest tests/test_causal_logic.py -v
6 passed
```
 
## What the run shows
 
The estimated coefficient says friction is now the dominant driver in this synthetic
world: usage has dropped out, and reducing friction is where the lever is. The exact
size, and any translation into churn-rate or revenue terms, is only meaningful here
because the effect was injected. On real data the same workflow would produce the
estimate; the difference is you would not know the answer in advance, and the
refutation and drift checks would be carrying real weight.
 
## What this run is meant to demonstrate
 
- Causal models drift. A relationship that held last quarter can stop holding, and
  metric dashboards won't surface it because they track the metrics, not the links.
- Refutation testing against a placebo distribution is what separates a supported effect
  from a spurious one - more useful here than a p-value.
- Recalibration can recover quickly when the monitor gives an early, specific signal
  about which relationship broke.
  
## Run metadata
 
Framework: DoWhy with backdoor adjustment, statsmodels OLS for estimation,
permutation-based refutation. The figures above come from a single synthetic run and
should be regenerated from `src/discovery.py` and `src/monitor.py` rather than quoted
from this file.
