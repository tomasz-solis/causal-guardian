# Architecture

## Module layout

```
src/causal_guardian/
├── __init__.py           Public API
├── config.py             All numeric tunables (DriftConfig, StreamingConfig, DGP configs)
├── dag.py                CausalDAG dataclass + named DAGs (single source of truth)
├── schemas.py            Pandera schemas for DataFrame validation
│
├── data/
│   ├── synthetic.py      Baseline DGP (card_usage → churn regime)
│   └── drift.py          Drifted DGP (friction → churn regime)
│
├── estimation/
│   └── backdoor.py       DoWhy identification + statsmodels OLS + EffectEstimate
│
├── refutation.py         DoWhy refutation suite → RefutationReport
├── detection/
│   └── cusum.py          Two-sided CUSUM detector
│
├── streaming.py          TimeSeriesPanel, RollingCausalEstimator, detect_drift_streaming
├── monitor.py            CausalDriftMonitor (static two-dataset comparison)
└── runner.py             CLI entry point + JSON run artifact
```

## Data flow

```
[Data source]
      │
      ▼
CompanyChurnSchema.validate()      ← pandera, at every public boundary
      │
      ▼
CausalDAG (dag.py)                 ← single source of truth for graph structure
      │
      ├──► DoWhy CausalModel.identify_effect()   ← validates backdoor criterion
      │
      └──► statsmodels OLS                       ← estimation + CIs
                 │
                 ├──► EffectEstimate (ate, ci, se, n_obs)
                 │
                 └──► RefutationReport  (placebo_p, cc_delta, subset_delta, boot_se)
                             │
                             ▼
                       CausalDriftMonitor.check_drift()
                             │
                             ▼
                         DriftReport (severity, reasons, both estimates)
                             │
                             ▼
                      runner.py → JSON artifact → runs/<timestamp>.json
```

## Design decisions

Why statsmodels for estimation, not DoWhy's built-in estimator?
See `docs/adr/0001-dowhy-vs-statsmodels.md`.

Why a separate CUSUM module, not a threshold on the drift report?
The `CausalDriftMonitor` does a one-shot comparison of two datasets (baseline vs current). The CUSUM module handles the *streaming* case where you have a rolling sequence of estimates and want to detect a gradual shift. These are different statistical problems that warrant different tools.

Why frozen dataclasses for DAG and Config?
Immutability makes it safe to share these objects across tests and across threads. It also prevents accidental mutation mid-run, which in a monitoring context could produce silent inconsistencies between the baseline and current-period estimation.

Why pandera and not pydantic?
Pandera is purpose-built for DataFrame validation and supports statistical constraints (ge=, le=, isin=) at the column level. Pydantic is better suited for row-level Python objects. Both would work; pandera is less verbose for this use case.
