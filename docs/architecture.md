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

| Question | Answer |
|---|---|
| Why statsmodels for estimation, not DoWhy's estimator? | See `docs/adr/0001-dowhy-vs-statsmodels.md`. |
| Why a separate CUSUM module, not a threshold on the drift report? | `CausalDriftMonitor` compares two datasets once (baseline vs current). The CUSUM module handles the streaming case: a rolling sequence of estimates where you want to catch a gradual shift. Different problems, different tools. |
| Why frozen dataclasses for the DAG and config? | Immutable objects are safe to share across tests and threads, and can't be changed mid-run, which in monitoring could quietly make the baseline and current estimates inconsistent. |
| Why pandera, not pydantic? | Pandera is built for DataFrame validation and supports column-level statistical constraints (`ge=`, `le=`, `isin=`). Pydantic suits row-level Python objects. Both would work; pandera is less verbose here. |
