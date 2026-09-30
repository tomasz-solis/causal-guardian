# Causal Guardian

Estimates causal effects in business data and flags when those effects stop holding. Most monitoring tracks metrics, not the relationships between them. When a driver you optimise for stops driving the outcome, dashboards miss it. This catches it.

The example is B2B SaaS / fintech churn, but the method works anywhere the question is "what actually moves this outcome", not "what correlates with it".

## The decision

Say your working model is "more card usage lowers churn". You push usage and churn doesn't move. Dashboards can't say why, because they track the metrics, not the causal link. If that link has weakened or reversed, you keep spending against a model that no longer describes reality.

## What it does

1. Estimates a causal effect from a defined DAG: DoWhy checks the backdoor adjustment, statsmodels OLS does the estimation.
2. Tests the effect against a placebo distribution (permuted treatment, 200 permutations by default) and requires it to be at least 3x the placebo spread. This is signal to noise, not significance: `p < 0.05` doesn't mean the effect passes refutation.
3. Checks for drift by re-estimating on new data against the baseline and flagging refutation failures, effect-size changes above 50%, and sign flips.
4. Tracks rolling estimates over time with a CUSUM detector, to see when a shift happened.

When drift is flagged, you test alternative DAGs yourself. `dag.py` holds the baseline and drift DAGs used in the example.

## Worked example (synthetic)

The data is synthetic with known effects injected, so the pipeline can be checked against the true answer. The numbers show the mechanism, not a business result.

The starting model is `card_usage → churn`. After a drift event, the monitor flags that the usage-to-churn link no longer passes refutation. Testing an alternative DAG points to `onboarding_friction → churn` instead: more friction, more churn, and the effect passes the placebo test. In this setup, the effect is +0.078 per friction point, so cutting friction by two points lowers churn probability by about 16 percentage points.

Full walkthrough: [docs/worked-example.md](docs/worked-example.md).

## Running it

Requires Python 3.10 to 3.12.

```bash
git clone https://github.com/tomasz-solis/causal-guardian.git
cd causal-guardian
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"

python -m causal_guardian.runner             # baseline vs drifted data, writes runs/<timestamp>.json
python -m causal_guardian.runner --no-drift  # control run, same regime both periods
pytest
```

`--fail-on-drift` exits with status 1 when drift is found, for CI or alerting jobs.

Validation scripts (power curve, ablation, E-values, real data) are in `scripts/`. Results are in [docs/validation-findings.md](docs/validation-findings.md).

## Limitations

- You define the starting DAG. This isn't automated structure discovery.
- The graph must be acyclic, so no feedback loops.
- Estimates need enough data to be stable (the example uses 5,000 rows).
- You need a working grasp of backdoor adjustment and causal inference. It isn't AutoML.

## Docs

| Doc | Contents |
|---|---|
| [docs/SUMMARY.md](docs/SUMMARY.md) | Technical write-up of the drift and recalibration run |
| [docs/product-strategy-brief.md](docs/product-strategy-brief.md) | The same result framed for a product audience |
| [docs/worked-example.md](docs/worked-example.md) | Step-by-step example with outputs |
| [docs/methodology.md](docs/methodology.md) | Backdoor criterion, refutation tests, CUSUM, assumptions |
| [docs/validation-findings.md](docs/validation-findings.md) | Numbers from the validation scripts |
| [docs/architecture.md](docs/architecture.md) | Module layout and data flow |
| [docs/adr/0001-dowhy-vs-statsmodels.md](docs/adr/0001-dowhy-vs-statsmodels.md) | Why DoWhy for identification and statsmodels for estimation |

## Credits

Built on DoWhy (identification and refutation), statsmodels (OLS) and pandas/NumPy. The method follows Pearl's framework, backdoor adjustment through regression and permutation testing.

## License

MIT, see [LICENSE](LICENSE).

## Contact

Tomasz Solis · [LinkedIn](https://www.linkedin.com/in/tomaszsolis/) · [GitHub](https://github.com/tomasz-solis)
