# Causal Guardian

A small framework for estimating causal effects in business data and detecting when
those effects stop holding. The idea: most monitoring tracks metrics, not the
relationships between them. When the link between two variables breaks - a driver you
were optimizing for stops driving the outcome - metric dashboards don't catch it. This
catches it.

It's built around a B2B SaaS / fintech churn example, but the method applies anywhere
the question is "what actually moves this outcome" rather than "what correlates with
it."

## Decision Context

Suppose your working model is "more card usage lowers churn." You optimize for usage
and churn doesn't move. The usual dashboards can't tell you why, because they track
the metrics, not the causal link between them. If that link has weakened or reversed,
you keep spending against a model that no longer describes reality.

## What it does

1. Estimates a causal effect from a defined DAG using DoWhy with backdoor adjustment,
   with OLS for the estimation step.
2. Validates the effect against a placebo distribution - 100 permutations - and
   requires the true effect to be at least 3x the placebo spread. The point is
   signal-to-noise, not statistical significance: `p < 0.05` does not mean the effect
   passes the refutation checks.
3. Monitors for drift by re-testing new data against the baseline effect and flagging
   sign flips, effect-size changes above 50%, or refutation failures.
4. Re-searches alternative DAG structures when drift is detected, iterating until the
   refutation tests pass again.
## A worked example (synthetic)

The data here is synthetic, with known effects injected, so the pipeline can be checked
against ground truth. The numbers below are illustrative of the mechanism, not a
measured business result.

The starting model is `card_usage → churn`. After a drift event, `monitor.py` flags
that the usage-to-churn link no longer passes refutation. Re-searching the DAG points
instead to `onboarding_friction → churn`, with a positive effect (more friction, more
churn) that passes the placebo test. In the synthetic setup, reducing friction by two
points maps to roughly a 16% drop in churn - the kind of conclusion the framework is
meant to surface, on data where the true answer is known.

## Method, in short

- Discovery: define the DAG, control confounders via backdoor adjustment, estimate
  with OLS.
- Refutation: compare the true effect to a permutation-based placebo distribution;
  pass if it's 3x above noise.
- Drift: baseline the effects on historical data, re-test incoming data, flag
  refutation failures, large effect-size moves, or sign changes.
- Recalibration: test alternative DAGs until refutation passes, then validate against
  the regression test suite.
## Running it

```bash
git clone https://github.com/tomasz-solis/causal-guardian.git
cd causal-guardian
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cd src
python discovery.py     # estimate + refute
python monitor.py        # check new data for drift
pytest tests/test_causal_logic.py -v
```

## Limitations

- You define the initial DAG; this is not automated structure discovery.
- It assumes an acyclic graph - no feedback loops.
- It needs enough data for the effect estimates to be stable (the example uses 5,000
  rows).
- It assumes a working understanding of backdoor adjustment and causal inference. It
  is not AutoML.

## Documentation

- `SUMMARY.md` - technical write-up of the drift and recalibration run.
- `Product_Strategy_Brief.md` - the same result framed for a product audience.
## Credits

Built on DoWhy (identification and refutation), statsmodels (OLS estimation), and
pandas/NumPy. Methodology follows Pearl's framework, backdoor adjustment via
regression, and permutation-based testing.

## License

MIT - see [LICENSE](LICENSE).

## Contact

Tomasz Solis - [LinkedIn](https://www.linkedin.com/in/tomaszsolis/) · [GitHub](https://github.com/tomasz-solis)
