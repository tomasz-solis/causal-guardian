# Example: what a strategy brief might look like

This is a worked example of how the Guardian's output could feed into a product strategy decision. It is written against synthetic data and all numbers are illustrative. A real version of this document would replace the synthetic coefficients with estimates from production data, validated by the causal inference team.

---

## Scenario

The Guardian detected a structural shift: `card_usage` no longer has a causal effect on churn. A test of the alternative DAG found that `onboarding_friction_score` now has a direct positive effect: `ATE = +0.078 [95% CI: 0.071, 0.085]`.

---

## What "ATE = +0.078" means operationally

A 1-point increase in the friction score corresponds to a 7.8 percentage-point increase in churn probability, after controlling for marketing spend and product usage.

If you trust the causal claim (and the refutation suite provides supporting evidence, not proof), then reducing friction has a higher expected return than driving engagement.

---

## What would need to be true for this to justify a strategic pivot

1. The friction score must be measurable in production, not just synthetic.
2. The ATE must be estimated on real observational data, not the DGP.
3. The causal DAG must be validated against domain knowledge - does it make sense that friction has a direct path to churn that bypasses usage entirely? Are there missing nodes (e.g., customer-success interactions, plan tier, competitor promotions)?
4. The effect size must be large enough relative to the cost of the intervention. An ATE of +0.078 in synthetic data means nothing until you know the scale of real friction scores and the cost of reducing them.
5. The refutation suite must pass on real data. Passing on synthetic data - where the DGP is built to be detectable - is a necessary but very weak condition.

---

## How not to turn synthetic coefficients into a business case

Earlier drafts of this project made a dollar-savings claim by multiplying the synthetic ATE by a synthetic customer count and a made-up ARPU. That is not analysis; it is made-up numbers formatted to look like analysis.

If you wanted to produce an honest business impact estimate, the process would be:

1. Estimate the ATE on real data with a proper CI.
2. Estimate the expected change in the friction score from a specific intervention (e.g., removing two onboarding steps).
3. Estimate churn reduction as `friction_delta × ATE`, with CI propagation.
4. Multiply churn reduction by actual customer base and actual ARPU.
5. Discount by intervention cost and uncertainty.

Each step introduces uncertainty. A real business case would show a range, not a point estimate, and would acknowledge what would have to be true for the high end of the range to be achievable.
