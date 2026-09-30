# Example strategy brief

How the Guardian's output could feed a product strategy decision. It uses synthetic data, so every number is illustrative. A real version would replace the synthetic coefficients with estimates from production data, checked by whoever owns causal inference.

## Scenario

The Guardian found a structural shift: `card_usage` no longer has a causal effect on churn. Testing an alternative DAG found that `onboarding_friction_score` now has a direct positive effect: `ATE = +0.078 [95% CI: 0.071, 0.085]`.

## What ATE = +0.078 means

A 1-point increase in the friction score goes with a 7.8 percentage-point increase in churn probability, after controlling for marketing spend and product usage.

If you trust the causal claim (the refutation suite supports it but doesn't prove it), reducing friction has a higher expected return than driving engagement.

## What would have to be true before a strategic pivot

1. The friction score is measurable in production, not only in synthetic data.
2. The ATE is estimated on real observational data, not the data-generating process.
3. The DAG holds up against domain knowledge. Does friction plausibly reach churn directly, bypassing usage entirely? Are nodes missing (customer-success contacts, plan tier, competitor promotions)?
4. The effect is large relative to the cost of the intervention. +0.078 on synthetic data means nothing until you know the scale of real friction scores and the cost of lowering them.
5. The refutation suite passes on real data. Passing on synthetic data, built to be detectable, is necessary but a very weak test.

## How not to turn synthetic coefficients into a business case

Earlier drafts claimed dollar savings by multiplying the synthetic ATE by a synthetic customer count and a made-up ARPU. That isn't analysis; it's invented numbers formatted to look like it.

An honest impact estimate would:

1. Estimate the ATE on real data, with a proper CI.
2. Estimate how much a specific intervention changes the friction score (for example, removing two onboarding steps).
3. Estimate churn reduction as `friction_delta × ATE`, carrying the CI through.
4. Multiply by the real customer base and real ARPU.
5. Discount for intervention cost and uncertainty.

Every step adds uncertainty. A real business case shows a range, not a point, and says what would have to be true for the top of the range to happen.
