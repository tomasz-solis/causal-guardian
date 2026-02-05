# Causal Guardian: Post-Mortem Analysis

## Executive Summary

The Causal Guardian successfully detected and recalibrated after a fundamental shift in the causal mechanisms driving customer churn. This document provides a post-mortem analysis of the original hypothesis, the detected drift, and the recalibration process.

---

## Original Causal Hypothesis

### The DAG
```
marketing_spend ──────┐
                      ├──> card_usage ──> churn
onboarding_friction ──┘
```

### Causal Relationships
1. **marketing_spend → card_usage**: Marketing campaigns increase product usage
2. **onboarding_friction_score → card_usage**: Friction during onboarding reduces usage
3. **card_usage → churn**: Higher usage reduces customer churn (causal effect: -0.001 per unit)

### Key Business Insight
The original model showed that **card usage was the primary protective factor against churn**. Marketing spend and onboarding experience only mattered insofar as they influenced product usage. The strategy was clear: drive engagement, reduce churn.

### Validation
- **Causal Effect**: -0.001 (negative, as expected)
- **Refutation Test**: PASSED (effect 34x stronger than placebo)
- **Statistical Significance**: Strong evidence of causality

---

## Detected Drift

### What Changed?
The drift monitor detected that the original causal model was no longer valid:

**Drift Indicators**:
- Refutation test failure: The card_usage → churn effect became indistinguishable from random noise
- Effect size dropped below detection threshold
- New patterns emerged in the data

### The New Reality
```
marketing_spend ──────┐
                      ├──> card_usage
onboarding_friction ──┴──> churn
```

**Critical Change**:
- **card_usage -/-> churn**: Card usage **NO LONGER** reduces churn
- **onboarding_friction_score → churn**: Onboarding friction **NOW DIRECTLY** drives churn (causal effect: +0.078 per unit)

### Why This Matters
This represents a **fundamental shift in business dynamics**:
- **Before**: Engaged users stayed (usage was protective)
- **After**: Bad onboarding experiences drive churn regardless of usage

**Potential Real-World Causes**:
- Market saturation: Users now have alternatives, making friction intolerable
- Competitor improvements: Other products have better onboarding
- Changing user expectations: Users are less patient with friction
- Product commoditization: Usage no longer differentiates the product

---

## Recalibration Process

### Detection Phase
The Guardian's drift monitor flagged the breakdown of the original causal model:
```bash
$ cd src && python monitor.py
DRIFT DETECTED: Refutation test failure
Old model: card_usage → churn is no longer causal
```

### Hypothesis Generation
Given the drift signal, we formed a new hypothesis:
- **H0** (Null): No causal structure exists
- **H1** (Alternative): onboarding_friction_score directly causes churn

**Rationale**: If usage no longer protects against churn, and friction reduces usage, perhaps friction itself is now the culprit.

### Iterative DAG Testing

**Iteration 1: Test New Hypothesis**
```python
# Updated DAG
causal_graph = """
digraph {
    marketing_spend -> card_usage;
    onboarding_friction_score -> card_usage;
    onboarding_friction_score -> churn;  # NEW!
}
"""

# Changed treatment
treatment = 'onboarding_friction_score'  # Was 'card_usage'
outcome = 'churn'
```

**Iteration 2: Estimate & Validate**
```bash
$ cd src && python discovery.py
Estimated ATE: 0.078186
Effect ratio: 34.0x stronger than placebo
Refutation test: PASSED ✓
```

**Success**: The new hypothesis passed all validation criteria.

### Test Suite Updates
Added regression tests for the new causal structure:
- `test_drifted_data_friction_causes_churn`: Validates positive effect
- `test_drifted_data_refutation_passes`: Ensures statistical robustness

```bash
$ pytest tests/test_causal_logic.py -v
============================== 6 passed ==============================
```

---

## Results

### Validated New Causal Model

**Causal Effect**: onboarding_friction_score → churn = +0.078
- **Interpretation**: Each 1-point increase in onboarding friction increases churn probability by 7.8 percentage points
- **Statistical Strength**: 34x stronger than random noise
- **Validation**: Refutation tests passed

### Business Impact

**Old Strategy (No Longer Valid)**:
- Focus: Drive card usage at all costs
- Tactics: Marketing spend, engagement campaigns, feature releases
- KPI: Monthly Active Users, Transactions per User

**New Strategy (Evidence-Based)**:
- Focus: Eliminate onboarding friction
- Tactics: Streamline signup, reduce time-to-value, fix UX pain points
- KPI: Onboarding completion rate, Time-to-First-Transaction, Friction score

### ROI Implications

**Scenario**: Company has 10,000 customers with average onboarding friction score of 5.0

**If we reduce friction by 2 points (from 5.0 to 3.0)**:
- Churn reduction: 2 × 0.078 = **15.6 percentage points**
- Customers saved: 10,000 × 0.156 = **1,560 customers**
- Annual value (at $1,000 LTV): **$1.56M saved**

---

## Lessons Learned

### 1. Causal Models Drift
Business dynamics change. What was true last quarter may not be true today. Continuous monitoring is essential.

### 2. Refutation Tests Are Critical
Without rigorous refutation testing, we might have chased spurious correlations instead of true causal effects.

### 3. Agility in Causal Inference
The ability to rapidly test new causal hypotheses allowed us to recalibrate within a single iteration cycle.

### 4. Data > Intuition
The original strategy (drive usage) made intuitive sense, but the data showed a new reality. Evidence-based decision-making saved us from doubling down on an outdated model.

---

## Recommendations

### For Data Science Teams
1. **Implement continuous drift monitoring**: Run `monitor.py` daily/weekly
2. **Build rapid recalibration workflows**: Automate hypothesis testing
3. **Maintain test suites**: Ensure causal models remain valid over time

### For Product Teams
1. **Prioritize onboarding friction reduction** over usage growth in the short term
2. **Instrument friction metrics**: Track onboarding_friction_score in production
3. **Conduct UX audits**: Identify and eliminate friction points systematically

### For Leadership
1. **Shift KPIs**: From usage metrics to onboarding quality metrics
2. **Reallocate resources**: From growth marketing to onboarding optimization
3. **Communicate the shift**: Ensure all teams understand the new causal reality

---

## Conclusion

The Causal Guardian successfully detected a critical shift in business dynamics and provided actionable evidence for strategic pivots. By moving from "drive usage" to "eliminate friction," we can reduce churn by 15.6 percentage points—a potential $1.56M annual impact.

**The Guardian's value**: Not just in finding causal relationships, but in **knowing when they change**.

---

**Generated**: 2026-02-05
**Framework**: DoWhy + Backdoor Adjustment + Refutation Testing
**Methodology**: Iterative DAG Testing with Permutation-Based Validation
