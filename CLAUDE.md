# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Vision

An agentic guardian for B2B Fintech causal discovery.

The Causal Guardian is a Python-based framework for discovering and validating causal relationships in business data. It uses the DoWhy library combined with statistical methods to identify true causal effects while guarding against spurious correlations and confounding variables.

## Development Commands

### Environment Setup
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Project

**Generate Synthetic Data:**
```bash
cd src
python generate_data.py
```

**Run Causal Discovery:**
```bash
cd src
python discovery.py
```

**Run Tests:**
```bash
pytest tests/test_causal_logic.py -v
```

**Run a Single Test:**
```bash
pytest tests/test_causal_logic.py::test_causal_effect_is_negative -v
```

### Drift Monitoring

**Generate Drifted Data:**
```bash
cd src
python drift_generator.py
```

**Run Drift Monitor:**
```bash
cd src
python monitor.py
```

The drift monitor:
- Establishes a baseline from original data
- Analyzes new/current data for causal relationships
- Detects drift via:
  - Refutation test failures (effect no longer distinguishable from noise)
  - Effect size changes > 50%
  - Effect sign flips
- Exits with error code 1 if drift detected

## Causal Graph

The current implementation models a B2B SaaS/Fintech scenario with the following Directed Acyclic Graph (DAG):

```
marketing_spend ──────┐
                      ├──> card_usage ──> churn
onboarding_friction ──┘
```

### Causal Structure:

1. **marketing_spend → card_usage**: Marketing campaigns increase product usage
2. **onboarding_friction_score → card_usage**: Friction during onboarding reduces usage
3. **card_usage → churn**: Higher usage reduces customer churn

### Key Insight:

Marketing spend has an **indirect** effect on churn through card_usage, but **no direct** causal path to churn. This demonstrates that simply spending more on marketing won't reduce churn unless it translates into actual product usage.

## Architecture

### Core Components

**src/generate_data.py**
- Generates synthetic data with a hidden causal structure
- Creates 5,000 companies with features: marketing_spend, onboarding_friction_score, card_usage, churn
- Implements the true causal relationships in the data generation process

**src/discovery.py**
- Uses DoWhy to define and identify causal models
- Estimates causal effects using OLS regression with backdoor adjustment
- Runs refutation tests using placebo treatment (permutation test)
- Validates that the true causal effect is significantly stronger than random effects

**src/drift_generator.py**
- Generates synthetic data with a DRIFTED causal structure
- Simulates a scenario where causal relationships have changed over time
- In the drifted data: onboarding_friction directly causes churn, card_usage effect is weakened
- Used to test the drift monitoring system

**src/monitor.py**
- Monitors for causal drift by comparing current data against baseline
- Detects refutation test failures, effect size changes > 50%, and sign flips
- Provides a `CausalDriftMonitor` class for integration into production systems
- Exits with error code 1 if drift is detected

**tests/test_causal_logic.py**
- Validates that the estimated causal effect is negative (usage reduces churn)
- Ensures refutation tests pass (effect ratio > 3x placebo effects)
- Provides regression testing for the causal discovery pipeline

### Statistical Approach

The project uses **backdoor adjustment** via OLS regression to control for confounders:
- Controls for both `marketing_spend` and `onboarding_friction_score`
- Estimates the direct effect of `card_usage` on `churn`
- Validates findings with permutation-based refutation tests

### Refutation Testing

Instead of traditional p-values, we use a more robust metric:
- Generate 100 placebo effects by permuting the treatment variable
- Compare the true effect magnitude to the distribution of placebo effects
- Pass criteria: True effect must be > 3x stronger than typical placebo effect

### Drift Detection

The Guardian monitors for three types of causal drift:

1. **Refutation Failure**: The causal effect becomes indistinguishable from random noise
   - Detected when effect ratio drops below 3x
   - Indicates the relationship may no longer be causal

2. **Effect Size Drift**: The magnitude of the causal effect changes significantly
   - Detected when effect size changes by > 50% from baseline
   - Indicates the strength of the relationship has changed

3. **Sign Flip**: The direction of the causal effect reverses
   - Detected when effect changes from positive to negative or vice versa
   - Indicates a fundamental change in the causal mechanism

**Use Case**: In production, run `monitor.py` periodically (daily/weekly) to ensure your causal models remain valid as business conditions evolve.

### Drift Recalibration Workflow

When drift is detected, the Guardian must be recalibrated to discover the new causal structure:

**Step 1: Detect Drift**
```bash
cd src
python monitor.py  # Exits with code 1 if drift detected
```

**Step 2: Generate Drifted Data (for testing)**
```bash
cd src
python drift_generator.py  # Creates drifted_data.csv
```

**Step 3: Update Causal Hypothesis**

Modify `src/discovery.py` to test new causal relationships:
- Update the DAG structure in the `causal_graph` variable
- Change the `treatment` variable to test different causal paths
- Adjust the `outcome` if needed
- Update confounders in the OLS regression

**Example: Testing if onboarding_friction causes churn directly**
```python
# Old DAG: card_usage -> churn
causal_graph = """
digraph {
    marketing_spend -> card_usage;
    onboarding_friction_score -> card_usage;
    card_usage -> churn;
}
"""

# New DAG: onboarding_friction -> churn
causal_graph = """
digraph {
    marketing_spend -> card_usage;
    onboarding_friction_score -> card_usage;
    onboarding_friction_score -> churn;
}
"""

# Update treatment
model = CausalModel(
    data=df,
    treatment='onboarding_friction_score',  # Changed from 'card_usage'
    outcome='churn',
    graph=causal_graph
)
```

**Step 4: Run Discovery on Drifted Data**
```bash
cd src
python discovery.py  # Now uses drifted_data.csv
```

**Step 5: Validate with Tests**
```bash
pytest tests/test_causal_logic.py::test_drifted_data_friction_causes_churn -v
pytest tests/test_causal_logic.py::test_drifted_data_refutation_passes -v
```

**Success Criteria**:
- Causal effect estimate has the expected sign
- Refutation test passes (effect ratio > 3x)
- All tests green

**Iteration**: If refutation tests fail, iterate on the DAG structure and try alternative causal hypotheses until refutation tests pass.

## Dependencies

- **dowhy**: Causal inference framework
- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **statsmodels**: OLS regression
- **pytest**: Testing framework
- **matplotlib**: Visualization
- **scipy**: Scientific computing
- **networkx**: Graph operations
- **pydot**: Graph visualization

## Development Notes

- The virtual environment (`venv/`) is required due to Python 3.14's externally-managed-environment restrictions
- All code uses absolute paths or changes directory before running
- Tests validate both the causal effect sign and the refutation test strength
