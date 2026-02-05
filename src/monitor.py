"""
Causal Drift Monitor

Monitors for changes in causal relationships over time by comparing
current data against baseline causal effects.

Detects:
1. Refutation test failures (effect no longer distinguishable from noise)
2. Effect size drift (> 50% change in causal effect magnitude)
"""

import sys
import pandas as pd
import numpy as np
from dowhy import CausalModel
import statsmodels.api as sm

from generate_data import generate_causal_data
from drift_generator import generate_drifted_data


class CausalDriftMonitor:
    """Monitor for detecting drift in causal relationships."""

    def __init__(self, baseline_effect: float = None, baseline_effect_ratio: float = None):
        """
        Initialize the monitor with baseline values.

        Args:
            baseline_effect: The baseline causal effect (from original data)
            baseline_effect_ratio: The baseline refutation test ratio
        """
        self.baseline_effect = baseline_effect
        self.baseline_effect_ratio = baseline_effect_ratio

    def estimate_causal_effect(self, df: pd.DataFrame, verbose: bool = True):
        """
        Estimate causal effect on given data.

        Returns:
            dict with 'causal_effect', 'refutation_passed', 'effect_ratio'
        """
        if verbose:
            print("\n" + "=" * 80)
            print("ANALYZING CAUSAL STRUCTURE")
            print("=" * 80)
            print(f"Dataset size: {len(df)} companies")
            print(f"Churn rate: {df['churn'].mean():.2%}")

        # Define the causal graph (assumes same structure)
        causal_graph = """
        digraph {
            marketing_spend -> card_usage;
            onboarding_friction_score -> card_usage;
            card_usage -> churn;
        }
        """

        model = CausalModel(
            data=df,
            treatment='card_usage',
            outcome='churn',
            graph=causal_graph
        )

        # Identify effect
        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

        # Estimate using OLS with backdoor adjustment
        X = df[['card_usage', 'marketing_spend', 'onboarding_friction_score']]
        X = sm.add_constant(X)
        y = df['churn']

        model_ols = sm.OLS(y, X).fit()
        causal_effect = model_ols.params['card_usage']

        if verbose:
            print(f"\nEstimated causal effect (card_usage -> churn): {causal_effect:.6f}")

        # Run refutation test using permutation
        if verbose:
            print("\nRunning refutation test (100 permutations)...")

        placebo_effects = []
        np.random.seed(42)

        for i in range(100):
            df_permuted = df.copy()
            df_permuted['card_usage'] = np.random.permutation(df_permuted['card_usage'].values)

            X_perm = df_permuted[['card_usage', 'marketing_spend', 'onboarding_friction_score']]
            X_perm = sm.add_constant(X_perm)
            y_perm = df_permuted['churn']

            model_perm = sm.OLS(y_perm, X_perm).fit()
            placebo_effects.append(model_perm.params['card_usage'])

        mean_placebo = np.mean(placebo_effects)
        std_placebo = np.std(placebo_effects)
        effect_ratio = np.abs(causal_effect) / (std_placebo + 1e-10)
        mean_placebo_ratio = np.abs(mean_placebo) / (std_placebo + 1e-10)

        refutation_passed = effect_ratio > 3 and mean_placebo_ratio < 0.5

        if verbose:
            print(f"Mean placebo effect: {mean_placebo:.6f}")
            print(f"Std placebo effect: {std_placebo:.6f}")
            print(f"Effect ratio: {effect_ratio:.2f}x")
            print(f"Refutation test: {'PASS ✓' if refutation_passed else 'FAIL ✗'}")

        return {
            'causal_effect': causal_effect,
            'refutation_passed': refutation_passed,
            'effect_ratio': effect_ratio,
            'mean_placebo': mean_placebo,
            'std_placebo': std_placebo
        }

    def check_drift(self, current_results: dict, verbose: bool = True):
        """
        Check if causal drift has occurred.

        Args:
            current_results: Results from estimate_causal_effect on current data

        Returns:
            dict with 'drift_detected', 'reasons', 'severity'
        """
        drift_reasons = []
        severity = "NONE"

        # Check 1: Refutation test failure
        if not current_results['refutation_passed']:
            drift_reasons.append(
                f"Refutation test FAILED (effect ratio: {current_results['effect_ratio']:.2f}x, "
                f"expected > 3x)"
            )
            severity = "CRITICAL"

        # Check 2: Effect size drift (> 50% change)
        if self.baseline_effect is not None:
            effect_change_pct = abs(
                (current_results['causal_effect'] - self.baseline_effect) / self.baseline_effect
            ) * 100

            if effect_change_pct > 50:
                drift_reasons.append(
                    f"Effect size changed by {effect_change_pct:.1f}% "
                    f"(baseline: {self.baseline_effect:.6f}, "
                    f"current: {current_results['causal_effect']:.6f})"
                )
                if severity == "NONE":
                    severity = "WARNING"

        # Check 3: Effect sign flip
        if self.baseline_effect is not None:
            if np.sign(self.baseline_effect) != np.sign(current_results['causal_effect']):
                drift_reasons.append(
                    f"Effect SIGN CHANGED (baseline: {self.baseline_effect:.6f}, "
                    f"current: {current_results['causal_effect']:.6f})"
                )
                severity = "CRITICAL"

        drift_detected = len(drift_reasons) > 0

        if verbose:
            print("\n" + "=" * 80)
            print("DRIFT DETECTION RESULTS")
            print("=" * 80)

            if drift_detected:
                print(f"🚨 CAUSAL DRIFT DETECTED - Severity: {severity}")
                print("\nReasons:")
                for i, reason in enumerate(drift_reasons, 1):
                    print(f"  {i}. {reason}")
            else:
                print("✓ No drift detected - Causal structure is stable")

            print("\nComparison:")
            if self.baseline_effect is not None:
                print(f"  Baseline effect: {self.baseline_effect:.6f}")
            print(f"  Current effect:  {current_results['causal_effect']:.6f}")
            if self.baseline_effect_ratio is not None:
                print(f"  Baseline effect ratio: {self.baseline_effect_ratio:.2f}x")
            print(f"  Current effect ratio:  {current_results['effect_ratio']:.2f}x")
            print("=" * 80)

        return {
            'drift_detected': drift_detected,
            'reasons': drift_reasons,
            'severity': severity
        }


def main():
    """Run drift monitoring comparing baseline to drifted data."""
    print("=" * 80)
    print("THE CAUSAL GUARDIAN - DRIFT MONITORING")
    print("=" * 80)

    # Step 1: Establish baseline on original data
    print("\n[1/3] Establishing baseline from original data...")
    baseline_df = generate_causal_data(n_companies=5000, random_seed=42)

    monitor = CausalDriftMonitor()
    baseline_results = monitor.estimate_causal_effect(baseline_df, verbose=True)

    # Store baseline values
    monitor.baseline_effect = baseline_results['causal_effect']
    monitor.baseline_effect_ratio = baseline_results['effect_ratio']

    print("\n✓ Baseline established")

    # Step 2: Analyze drifted data
    print("\n[2/3] Analyzing NEW data (potentially drifted)...")
    drifted_df = generate_drifted_data(n_companies=5000, random_seed=100)

    current_results = monitor.estimate_causal_effect(drifted_df, verbose=True)

    # Step 3: Check for drift
    print("\n[3/3] Checking for causal drift...")
    drift_report = monitor.check_drift(current_results, verbose=True)

    # Exit with error code if drift detected
    if drift_report['drift_detected']:
        print("\n❌ ALERT: The Guardian has detected causal drift!")
        print("   The causal relationships in your data have changed.")
        print("   Your existing models may no longer be valid.")
        sys.exit(1)
    else:
        print("\n✓ No drift detected - All systems nominal")
        sys.exit(0)


if __name__ == "__main__":
    main()
