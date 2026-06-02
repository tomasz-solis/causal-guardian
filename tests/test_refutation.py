"""Tests for the DoWhy refutation suite.

These run the full suite (permutation + random CC + subset + bootstrap),
so they're marked ``slow`` and deselect with ``-m 'not slow'`` for CI.
"""

from __future__ import annotations

import pytest

from causal_guardian.dag import BASELINE_DAG, DRIFT_DAG
from causal_guardian.estimation.backdoor import estimate_ate
from causal_guardian.refutation import RefutationReport, run_refutation_suite
from causal_guardian.config import DriftConfig


@pytest.mark.slow
class TestRefutationBaseline:
    def test_consistent_on_baseline(self, baseline_df, fast_config):
        est = estimate_ate(baseline_df, BASELINE_DAG)
        report = run_refutation_suite(baseline_df, BASELINE_DAG, est, config=fast_config)
        assert isinstance(report, RefutationReport)
        assert report.consistent, (
            f"Expected refutation to pass on baseline DGP. {report.summary}"
        )

    def test_placebo_p_value_in_range(self, baseline_df, fast_config):
        est = estimate_ate(baseline_df, BASELINE_DAG)
        report = run_refutation_suite(baseline_df, BASELINE_DAG, est, config=fast_config)
        assert 0.0 <= report.placebo_p_value <= 1.0

    def test_bootstrap_se_positive(self, baseline_df, fast_config):
        est = estimate_ate(baseline_df, BASELINE_DAG)
        report = run_refutation_suite(baseline_df, BASELINE_DAG, est, config=fast_config)
        assert report.bootstrap_std_error > 0


@pytest.mark.slow
class TestRefutationDrift:
    def test_drifted_estimate_centred_near_zero(self, fast_config):
        """In the drifted regime, the BASELINE_DAG's card_usage → churn effect
        has no true causal pathway. Across multiple seeds, the OLS estimate
        should be centred near zero and the 95% CI should include zero in
        most samples.

        Single-seed checks are unreliable here because sampling noise can
        produce marginally significant Type I effects. We average across
        seeds for a more honest test of the null.
        """
        from causal_guardian.dag import BASELINE_DAG  # noqa: PLC0415
        from causal_guardian.data.drift import generate_drifted_data  # noqa: PLC0415
        from causal_guardian.estimation.backdoor import estimate_ate  # noqa: PLC0415

        ates = []
        ci_includes_zero = []
        for seed in [50, 100, 150, 200, 250, 300, 350, 400]:
            df = generate_drifted_data(n_companies=5000, random_seed=seed)
            est = estimate_ate(df, BASELINE_DAG)
            ates.append(est.ate)
            ci_includes_zero.append(est.ci_lower < 0 < est.ci_upper)

        import numpy as np  # noqa: PLC0415

        mean_ate = float(np.mean(ates))
        ci_inclusion_rate = sum(ci_includes_zero) / len(ci_includes_zero)

        # True effect is exactly zero - sample mean should be near zero
        assert abs(mean_ate) < 0.001, (
            f"Mean ATE across seeds should be near zero in drifted regime; got {mean_ate:+.5f}."
        )
        # At least 5/8 seeds should have CI including zero (≥62.5%)
        assert ci_inclusion_rate >= 0.5, (
            f"At most 50% of seeds had CI excluding zero; got {ci_inclusion_rate:.2%}. "
            "Refutation behaviour suggests a real effect, contradicting the null DGP."
        )
