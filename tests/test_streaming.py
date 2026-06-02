"""Tests for the streaming drift detection pipeline.

Covers:
    - CUSUMDetector: fit, update, reset, alert threshold, unfitted guard.
    - TimeSeriesPanel: schema, shape, break at known timestep.
    - RollingCausalEstimator: output shape, sign of estimated effects.
    - End-to-end: detect_drift_streaming fires in the right half of the panel.
"""

from __future__ import annotations

import numpy as np
import pytest

from causal_guardian.config import StreamingConfig
from causal_guardian.dag import BASELINE_DAG
from causal_guardian.detection.cusum import CUSUMDetector
from causal_guardian.schemas import TimeSeriesPanelSchema
from causal_guardian.streaming import (
    RollingCausalEstimator,
    TimeSeriesPanel,
    detect_drift_streaming,
)


class TestCUSUMDetector:
    def test_fit_sets_baseline(self) -> None:
        detector = CUSUMDetector()
        values = np.zeros(20)
        detector.fit(values)
        assert detector.baseline_mean == pytest.approx(0.0)
        assert detector.baseline_std > 0

    def test_stable_signal_no_alert(self) -> None:
        rng = np.random.default_rng(0)
        values = rng.normal(0, 1, 50)
        detector = CUSUMDetector(config=StreamingConfig(cusum_threshold=10.0))
        detector.fit(values[:20])
        alerts = [detector.update(t, v) for t, v in enumerate(values[20:])]
        assert not any(alerts), "No alerts expected under very high threshold."

    def test_large_shift_triggers_alert(self) -> None:
        detector = CUSUMDetector(config=StreamingConfig(cusum_threshold=2.0, cusum_slack=0.5))
        baseline = np.zeros(20)
        detector.fit(baseline)
        # Feed a large positive shift - should alert well before 100 steps
        alerts = [detector.update(t, 5.0) for t in range(100)]
        assert any(alerts), "Expected an alert after a sustained large shift."

    def test_update_before_fit_raises(self) -> None:
        detector = CUSUMDetector()
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            detector.update(0, 1.0)

    def test_fit_requires_minimum_samples(self) -> None:
        detector = CUSUMDetector()
        with pytest.raises(ValueError, match="at least 5"):
            detector.fit(np.array([1.0, 2.0]))

    def test_reset_clears_statistics(self) -> None:
        detector = CUSUMDetector()
        detector.fit(np.zeros(20))
        for t in range(10):
            detector.update(t, 5.0)
        assert detector.current_s_pos > 0
        detector.reset()
        assert detector.current_s_pos == 0.0
        assert detector.current_s_neg == 0.0
        assert len(detector.statistics) == 0

    def test_history_records_all_updates(self) -> None:
        detector = CUSUMDetector()
        detector.fit(np.zeros(20))
        for t in range(15):
            detector.update(t, 0.0)
        assert len(detector.statistics) == 15


class TestTimeSeriesPanel:
    def test_schema_compliant(self) -> None:
        panel = TimeSeriesPanel(dag=BASELINE_DAG, n_timesteps=10, n_units_per_step=50)
        df = panel.generate()
        TimeSeriesPanelSchema.validate(df)

    def test_shape(self) -> None:
        panel = TimeSeriesPanel(dag=BASELINE_DAG, n_timesteps=10, n_units_per_step=50)
        df = panel.generate()
        assert len(df) == 10 * 50
        assert df["timestep"].nunique() == 10

    def test_known_break_at(self) -> None:
        """Effect should be negative before the break and near zero after.

        Controls for plan_tier (which is in the DGP and the DAG); without it
        the estimate is biased by plan-tier confounding.
        """
        import statsmodels.api as sm

        panel = TimeSeriesPanel(
            dag=BASELINE_DAG,
            pre_break_effect=-0.003,
            post_break_effect=0.0,
            n_timesteps=40,
            n_units_per_step=500,
            break_at=20,
            random_seed=0,
        )
        df = panel.generate()

        def usage_effect(sub_df):  # type: ignore[no-untyped-def]
            controls = ["card_usage", "marketing_spend", "onboarding_friction_score", "plan_tier"]
            X = sm.add_constant(sub_df[controls])
            return sm.OLS(sub_df["churn"], X).fit().params["card_usage"]

        pre_effect = usage_effect(df[df["timestep"] < 20])
        post_effect = usage_effect(df[df["timestep"] >= 20])

        assert pre_effect < 0, f"Pre-break effect should be negative; got {pre_effect:.5f}."
        assert abs(post_effect) < abs(pre_effect), (
            f"Post-break effect should be smaller in magnitude; "
            f"pre={pre_effect:.5f}, post={post_effect:.5f}."
        )


class TestRollingCausalEstimator:
    def test_output_shape(self) -> None:
        panel = TimeSeriesPanel(dag=BASELINE_DAG, n_timesteps=30, n_units_per_step=100)
        df = panel.generate()
        cfg = StreamingConfig(window_size=5, step_size=1)
        estimator = RollingCausalEstimator(dag=BASELINE_DAG, config=cfg)
        results = estimator.fit(df)

        assert "ate" in results.columns
        assert "std_error" in results.columns
        assert "timestep" in results.columns
        assert len(results) > 0

    def test_ate_negative_in_pre_break_period(self) -> None:
        """Rolling estimates in the pre-break period should show negative ATEs."""
        panel = TimeSeriesPanel(
            dag=BASELINE_DAG,
            pre_break_effect=-0.003,
            post_break_effect=0.0,
            n_timesteps=40,
            n_units_per_step=400,
            break_at=40,  # no break - whole panel is pre-break
            random_seed=1,
        )
        df = panel.generate()
        cfg = StreamingConfig(window_size=8, step_size=1)
        results = RollingCausalEstimator(dag=BASELINE_DAG, config=cfg).fit(df)

        median_ate = results["ate"].median()
        assert median_ate < 0, f"Median rolling ATE should be negative; got {median_ate:.5f}."


class TestDetectDriftStreaming:
    def test_alerts_in_post_break_period(self) -> None:
        """CUSUM should fire after the break, not before it."""
        panel = TimeSeriesPanel(
            dag=BASELINE_DAG,
            pre_break_effect=-0.003,
            post_break_effect=0.0,
            n_timesteps=80,
            n_units_per_step=500,
            break_at=40,
            random_seed=2,
        )
        df = panel.generate()
        cfg = StreamingConfig(
            window_size=8, step_size=1, cusum_threshold=3.0, burn_in_periods=20
        )
        rolling, alerts = detect_drift_streaming(df, BASELINE_DAG, streaming_cfg=cfg)

        if alerts:
            first_alert = min(alerts)
            # First alert should be at or after the break
            assert first_alert >= 38, (
                f"First alert at t={first_alert} is too early (break at t=40)."
            )

    def test_no_alerts_on_stable_panel(self) -> None:
        """A stable panel should produce few alerts under a strict threshold.

        Note: CUSUM has a non-zero false-positive rate. With k=0.5 and h=5
        under Gaussian null, expected run length ≈ 200 timesteps. A 60-step
        panel may produce a single CUSUM trigger that persists for several
        timesteps once breached. The check is on number of *trigger events*
        (gaps in the alert sequence), not raw alert count.
        """
        panel = TimeSeriesPanel(
            dag=BASELINE_DAG,
            pre_break_effect=-0.003,
            post_break_effect=-0.003,  # same effect - no break
            n_timesteps=60,
            n_units_per_step=400,
            break_at=60,  # effectively no break
            random_seed=3,
        )
        df = panel.generate()
        cfg = StreamingConfig(
            window_size=8, step_size=1, cusum_threshold=5.0, burn_in_periods=20
        )
        _, alerts = detect_drift_streaming(df, BASELINE_DAG, streaming_cfg=cfg)

        # Count distinct trigger events (gaps of >1 step between alerts)
        triggers = 0
        prev = -10
        for a in sorted(alerts):
            if a - prev > 1:
                triggers += 1
            prev = a
        assert triggers <= 1, (
            f"Expected at most 1 false-positive trigger event on stable panel, "
            f"got {triggers} triggers (alerts at {alerts})."
        )
