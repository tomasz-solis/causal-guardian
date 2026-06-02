"""Causal Guardian - causal drift detection for B2B SaaS product analytics.

Core public API:
    CausalDriftMonitor - detect drift in a causal relationship.
    CausalDAG, BASELINE_DAG, DRIFT_DAG - graph specifications.
    DriftConfig, StreamingConfig - tunables.
    estimate_ate - backdoor-adjusted effect estimation.
    run_refutation_suite - DoWhy refutation suite.
    detect_drift_streaming - rolling CUSUM-based monitoring.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def _configure_matplotlib_cache() -> None:
    """Point Matplotlib at a writable cache directory before DoWhy imports it."""
    cache_dir = Path(tempfile.gettempdir()) / "causal-guardian-matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))


_configure_matplotlib_cache()

from causal_guardian.config import DriftConfig, StreamingConfig
from causal_guardian.dag import BASELINE_DAG, DRIFT_DAG, CausalDAG
from causal_guardian.estimation.backdoor import EffectEstimate, estimate_ate
from causal_guardian.monitor import CausalDriftMonitor, DriftReport, Severity
from causal_guardian.refutation import RefutationReport, run_refutation_suite
from causal_guardian.streaming import (
    RollingCausalEstimator,
    TimeSeriesPanel,
    detect_drift_streaming,
)

__all__ = [
    "BASELINE_DAG",
    "CausalDAG",
    "CausalDriftMonitor",
    "DRIFT_DAG",
    "DriftConfig",
    "DriftReport",
    "EffectEstimate",
    "RefutationReport",
    "RollingCausalEstimator",
    "Severity",
    "StreamingConfig",
    "TimeSeriesPanel",
    "detect_drift_streaming",
    "estimate_ate",
    "run_refutation_suite",
]
