"""End-to-end runner for drift monitoring.

Orchestrates: data loading → baseline estimation → current-period estimation
→ drift check → structured JSON artifact.

Usage:
    python -m causal_guardian.runner \\
        --baseline-seed 42 \\
        --current-seed 100 \\
        --use-drifted \\
        --output-dir runs/

The run artifact is written to ``output_dir/<timestamp>.json`` and contains
enough information to reproduce the run: the config used, the effect estimates
with CIs, and the drift verdict.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from causal_guardian.config import DriftConfig
from causal_guardian.dag import BASELINE_DAG
from causal_guardian.data.drift import generate_drifted_data
from causal_guardian.data.synthetic import generate_causal_data
from causal_guardian.monitor import CausalDriftMonitor

logger = logging.getLogger(__name__)


def _configure_cli_logging(verbose: bool) -> None:
    """Configure CLI logging without overwhelming the default output.

    DoWhy first tries optional pygraphviz parsing, logs that miss as an
    error, then successfully falls back to pydot. The fallback is expected
    in this project because pydot is the supported graph parser dependency.
    """
    logging.basicConfig(
        level=logging.INFO if verbose else logging.ERROR,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )
    logging.getLogger("dowhy").setLevel(logging.WARNING)
    logging.getLogger("dowhy.causal_graph").setLevel(logging.CRITICAL)


def run(
    baseline_seed: int = 42,
    current_seed: int = 100,
    n_companies: int = 5000,
    use_drifted: bool = True,
    output_dir: Path | None = None,
    config: DriftConfig | None = None,
) -> dict[str, Any]:
    """Run a full baseline → check cycle and return the result as a dict.

    Args:
        baseline_seed: Random seed for the baseline dataset.
        current_seed: Random seed for the current dataset.
        n_companies: Companies per dataset.
        use_drifted: If True, current dataset uses the drifted DGP.
            If False, uses the same baseline DGP (should show no drift).
        output_dir: If provided, write a JSON artifact to this directory.
        config: DriftConfig overrides. Uses defaults if None.

    Returns:
        Dict with keys: config, baseline, current, drift_report, metadata.
    """
    cfg = config or DriftConfig()
    logger.info("Generating baseline data (seed=%d, n=%d).", baseline_seed, n_companies)
    baseline_df = generate_causal_data(n_companies=n_companies, random_seed=baseline_seed)

    logger.info(
        "Generating current data (%s, seed=%d).",
        "drifted" if use_drifted else "baseline",
        current_seed,
    )
    current_df: pd.DataFrame
    if use_drifted:
        current_df = generate_drifted_data(n_companies=n_companies, random_seed=current_seed)
    else:
        current_df = generate_causal_data(n_companies=n_companies, random_seed=current_seed)

    monitor = CausalDriftMonitor(dag=BASELINE_DAG, config=cfg)
    baseline_est = monitor.establish_baseline(baseline_df)
    report = monitor.check_drift(current_df)

    result = {
        "metadata": {
            "run_at": datetime.now(tz=timezone.utc).isoformat(),
            "baseline_seed": baseline_seed,
            "current_seed": current_seed,
            "n_companies": n_companies,
            "use_drifted": use_drifted,
        },
        "config": {
            "refutation_threshold": cfg.refutation_threshold,
            "effect_change_threshold": cfg.effect_change_threshold,
            "n_permutations": cfg.n_permutations,
            "random_seed": cfg.random_seed,
        },
        "baseline": {
            "ate": baseline_est.ate,
            "ci_lower": baseline_est.ci_lower,
            "ci_upper": baseline_est.ci_upper,
            "std_error": baseline_est.std_error,
            "n_obs": baseline_est.n_obs,
        },
        "current": {
            "ate": report.current_estimate.ate,
            "ci_lower": report.current_estimate.ci_lower,
            "ci_upper": report.current_estimate.ci_upper,
            "std_error": report.current_estimate.std_error,
            "n_obs": report.current_estimate.n_obs,
        },
        "drift_report": {
            "drift_detected": report.drift_detected,
            "severity": report.severity.value,
            "reasons": list(report.reasons),
        },
    }

    if output_dir is not None:
        _write_artifact(result, output_dir)

    if report.drift_detected:
        logger.warning("Drift detected: %s", report.summary)
    else:
        logger.info("No drift: %s", report.summary)

    return result


def _write_artifact(result: dict[str, Any], output_dir: Path) -> None:
    """Write run result to a timestamped JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = output_dir / f"{ts}.json"
    path.write_text(json.dumps(result, indent=2))
    logger.info("Run artifact written to %s.", path)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run causal drift detection on synthetic data."
    )
    parser.add_argument("--baseline-seed", type=int, default=42)
    parser.add_argument("--current-seed", type=int, default=100)
    parser.add_argument("--n-companies", type=int, default=5000)
    parser.add_argument(
        "--use-drifted",
        action="store_true",
        default=True,
        help="Use the drifted DGP for the current dataset.",
    )
    parser.add_argument(
        "--no-drift",
        dest="use_drifted",
        action="store_false",
        help="Use the baseline DGP for both periods (control run).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show timestamped progress logs in addition to the JSON report.",
    )
    parser.add_argument(
        "--fail-on-drift",
        action="store_true",
        help="Exit with status 1 when drift is detected. Useful for CI or alerting jobs.",
    )
    args = parser.parse_args(argv)
    _configure_cli_logging(verbose=args.verbose)

    result = run(
        baseline_seed=args.baseline_seed,
        current_seed=args.current_seed,
        n_companies=args.n_companies,
        use_drifted=args.use_drifted,
        output_dir=args.output_dir,
    )

    print(json.dumps(result["drift_report"], indent=2))
    if args.fail_on_drift and result["drift_report"]["drift_detected"]:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
