"""Tests for the command-line runner."""

from __future__ import annotations

import pytest

from causal_guardian import runner


def _drift_result() -> dict:
    """Return the smallest runner result needed by the CLI exit path."""
    return {
        "drift_report": {
            "drift_detected": True,
            "severity": "CRITICAL",
            "reasons": ["Effect direction reversed."],
        }
    }


def test_cli_drift_alert_exits_zero_by_default(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The demo command should print drift without failing the shell command."""
    monkeypatch.setattr(runner, "run", lambda **_: _drift_result())

    with pytest.raises(SystemExit) as exc:
        runner.main(["--use-drifted"])

    assert exc.value.code == 0
    assert '"severity": "CRITICAL"' in capsys.readouterr().out


def test_cli_fail_on_drift_exits_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """CI-style mode should fail when drift is detected."""
    monkeypatch.setattr(runner, "run", lambda **_: _drift_result())

    with pytest.raises(SystemExit) as exc:
        runner.main(["--use-drifted", "--fail-on-drift"])

    assert exc.value.code == 1
