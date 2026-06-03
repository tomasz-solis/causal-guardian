"""Shared pytest fixtures.

All fixtures are session-scoped where the data is expensive to generate
(n=5000), so they run once per test session rather than per test.
"""

from __future__ import annotations

import pandas as pd
import pytest

from causal_guardian.config import DriftConfig
from causal_guardian.data.drift import generate_drifted_data
from causal_guardian.data.synthetic import generate_causal_data


@pytest.fixture(scope="session")
def baseline_df() -> pd.DataFrame:
    """5000-company baseline dataset."""
    return generate_causal_data(n_companies=5000, random_seed=42)


@pytest.fixture(scope="session")
def drifted_df() -> pd.DataFrame:
    """5000-company post-drift dataset."""
    return generate_drifted_data(n_companies=5000, random_seed=100)


@pytest.fixture(scope="session")
def small_baseline_df() -> pd.DataFrame:
    """500-company baseline dataset for fast tests."""
    return generate_causal_data(n_companies=500, random_seed=42)


@pytest.fixture(scope="session")
def fast_config() -> DriftConfig:
    """DriftConfig with reduced simulation counts for fast tests."""
    return DriftConfig(n_permutations=50, n_bootstrap=50, n_subset_reps=20, n_random_cc=20)
