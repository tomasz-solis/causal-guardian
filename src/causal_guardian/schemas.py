"""Pandera schemas for DataFrame validation at package boundaries.

Every public function that takes a DataFrame validates the schema at entry.
A bad column name or out-of-range value raises a clear error rather than
propagating silently into statsmodels.
"""

from __future__ import annotations

import pandera as pa
from pandera.typing import Series


class CompanyChurnSchema(pa.DataFrameModel):
    """Schema for the core company-level churn dataset.

    Column constraints reflect the synthetic DGP but are intentionally
    permissive on upper bounds to accommodate real data at different scales.
    """

    marketing_spend: Series[float] = pa.Field(ge=0, nullable=False)
    onboarding_friction_score: Series[float] = pa.Field(ge=0, nullable=False)
    card_usage: Series[float] = pa.Field(ge=0, nullable=False)
    churn: Series[int] = pa.Field(isin=[0, 1], nullable=False)
    plan_tier: Series[int] = pa.Field(ge=0, le=10, nullable=False)

    class Config:
        coerce = True  # cast compatible dtypes (e.g., float churn → int)
        strict = False  # allow extra columns (cohort_month, timestep, etc.)


class TimeSeriesPanelSchema(pa.DataFrameModel):
    """Schema for the rolling-window streaming dataset.

    Requires a `timestep` column so the rolling estimator can group by period.
    """

    timestep: Series[int] = pa.Field(ge=0, nullable=False)
    marketing_spend: Series[float] = pa.Field(ge=0, nullable=False)
    onboarding_friction_score: Series[float] = pa.Field(ge=0, nullable=False)
    card_usage: Series[float] = pa.Field(ge=0, nullable=False)
    churn: Series[int] = pa.Field(isin=[0, 1], nullable=False)
    plan_tier: Series[int] = pa.Field(ge=0, le=10, nullable=False)

    class Config:
        coerce = True
        strict = False
