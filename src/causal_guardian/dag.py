"""Causal DAG specifications for Causal Guardian.

All DAG definitions live here. The core monitoring code and tests
import from this module - the graph is never re-defined elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CausalDAG:
    """Declarative specification of a causal directed acyclic graph.

    Args:
        edges: Directed edges as (cause, effect) pairs.
        treatment: Variable whose causal effect on `outcome` is estimated.
        outcome: Target variable.
        confounders: Variables to control for in backdoor adjustment.
            These are the nodes that open backdoor paths from treatment
            to outcome and must be included in the adjustment set.

    Example:
        >>> dag = CausalDAG(
        ...     edges=(("X", "M"), ("X", "Y"), ("M", "Y")),
        ...     treatment="X",
        ...     outcome="Y",
        ...     confounders=(),
        ... )
        >>> "X -> M;" in dag.to_dot()
        True
    """

    edges: tuple[tuple[str, str], ...]
    treatment: str
    outcome: str
    confounders: tuple[str, ...] = field(default_factory=tuple)

    def to_dot(self) -> str:
        """Render the graph as a DOT string suitable for DoWhy."""
        edge_lines = "\n  ".join(f"{a} -> {b};" for a, b in self.edges)
        return f"digraph {{\n  {edge_lines}\n}}"

    @property
    def variables(self) -> frozenset[str]:
        """All variable names mentioned in the graph."""
        nodes: set[str] = set()
        for a, b in self.edges:
            nodes.add(a)
            nodes.add(b)
        return frozenset(nodes)


# ------------------------------------------------------------------
# Named DAGs for the card-usage churn scenario
# ------------------------------------------------------------------

BASELINE_DAG = CausalDAG(
    edges=(
        ("marketing_spend", "card_usage"),
        ("onboarding_friction_score", "card_usage"),
        ("plan_tier", "card_usage"),
        ("plan_tier", "churn"),
        ("card_usage", "churn"),
    ),
    treatment="card_usage",
    outcome="churn",
    confounders=("marketing_spend", "onboarding_friction_score", "plan_tier"),
)
"""Pre-drift regime.

card_usage mediates the effect of marketing and friction on churn.
plan_tier is a confounder: it affects both card_usage (higher tiers onboard
more deliberately) and churn (higher tiers retain better via dedicated CSMs).
The backdoor adjustment must include plan_tier to recover the unbiased ATE.
"""

DRIFT_DAG = CausalDAG(
    edges=(
        ("marketing_spend", "card_usage"),
        ("onboarding_friction_score", "card_usage"),
        ("plan_tier", "card_usage"),
        ("plan_tier", "churn"),
        ("onboarding_friction_score", "churn"),
    ),
    treatment="onboarding_friction_score",
    outcome="churn",
    confounders=("marketing_spend", "plan_tier"),
)
"""Post-drift regime.

friction now has a direct effect on churn; the usage → churn edge is gone.
plan_tier remains a confounder of card_usage and churn, but since usage is
no longer on the causal path, it's not needed as a confounder of friction
unless plan_tier also affects friction (it does not in this DGP).
We still include plan_tier as a control to reduce variance.
"""
