"""
fitness.py  — v3.1

Fitness = execution time in seconds. Nothing else.

Rationale: plan cost is the optimizer's estimate based on statistics.
The log showed two agents with identical plan cost (147,052) executing in
140s and 267s respectively — plan cost does not predict wall clock time.
Mixing the two in one score produced a number that was neither meaningful
nor explainable.

Plan shape (shuffle_ops, broadcast_ops, full_scans, plan_cost) is still
extracted via EXPLAIN after execution and surfaced for display — it helps
understand WHY a query is fast or slow, but it does not affect selection.

Lower fitness = faster query = better.
"""

from typing import Dict, Any
from debug_logger import get_logger

log = get_logger("fitness")

PENALTY_VALUE = 99999.0   # seconds — clearly impossible, visually obvious in UI


class FitnessEvaluator:

    def __init__(self):
        log.debug("FitnessEvaluator created — fitness = execution_time (seconds)")

    def compute(self, metrics: Dict[str, Any]) -> float:
        """
        Returns execution_time in seconds as the fitness score.
        Lower = faster = better.
        """
        if not metrics:
            log.warning("compute() received empty metrics — returning penalty")
            return PENALTY_VALUE

        t = metrics.get("execution_time")
        if t is None:
            log.warning("compute() — execution_time missing from metrics, returning penalty")
            return PENALTY_VALUE

        log.debug(f"compute() → {t:.3f}s")
        return float(t)

    def penalize(self) -> float:
        log.debug(f"penalize() → {PENALTY_VALUE}")
        return PENALTY_VALUE

    def improvement_percentage(self, baseline_time: float, candidate_time: float) -> float:
        """Positive = faster. Negative = slower."""
        if baseline_time <= 0:
            log.warning("improvement_percentage() — baseline_time is 0, cannot compute %")
            return 0.0
        pct = ((baseline_time - candidate_time) / baseline_time) * 100
        log.debug(
            f"improvement_percentage(): baseline={baseline_time:.2f}s  "
            f"candidate={candidate_time:.2f}s  → {pct:.2f}%"
        )
        return pct

    @staticmethod
    def plan_shape_summary(plan_metrics: Dict[str, Any]) -> str:
        if not plan_metrics:
            return "plan unavailable"
        return (
            f"cost={plan_metrics.get('total_cost', 0):.0f}  "
            f"shuffle={plan_metrics.get('shuffle_ops', 0)}  "
            f"broadcast={plan_metrics.get('broadcast_ops', 0)}  "
            f"scans={plan_metrics.get('full_scans', 0)}  "
            f"stale={plan_metrics.get('stale_stats', 0)}"
        )
