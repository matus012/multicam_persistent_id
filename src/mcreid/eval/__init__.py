"""Evaluation: identity consistency, multi-view detection metrics."""

from mcreid.eval.id_metrics import (
    DEFAULT_MATCH_RADIUS_M,
    IdConsistencyReport,
    evaluate_id_consistency,
)

__all__ = ["DEFAULT_MATCH_RADIUS_M", "IdConsistencyReport", "evaluate_id_consistency"]
