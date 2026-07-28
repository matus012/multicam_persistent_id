"""Per-camera detection + tracking + ReID embedding."""

from mcreid.track.per_view import Detection, PerViewConfig, PerViewTracker, iou_matrix

__all__ = ["Detection", "PerViewConfig", "PerViewTracker", "iou_matrix"]
