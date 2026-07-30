"""Measure the foot-point error that dominates crowded multi-view fusion.

The claim this module exists to make reproducible: a detection box's bottom edge
is the ground-contact point only when the feet are visible. In a crowd they are
routinely occluded, so the box truncates at whoever stands in front, the
projected "foot point" lands short, and the same person's position disagrees
between cameras by far more than the calibration error.

The measurement is a controlled comparison. For one annotated person seen by two
cameras, project a foot point from each view independently and take the distance
between the two world positions. Do it twice: once from WILDTRACK's
**ground-truth** boxes, once from **detector** boxes matched to those same
people. The homography, the foot-point rule and the undistortion are identical
across both arms, so the only thing that changes is where the box's bottom edge
sits. Any gap between the two arms is attributable to the boxes.

Everything here is pure geometry over boxes supplied by the caller — no detector,
no dataset, no torch — so it is unit-testable on synthetic rigs. The CLI in
``mcreid.cli.wildtrack_run`` supplies real boxes.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from mcreid.calib.geometry import feet_point, image_to_ground
from mcreid.calib.schema import CameraCalib

FloatArray = npt.NDArray[np.float64]

# Thresholds reported alongside the distribution. 1.00 m is the birth-cluster
# radius the fusion stage actually uses, so "beyond 1.00 m" is the operationally
# meaningful column: those pairs cannot be grouped by any radius at that setting.
# 0.35 m is the geometric-merge distance that was tried and rejected.
CLUSTER_RADIUS_M = 1.00
MERGE_RADIUS_M = 0.35


def iou_matrix(boxes_a: npt.ArrayLike, boxes_b: npt.ArrayLike) -> FloatArray:
    """Pairwise IoU between two sets of xyxy boxes.

    Returns an ``(len(a), len(b))`` matrix. Empty inputs give a correctly-shaped
    empty result rather than raising, because a frame with no detections is
    ordinary rather than exceptional.
    """
    a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4)
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)

    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0.0, None) * np.clip(y2 - y1, 0.0, None)

    area_a = np.clip(a[:, 2] - a[:, 0], 0.0, None) * np.clip(a[:, 3] - a[:, 1], 0.0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0.0, None) * np.clip(b[:, 3] - b[:, 1], 0.0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0.0, inter / union, 0.0)


def match_detections_to_gt(
    gt_boxes: npt.ArrayLike,
    det_boxes: npt.ArrayLike,
    iou_threshold: float = 0.5,
) -> dict[int, int]:
    """Greedy highest-IoU-first matching of detector boxes to GT boxes.

    Greedy rather than Hungarian on purpose: this is an attribution step, not an
    evaluation. A globally optimal assignment could pair a GT person with a
    mediocre box in order to improve some other pair's score, which would put a
    box on the wrong person and corrupt the very quantity being measured. Taking
    the best available pair first, and never reusing either side, means every
    accepted match is the best box for that person among those still free.

    Returns ``{gt_index: det_index}`` for pairs at or above ``iou_threshold``.
    """
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError(f"iou_threshold must be in (0, 1], got {iou_threshold}")

    ious = iou_matrix(gt_boxes, det_boxes)
    if ious.size == 0:
        return {}

    order = np.argsort(ious, axis=None)[::-1]
    used_gt: set[int] = set()
    used_det: set[int] = set()
    matches: dict[int, int] = {}
    for flat in order:
        gt_i, det_i = np.unravel_index(flat, ious.shape)
        score = float(ious[gt_i, det_i])
        if score < iou_threshold:
            break
        if int(gt_i) in used_gt or int(det_i) in used_det:
            continue
        used_gt.add(int(gt_i))
        used_det.add(int(det_i))
        matches[int(gt_i)] = int(det_i)
    return matches


def ground_points_per_camera(
    boxes_by_camera: Mapping[str, npt.ArrayLike],
    cameras: Mapping[str, CameraCalib],
) -> dict[str, FloatArray]:
    """Project one box per camera to world XY via the shipped foot-point path.

    Cameras whose projection is invalid (foot point beyond the horizon) are
    dropped rather than returned as NaN, so callers cannot accidentally average
    them in.
    """
    out: dict[str, FloatArray] = {}
    for camera_id, box in boxes_by_camera.items():
        cam = cameras.get(camera_id)
        if cam is None:
            continue
        world, valid = image_to_ground(cam, feet_point(box))
        if bool(valid[0]) and np.all(np.isfinite(world[0])):
            out[camera_id] = np.asarray(world[0], dtype=np.float64)
    return out


def pairwise_disagreements(points_by_camera: Mapping[str, FloatArray]) -> list[float]:
    """Distances between every unordered pair of per-camera world positions.

    A person seen by one camera contributes nothing: with a single view there is
    no cross-camera disagreement to measure, and counting it as zero would
    silently dilute the statistic toward the number of poorly-covered people.
    """
    ids = sorted(points_by_camera)
    return [
        float(np.linalg.norm(points_by_camera[a] - points_by_camera[b]))
        for a, b in itertools.combinations(ids, 2)
    ]


@dataclass(frozen=True)
class DisagreementStats:
    """Distribution of same-person cross-camera position disagreement."""

    n_pairs: int
    mean_m: float
    p50_m: float
    p90_m: float
    max_m: float
    frac_beyond_merge_radius: float
    frac_beyond_cluster_radius: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "n_pairs": self.n_pairs,
            "mean_m": self.mean_m,
            "p50_m": self.p50_m,
            "p90_m": self.p90_m,
            "max_m": self.max_m,
            f"frac_beyond_{MERGE_RADIUS_M:.2f}m": self.frac_beyond_merge_radius,
            f"frac_beyond_{CLUSTER_RADIUS_M:.2f}m": self.frac_beyond_cluster_radius,
        }


def summarize(distances: Sequence[float]) -> DisagreementStats:
    """Reduce a list of pair distances to the reported distribution."""
    if not distances:
        nan = float("nan")
        return DisagreementStats(0, nan, nan, nan, nan, 0.0, 0.0)
    d = np.asarray(distances, dtype=np.float64)
    return DisagreementStats(
        n_pairs=int(d.size),
        mean_m=float(d.mean()),
        p50_m=float(np.percentile(d, 50)),
        p90_m=float(np.percentile(d, 90)),
        max_m=float(d.max()),
        frac_beyond_merge_radius=float((d > MERGE_RADIUS_M).mean()),
        frac_beyond_cluster_radius=float((d > CLUSTER_RADIUS_M).mean()),
    )
