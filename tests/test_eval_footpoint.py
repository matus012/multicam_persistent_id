"""Foot-point disagreement measurement — GPU-free, synthetic rigs only.

The behavioural test that matters is `test_truncating_the_box_inflates_the_disagreement`:
it proves the measurement actually detects the phenomenon it is used to claim,
rather than merely producing numbers. Everything else pins the mechanics.
"""

from __future__ import annotations

import numpy as np
import pytest

from mcreid.calib.schema import CameraCalib
from mcreid.eval.footpoint import (
    CLUSTER_RADIUS_M,
    ground_points_per_camera,
    iou_matrix,
    match_detections_to_gt,
    pairwise_disagreements,
    summarize,
)
from mcreid.sim.toy import bedroom_rig
from mcreid.sim.virtual_camera import VirtualCamera

PERSON_HEIGHT_M = 1.75
ROOM = (6.0, 5.0)


def _calib(cam: VirtualCamera) -> CameraCalib:
    return cam.to_calib(floor_extent_m=(0.0, 0.0, *ROOM))


def _rig() -> tuple[tuple[VirtualCamera, ...], dict[str, CameraCalib]]:
    cams = tuple(bedroom_rig())
    return cams, {c.camera_id: _calib(c) for c in cams}


# --------------------------------------------------------------------------- IoU


def test_iou_matrix_identical_boxes_is_one() -> None:
    box = [[0.0, 0.0, 10.0, 20.0]]
    assert iou_matrix(box, box)[0, 0] == pytest.approx(1.0)


def test_iou_matrix_disjoint_boxes_is_zero() -> None:
    a = [[0.0, 0.0, 10.0, 10.0]]
    b = [[100.0, 100.0, 110.0, 110.0]]
    assert iou_matrix(a, b)[0, 0] == pytest.approx(0.0)


def test_iou_matrix_half_overlap() -> None:
    a = [[0.0, 0.0, 10.0, 10.0]]  # area 100
    b = [[5.0, 0.0, 15.0, 10.0]]  # area 100, intersection 50 -> IoU 50/150
    assert iou_matrix(a, b)[0, 0] == pytest.approx(50.0 / 150.0)


def test_iou_matrix_empty_inputs_give_shaped_empty_not_an_error() -> None:
    """A frame with no detections is ordinary, not exceptional."""
    assert iou_matrix(np.zeros((0, 4)), [[0.0, 0.0, 1.0, 1.0]]).shape == (0, 1)
    assert iou_matrix([[0.0, 0.0, 1.0, 1.0]], np.zeros((0, 4))).shape == (1, 0)


def test_iou_matrix_degenerate_zero_area_box_does_not_divide_by_zero() -> None:
    assert iou_matrix([[5.0, 5.0, 5.0, 5.0]], [[0.0, 0.0, 10.0, 10.0]])[0, 0] == pytest.approx(0.0)


# ------------------------------------------------------------------ GT matching


def test_match_prefers_the_best_box_and_never_reuses_one() -> None:
    gt = [[0.0, 0.0, 10.0, 10.0], [100.0, 0.0, 110.0, 10.0]]
    det = [
        [100.0, 0.0, 110.0, 10.0],  # exact for gt[1]
        [0.0, 0.0, 10.0, 10.0],  # exact for gt[0]
    ]
    assert match_detections_to_gt(gt, det, iou_threshold=0.5) == {0: 1, 1: 0}


def test_match_rejects_pairs_below_threshold() -> None:
    gt = [[0.0, 0.0, 10.0, 10.0]]
    det = [[9.0, 9.0, 19.0, 19.0]]  # tiny overlap
    assert match_detections_to_gt(gt, det, iou_threshold=0.5) == {}


def test_match_is_greedy_so_a_contested_box_goes_to_its_best_owner() -> None:
    """Two GT people, one detection. It must go to the better-fitting person."""
    gt = [[0.0, 0.0, 10.0, 10.0], [2.0, 0.0, 12.0, 10.0]]
    det = [[2.0, 0.0, 12.0, 10.0]]  # exact for gt[1]
    assert match_detections_to_gt(gt, det, iou_threshold=0.5) == {1: 0}


def test_match_rejects_an_invalid_threshold() -> None:
    with pytest.raises(ValueError, match="iou_threshold"):
        match_detections_to_gt([[0.0, 0.0, 1.0, 1.0]], [[0.0, 0.0, 1.0, 1.0]], iou_threshold=0.0)


# ------------------------------------------------------------------- projection


def test_ground_points_recover_the_true_position_from_every_camera() -> None:
    """Sanity floor: with exact boxes and exact calibration, cameras agree."""
    cams, cameras = _rig()
    foot = np.array([3.0, 2.5])
    boxes = {}
    for cam in cams:
        box = cam.person_bbox(foot, PERSON_HEIGHT_M)
        if box is not None:
            boxes[cam.camera_id] = box

    points = ground_points_per_camera(boxes, cameras)
    assert len(points) >= 2, "the toy rig should see the room centre from several cameras"
    for camera_id, xy in points.items():
        assert np.linalg.norm(xy - foot) < 0.05, f"{camera_id} misprojected: {xy} vs {foot}"


def test_ground_points_skip_unknown_cameras() -> None:
    _, cameras = _rig()
    assert ground_points_per_camera({"nonexistent": [0.0, 0.0, 10.0, 20.0]}, cameras) == {}


def test_truncating_the_box_inflates_the_disagreement() -> None:
    """THE claim, in miniature.

    Raise one camera's box bottom edge — exactly what an occluder does when it
    hides the feet — and the two cameras' independent ground estimates must move
    apart. If this failed, the whole foot-point diagnosis would be unfalsifiable
    by this measurement, and the reported numbers would mean nothing.
    """
    cams, cameras = _rig()
    foot = np.array([3.0, 2.5])

    full: dict[str, np.ndarray] = {}
    for cam in cams:
        box = cam.person_bbox(foot, PERSON_HEIGHT_M)
        if box is not None:
            full[cam.camera_id] = np.asarray(box, dtype=np.float64)
    assert len(full) >= 2

    clean = pairwise_disagreements(ground_points_per_camera(full, cameras))
    assert max(clean) < 0.1, f"exact boxes should agree closely, got {clean}"

    # Truncate ONE camera's box: pull the bottom edge up by 30% of box height,
    # leaving the others untouched. This is the occluded-feet case.
    victim = sorted(full)[0]
    truncated = dict(full)
    box = truncated[victim].copy()
    box[3] -= 0.30 * (box[3] - box[1])
    truncated[victim] = box

    dirty = pairwise_disagreements(ground_points_per_camera(truncated, cameras))
    assert max(dirty) > 10 * max(clean), (
        f"truncation must inflate disagreement: clean max {max(clean):.3f} m, "
        f"truncated max {max(dirty):.3f} m"
    )


# ------------------------------------------------------------------ aggregation


def test_a_person_seen_by_one_camera_contributes_no_pair() -> None:
    """Counting a single view as zero disagreement would dilute the statistic."""
    assert pairwise_disagreements({"cam0": np.array([1.0, 1.0])}) == []


def test_pair_count_is_every_unordered_camera_pair() -> None:
    points = {
        "cam0": np.array([0.0, 0.0]),
        "cam1": np.array([3.0, 4.0]),
        "cam2": np.array([0.0, 0.0]),
    }
    distances = pairwise_disagreements(points)
    assert len(distances) == 3
    assert sorted(distances) == pytest.approx([0.0, 5.0, 5.0])


def test_summarize_reports_the_distribution_and_the_radius_fractions() -> None:
    stats = summarize([0.1, 0.2, 0.3, 5.0])
    assert stats.n_pairs == 4
    assert stats.mean_m == pytest.approx(1.4)
    assert stats.max_m == pytest.approx(5.0)
    # 5.0 is the only value beyond the 1.0 m clustering radius.
    assert stats.frac_beyond_cluster_radius == pytest.approx(0.25)
    assert f"frac_beyond_{CLUSTER_RADIUS_M:.2f}m" in stats.as_dict()


def test_summarize_of_nothing_is_empty_rather_than_a_crash() -> None:
    stats = summarize([])
    assert stats.n_pairs == 0
    assert np.isnan(stats.mean_m)
