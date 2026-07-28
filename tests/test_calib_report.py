"""Tests for the calibration sanity gate (workstream 2a).

This gate is the thing standing between a mis-measured room and a whole session
of quietly meaningless tracking numbers, so it is tested for its ability to
REJECT, not only to pass.
"""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt
import pytest

from mcreid.calib.homography import TagPlacement
from mcreid.calib.report import (
    DEFAULT_MAX_FIT_RESIDUAL_M,
    analyse_camera,
    leave_one_out_error,
)
from mcreid.calib.schema import CameraCalib
from mcreid.sim.virtual_camera import VirtualCamera

ROOM = (6.0, 5.0)
TAGS = [
    TagPlacement(0, (1.0, 1.0), 0.26, 0.0),
    TagPlacement(1, (5.0, 1.2), 0.26, 0.0),
    TagPlacement(2, (4.8, 3.8), 0.26, 0.0),
    TagPlacement(3, (1.2, 3.6), 0.26, 0.0),
    TagPlacement(4, (3.0, 2.5), 0.26, 0.0),
]


def _camera() -> VirtualCamera:
    return VirtualCamera("cam0", (0.3, 0.3, 2.4), yaw_deg=45.0, pitch_deg=30.0)


def _render_tags(cam: VirtualCamera) -> npt.NDArray[np.uint8]:
    """Render the floor tags as this camera sees them."""
    width, height = cam.image_size
    canvas = np.full((height, width, 3), 190, dtype=np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)
    for tag in TAGS:
        marker = cv2.cvtColor(
            cv2.aruco.generateImageMarker(dictionary, tag.tag_id, 400), cv2.COLOR_GRAY2BGR
        )
        world = np.hstack([tag.world_corners(), np.zeros((4, 1))])
        pixels, ok = cam.project(world)
        if not ok.all():
            continue
        src = np.array([[0, 0], [399, 0], [399, 399], [0, 399]], dtype=np.float32)
        H = cv2.getPerspectiveTransform(src, pixels.astype(np.float32))
        warped = cv2.warpPerspective(marker, H, (width, height), borderValue=(190, 190, 190))
        mask = cv2.warpPerspective(
            np.full((400, 400), 255, np.uint8), H, (width, height), borderValue=0
        )
        canvas[mask > 0] = warped[mask > 0]
    return canvas


def _calib(cam: VirtualCamera) -> CameraCalib:
    return cam.to_calib(floor_extent_m=(0.0, 0.0, *ROOM))


def _corrupt(calib: CameraCalib, mode: str) -> CameraCalib:
    """Corrupt a calibration the way a real mistake would."""
    H = np.array(calib.ground.H_img2world, dtype=np.float64)
    if mode == "shift":  # tag centres mis-measured
        H[0, 2] += 0.30
        H[1, 2] += 0.22
    elif mode == "mirror":  # correspondences entered in the wrong order
        H[:, 0] *= -1.0
    elif mode == "scale":  # square size entered in cm instead of m
        H[:2, :] *= 1.20
    else:  # pragma: no cover
        raise ValueError(mode)
    return calib.model_copy(
        update={
            "ground": calib.ground.model_copy(
                update={"H_img2world": [[float(v) for v in row] for row in H]}
            )
        }
    )


# --- the gate accepts a correct calibration ------------------------------------------------


def test_good_calibration_passes() -> None:
    cam = _camera()
    report, detections = analyse_camera(_calib(cam), _render_tags(cam), TAGS)

    assert detections, "no tags detected in the synthetic render"
    assert report.ok, f"a correct calibration must pass; problems: {report.problems}"
    assert report.fit_residual_m < DEFAULT_MAX_FIT_RESIDUAL_M
    assert report.grid_is_convex


# --- the gate rejects realistic mistakes ---------------------------------------------------


@pytest.mark.parametrize("mode", ["shift", "mirror", "scale"])
def test_corrupted_calibration_is_rejected(mode: str) -> None:
    cam = _camera()
    report, _ = analyse_camera(_corrupt(_calib(cam), mode), _render_tags(cam), TAGS)

    assert not report.ok, f"a {mode}-corrupted calibration must be rejected"
    assert report.problems, "a rejection must come with an actionable explanation"


def test_horizon_flip_is_flagged_as_non_convex() -> None:
    """A metric square on the floor must image as a convex quadrilateral.

    The convexity check exists for the horizon-flip class specifically: negating
    the homogeneous row sends the visible floor to the wrong side of the horizon,
    which produces finite but sign-flipped world coordinates. Other corruptions
    (a shifted or mirrored plane) stay convex and are caught by the residual
    instead — see the parametrised rejection test above.
    """
    cam = _camera()
    calib = _calib(cam)
    H = np.array(calib.ground.H_img2world, dtype=np.float64)
    H[2, :] *= -1.0
    flipped = calib.model_copy(
        update={
            "ground": calib.ground.model_copy(
                update={"H_img2world": [[float(v) for v in row] for row in H]}
            )
        }
    )
    report, _ = analyse_camera(flipped, _render_tags(cam), TAGS)
    assert not report.ok, "a horizon-flipped calibration must be rejected"


def test_missing_tags_are_reported() -> None:
    cam = _camera()
    placements = [*TAGS, TagPlacement(9, (2.0, 2.0), 0.26)]
    report, _ = analyse_camera(_calib(cam), _render_tags(cam), placements)
    assert 9 in report.missing_tag_ids


# --- leave-one-out validates tags.yaml, not calib.json -------------------------------------


def test_leave_one_out_needs_at_least_three_tags() -> None:
    cam = _camera()
    calib = _calib(cam)
    detections = {t.tag_id: np.zeros((4, 2)) for t in TAGS[:2]}
    mean, worst = leave_one_out_error(detections, TAGS[:2], calib)
    assert np.isnan(mean) and np.isnan(worst), "two tags leave nothing to hold out"


def test_leave_one_out_catches_a_mismeasured_tag() -> None:
    """LOO's job: a tag whose recorded world position is wrong.

    It deliberately does not consult the stored homography, so it is the check
    that survives a calibration which fits its own bad inputs perfectly.
    """
    cam = _camera()
    calib = _calib(cam)
    image = _render_tags(cam)

    truthful, detections = analyse_camera(calib, image, TAGS)
    # Move a tag this camera can actually see — displacing an unseen tag proves
    # nothing, and which tags are visible depends on the mount.
    visible = sorted(set(detections) & {t.tag_id for t in TAGS})
    assert len(visible) >= 3, f"need >= 3 visible tags for LOO, got {visible}"
    target = visible[0]

    moved = [
        TagPlacement(
            t.tag_id, (t.center_xy_m[0] + 0.4, t.center_xy_m[1] + 0.3), t.size_m, t.yaw_deg
        )
        if t.tag_id == target
        else t
        for t in TAGS
    ]
    mismeasured, _ = analyse_camera(calib, image, moved)

    assert truthful.loo_error_mean_m < 0.05
    assert mismeasured.loo_error_mean_m > 0.10, (
        f"moving one tag by 50 cm must show up in LOO; got "
        f"{mismeasured.loo_error_mean_m:.4f} m"
    )
    assert not mismeasured.ok
