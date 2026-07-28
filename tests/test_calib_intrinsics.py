"""Tests for mcreid.calib.intrinsics — checkerboard intrinsics recovery.

No image files: corners are synthesised by projecting a known `CheckerboardSpec`
through a known `VirtualCamera.project()`, mirroring a real multi-pose
checkerboard shoot without any I/O.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from mcreid.calib.intrinsics import CheckerboardSpec, calibrate_intrinsics_from_corners
from mcreid.sim.virtual_camera import VirtualCamera

FloatArray = npt.NDArray[np.float64]


def _rotation_xyz(rx_deg: float, ry_deg: float, rz_deg: float) -> FloatArray:
    """Intrinsic X-Y-Z Euler rotation, used only to tilt synthetic board poses."""
    rx, ry, rz = np.radians([rx_deg, ry_deg, rz_deg])
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _synthetic_corner_sets(
    cam: VirtualCamera, spec: CheckerboardSpec, n_poses: int, seed: int = 42
) -> list[FloatArray]:
    """Project `n_poses` random-but-in-frame board poses through `cam`.

    Board poses are generated directly in the camera frame (depth ahead of the
    camera, small lateral/vertical offset, random tilt) and then mapped to
    world coordinates via world = C + cam_frame @ R, which is the algebraic
    inverse of VirtualCamera's own (pts - C) @ R.T. Poses that would fall
    outside the image are discarded.
    """
    obj = spec.object_points()
    center = obj.mean(axis=0)
    R, C = cam.R, cam.C
    rng = np.random.default_rng(seed)
    width, height = cam.image_size

    corner_sets: list[FloatArray] = []
    attempts = 0
    while len(corner_sets) < n_poses and attempts < n_poses * 20:
        attempts += 1
        depth = rng.uniform(0.7, 2.0)
        max_lateral = depth * np.tan(np.radians(cam.hfov_deg / 2.0)) * 0.5
        cx = rng.uniform(-max_lateral, max_lateral)
        cy = rng.uniform(-max_lateral * 0.5, max_lateral * 0.5)
        rot = _rotation_xyz(rng.uniform(-25, 25), rng.uniform(-25, 25), rng.uniform(-180, 180))
        board_cam = (obj - center) @ rot.T + np.array([cx, cy, depth])
        world = C + board_cam @ R
        px, in_front = cam.project(world)
        in_frame = (
            in_front.all()
            and (px[:, 0] >= 0).all()
            and (px[:, 0] <= width).all()
            and (px[:, 1] >= 0).all()
            and (px[:, 1] <= height).all()
        )
        if in_frame:
            corner_sets.append(px)
    assert (
        len(corner_sets) == n_poses
    ), f"only generated {len(corner_sets)}/{n_poses} in-frame synthetic board poses"
    return corner_sets


# --- CheckerboardSpec validation ------------------------------------------------


def test_checkerboard_spec_rejects_square_pattern() -> None:
    with pytest.raises(ValueError, match="rotationally ambiguous"):
        CheckerboardSpec(inner_corners=(6, 6), square_size_m=0.025)


def test_checkerboard_spec_rejects_non_positive_square_size() -> None:
    with pytest.raises(ValueError, match="square_size_m must be positive"):
        CheckerboardSpec(inner_corners=(9, 6), square_size_m=0.0)
    with pytest.raises(ValueError, match="square_size_m must be positive"):
        CheckerboardSpec(inner_corners=(9, 6), square_size_m=-0.01)


def test_checkerboard_spec_rejects_too_few_inner_corners() -> None:
    with pytest.raises(ValueError, match="inner_corners"):
        CheckerboardSpec(inner_corners=(2, 6), square_size_m=0.025)


def test_checkerboard_spec_object_points_shape_and_scale() -> None:
    spec = CheckerboardSpec(inner_corners=(9, 6), square_size_m=0.03)
    obj = spec.object_points()
    assert obj.shape == (54, 3)
    assert np.allclose(obj[:, 2], 0.0), "board frame is Z=0"
    assert obj[:, :2].max() == pytest.approx(8 * 0.03), "grid must span (cols-1)*square_size"


# --- calibrate_intrinsics_from_corners ------------------------------------------


def test_calibrate_intrinsics_from_synthetic_corners_recovers_ground_truth() -> None:
    cam = VirtualCamera(
        "cam_calib",
        (1.0, 1.0, 1.5),
        yaw_deg=20.0,
        pitch_deg=25.0,
        hfov_deg=70.0,
        image_size=(1280, 720),
    )
    spec = CheckerboardSpec(inner_corners=(9, 6), square_size_m=0.03)
    corner_sets = _synthetic_corner_sets(cam, spec, n_poses=12)

    intr = calibrate_intrinsics_from_corners(corner_sets, spec, cam.image_size, min_views=8)

    true_fx, true_fy = cam.K[0, 0], cam.K[1, 1]
    true_cx, true_cy = cam.K[0, 2], cam.K[1, 2]
    assert intr.fx == pytest.approx(true_fx, rel=0.02), f"fx off: {intr.fx} vs {true_fx}"
    assert intr.fy == pytest.approx(true_fy, rel=0.02), f"fy off: {intr.fy} vs {true_fy}"
    assert intr.cx == pytest.approx(true_cx, rel=0.02), f"cx off: {intr.cx} vs {true_cx}"
    assert intr.cy == pytest.approx(true_cy, rel=0.02), f"cy off: {intr.cy} vs {true_cy}"
    assert intr.rms_reproj_px < 0.5, f"rms_reproj_px too high: {intr.rms_reproj_px}"
    assert intr.n_views == len(corner_sets)


def test_calibrate_intrinsics_rejects_too_few_views() -> None:
    spec = CheckerboardSpec(inner_corners=(9, 6), square_size_m=0.03)
    with pytest.raises(ValueError, match=">= 8"):
        calibrate_intrinsics_from_corners([np.zeros((54, 2))] * 3, spec, (1280, 720), min_views=8)


def test_calibrate_intrinsics_rejects_wrong_corner_shape() -> None:
    spec = CheckerboardSpec(inner_corners=(9, 6), square_size_m=0.03)
    with pytest.raises(ValueError, match="expected"):
        calibrate_intrinsics_from_corners([np.zeros((10, 2))] * 9, spec, (1280, 720), min_views=8)
