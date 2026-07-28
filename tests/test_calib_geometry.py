"""Tests for mcreid.calib.geometry — image <-> ground-plane projection.

Includes the G-M1-1 calibration round-trip gate: world -> image -> world error
must be < 1e-9 m for every camera in `bedroom_rig()`.
"""

from __future__ import annotations

import numpy as np
import pytest

from mcreid.calib.geometry import (
    apply_homography,
    feet_point,
    ground_covariance,
    ground_to_image,
    horizon_sign,
    image_to_ground,
)
from mcreid.calib.schema import CameraCalib
from mcreid.sim.toy import bedroom_rig
from mcreid.sim.virtual_camera import VirtualCamera

ROOM = (6.0, 5.0)


def _calib(cam: VirtualCamera) -> CameraCalib:
    return cam.to_calib(floor_extent_m=(0.0, 0.0, *ROOM))


# --- G-M1-1 gate: calibration round-trip -------------------------------------


def test_gm1_1_world_to_image_to_world_round_trip_under_1e9() -> None:
    """THE G-M1-1 gate: for every camera in bedroom_rig(), points inside the
    room round-trip world -> image -> world to under 1e-9 m."""
    interior_points = np.array(
        [[1.0, 1.0], [3.0, 2.5], [5.0, 4.0], [0.5, 4.5], [2.2, 0.6], [4.4, 3.9]]
    )
    for cam in bedroom_rig():
        calib = _calib(cam)
        pixels, in_front = cam.project(
            np.hstack([interior_points, np.zeros((interior_points.shape[0], 1))])
        )
        assert in_front.all(), f"{cam.camera_id}: interior points must be in front of camera"

        world, valid = image_to_ground(calib, pixels)
        assert valid.all(), f"{cam.camera_id}: interior points must be valid (not past horizon)"

        err = np.linalg.norm(world - interior_points, axis=1)
        assert np.all(
            err < 1e-9
        ), f"{cam.camera_id}: round-trip error {err.max():.3e} m exceeds 1e-9 m gate"


def test_image_to_ground_marks_above_horizon_points_invalid() -> None:
    """A pixel near the top of the frame, for a shallow-pitch camera, lies above
    the horizon and must be rejected with NaN + valid=False."""
    shallow = VirtualCamera("cam_shallow", (0.3, 0.3, 1.4), yaw_deg=45.0, pitch_deg=16.0)
    calib = _calib(shallow)

    above_horizon = np.array([[640.0, 5.0], [640.0, 10.0], [500.0, 50.0]])
    world, valid = image_to_ground(calib, above_horizon)
    assert not valid.any(), "pixels near the top of a shallow-pitch frame must be invalid"
    assert np.isnan(world).all(), "invalid rows must be NaN"

    below_horizon = np.array([[640.0, 700.0]])
    world_ok, valid_ok = image_to_ground(calib, below_horizon)
    assert valid_ok.all(), "a pixel near the bottom of the frame must be valid"
    assert np.isfinite(world_ok).all()


def test_horizon_sign_normal_case() -> None:
    cam = bedroom_rig()[0]
    calib = _calib(cam)
    sign = horizon_sign(calib.ground.H, calib.intrinsics.image_size)
    assert sign in (1.0, -1.0)


def test_horizon_sign_raises_on_degenerate_bottom_centre() -> None:
    image_size = (1280, 720)
    width, _height = image_size
    H = np.eye(3)
    # Row 2 chosen so H[2] @ [w/2, h-1, 1] == 0 exactly.
    H[2] = [1.0, 0.0, -(width / 2.0)]
    with pytest.raises(ValueError, match="horizon"):
        horizon_sign(H, image_size)


# --- apply_homography ---------------------------------------------------------


def test_apply_homography_basic_mapping() -> None:
    H = np.array([[0.01, 0.0, -1.0], [0.0, 0.01, -1.0], [0.0, 0.0, 1.0]])
    pts = np.array([[100.0, 100.0], [200.0, 300.0]])
    out, valid = apply_homography(H, pts)
    expected = pts * 0.01 - 1.0
    assert valid.all()
    assert np.allclose(out, expected)


def test_apply_homography_flags_near_horizon_as_invalid() -> None:
    # Row 2 == [1, 0, 0] sends x == 0 to scale 0 (on the horizon).
    H = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    pts = np.array([[0.0, 5.0], [10.0, 5.0]])
    out, valid = apply_homography(H, pts)
    assert not valid[0], "a point with homogeneous scale ~ 0 must be invalid"
    assert np.isnan(out[0]).all()
    assert valid[1]


def test_apply_homography_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match="H must be 3x3"):
        apply_homography(np.eye(2), np.array([[0.0, 0.0]]))


# --- feet_point ----------------------------------------------------------------


def test_feet_point_bottom_centre() -> None:
    boxes = np.array([[10.0, 20.0, 30.0, 60.0], [0.0, 0.0, 4.0, 8.0]])
    feet = feet_point(boxes)
    assert np.allclose(feet, [[20.0, 60.0], [2.0, 8.0]])


def test_feet_point_accepts_single_box_1d() -> None:
    feet = feet_point(np.array([10.0, 20.0, 30.0, 60.0]))
    assert feet.shape == (1, 2)
    assert np.allclose(feet, [[20.0, 60.0]])


def test_feet_point_rejects_malformed_box() -> None:
    with pytest.raises(ValueError, match="x2 >= x1"):
        feet_point(np.array([10.0, 20.0, 5.0, 60.0]))  # x2 < x1
    with pytest.raises(ValueError, match="x2 >= x1"):
        feet_point(np.array([10.0, 20.0, 30.0, 5.0]))  # y2 < y1


def test_feet_point_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match=r"\(N, 4\)"):
        feet_point(np.array([[1.0, 2.0, 3.0]]))


# --- ground_covariance ---------------------------------------------------------


def test_ground_covariance_grows_with_distance_and_has_model_sigma_floor() -> None:
    cam = VirtualCamera("cam0", (0.3, 0.3, 2.2), yaw_deg=45.0, pitch_deg=28.0)
    calib = _calib(cam)
    model_sigma_m = 0.15
    floor = 2.0 * model_sigma_m**2

    world_points = np.array([[0.6, 0.6], [2.0, 1.8], [4.0, 3.2], [5.6, 4.6]])
    pixels, in_front = cam.project(np.hstack([world_points, np.zeros((world_points.shape[0], 1))]))
    assert in_front.all()
    distances = np.linalg.norm(world_points - np.array(cam.position_m[:2]), axis=1)

    cov = ground_covariance(calib, pixels, model_sigma_m=model_sigma_m)
    traces = np.trace(cov, axis1=1, axis2=2)

    assert np.all(
        traces >= floor - 1e-12
    ), f"trace must never fall below the model_sigma_m floor of {floor}, got min {traces.min()}"
    order = np.argsort(distances)
    assert np.all(np.diff(traces[order]) >= -1e-9), (
        "ground covariance trace must grow (monotonically, up to numerical noise) "
        "with distance from the camera"
    )


def test_ground_covariance_rejects_non_positive_sigma_px() -> None:
    cam = bedroom_rig()[0]
    calib = _calib(cam)
    with pytest.raises(ValueError, match="sigma_px"):
        ground_covariance(calib, np.array([[640.0, 500.0]]), sigma_px=0.0)


def test_ground_covariance_nan_for_invalid_points() -> None:
    shallow = VirtualCamera("cam_shallow", (0.3, 0.3, 1.4), yaw_deg=45.0, pitch_deg=16.0)
    calib = _calib(shallow)
    cov = ground_covariance(calib, np.array([[640.0, 5.0]]))
    assert np.isnan(cov).all()


# --- ground_to_image (inverse direction) ---------------------------------------


def test_ground_to_image_round_trips_with_image_to_ground() -> None:
    cam = bedroom_rig()[2]
    calib = _calib(cam)
    world_points = np.array([[1.5, 1.2], [4.0, 3.0]])
    pixels, valid = ground_to_image(calib, world_points)
    assert valid.all()

    back, valid_back = image_to_ground(calib, pixels)
    assert valid_back.all()
    assert np.allclose(back, world_points, atol=1e-6)
