"""Tests for mcreid.sim.virtual_camera — the analytic pinhole camera fixture."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mcreid.sim.toy import bedroom_rig
from mcreid.sim.virtual_camera import VirtualCamera

# --- construction validation -----------------------------------------------------


def test_rejects_camera_at_or_below_floor() -> None:
    with pytest.raises(ValueError, match="above the floor"):
        VirtualCamera("c", (0.0, 0.0, 0.0), yaw_deg=0.0, pitch_deg=30.0)


def test_rejects_hfov_out_of_range() -> None:
    with pytest.raises(ValueError, match="hfov_deg"):
        VirtualCamera("c", (0.0, 0.0, 2.0), yaw_deg=0.0, pitch_deg=30.0, hfov_deg=180.0)
    with pytest.raises(ValueError, match="hfov_deg"):
        VirtualCamera("c", (0.0, 0.0, 2.0), yaw_deg=0.0, pitch_deg=30.0, hfov_deg=0.0)


def test_rejects_pitch_out_of_range() -> None:
    with pytest.raises(ValueError, match="pitch_deg"):
        VirtualCamera("c", (0.0, 0.0, 2.0), yaw_deg=0.0, pitch_deg=0.0)
    with pytest.raises(ValueError, match="pitch_deg"):
        VirtualCamera("c", (0.0, 0.0, 2.0), yaw_deg=0.0, pitch_deg=90.0)


def test_rejects_invalid_image_size() -> None:
    with pytest.raises(ValueError, match="image_size"):
        VirtualCamera("c", (0.0, 0.0, 2.0), yaw_deg=0.0, pitch_deg=30.0, image_size=(0, 720))


# --- K / R properties --------------------------------------------------------------


def test_k_matrix_matches_hfov_formula() -> None:
    cam = VirtualCamera(
        "c", (0.0, 0.0, 2.0), yaw_deg=0.0, pitch_deg=30.0, hfov_deg=70.0, image_size=(1280, 720)
    )
    expected_fx = (1280 / 2.0) / math.tan(math.radians(70.0) / 2.0)
    assert cam.K[0, 0] == pytest.approx(expected_fx)
    assert cam.K[1, 1] == pytest.approx(expected_fx), "pixels are square"
    assert cam.K[0, 2] == pytest.approx(640.0)
    assert cam.K[1, 2] == pytest.approx(360.0)


def test_r_is_orthonormal() -> None:
    cam = VirtualCamera("c", (1.0, 2.0, 2.0), yaw_deg=37.0, pitch_deg=22.0)
    R = cam.R
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9), "R must be orthonormal"
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-9), "R must be a proper rotation"


# --- project ------------------------------------------------------------------------


def test_project_marks_points_behind_camera_as_not_in_front() -> None:
    cam = VirtualCamera("c", (3.0, 2.5, 2.0), yaw_deg=0.0, pitch_deg=30.0)
    behind = np.array([[-10.0, 2.5, 0.0]])  # far -x, camera yaw=0 looks toward +x
    pix, in_front = cam.project(behind)
    assert not in_front[0]
    assert np.isnan(pix[0]).all()


def test_project_accepts_1d_single_point() -> None:
    cam = VirtualCamera("c", (0.3, 0.3, 2.2), yaw_deg=45.0, pitch_deg=28.0)
    pix, in_front = cam.project(np.array([3.0, 2.5, 0.0]))
    assert pix.shape == (1, 2)
    assert in_front.shape == (1,)


def test_project_rejects_wrong_shape() -> None:
    cam = VirtualCamera("c", (0.3, 0.3, 2.2), yaw_deg=45.0, pitch_deg=28.0)
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        cam.project(np.array([[1.0, 2.0]]))


# --- person_bbox / visible_fraction ---------------------------------------------


def test_person_bbox_none_when_behind_camera() -> None:
    cam = VirtualCamera("c", (3.0, 2.5, 2.0), yaw_deg=0.0, pitch_deg=30.0)
    assert cam.person_bbox((-10.0, 2.5), height_m=1.75) is None


def test_person_bbox_shape_and_width_scaling() -> None:
    cam = bedroom_rig()[0]
    box = cam.person_bbox((3.0, 2.5), height_m=1.75, width_m=0.55)
    assert box is not None
    assert box.shape == (4,)
    assert box[2] > box[0] and box[3] > box[1], "box must be well-formed xyxy"

    # The box's y-extent runs foot-to-head; its x-centre is the midpoint
    # between the projected foot and head pixels (they differ under
    # perspective unless the camera looks straight down).
    foot_px, in_front = cam.project(np.array([[3.0, 2.5, 0.0], [3.0, 2.5, 1.75]]))
    assert in_front.all()
    expected_cx = 0.5 * (foot_px[0, 0] + foot_px[1, 0])
    box_cx = 0.5 * (box[0] + box[2])
    assert box_cx == pytest.approx(expected_cx, abs=1e-6)
    assert box[3] == pytest.approx(foot_px[0, 1], abs=1e-6), "box bottom must be the foot pixel row"

    # Pixel width scales with the true metric width at the same depth as the body.
    px_height = abs(foot_px[0, 1] - foot_px[1, 1])
    expected_width = px_height * (0.55 / 1.75)
    assert (box[2] - box[0]) == pytest.approx(expected_width, rel=1e-6)


def test_person_bbox_none_when_fully_outside_frame() -> None:
    cam = bedroom_rig()[0]
    # Far outside the room and off to the side of this camera's view.
    assert cam.person_bbox((100.0, 100.0), height_m=1.75) is None


def test_visible_fraction_full_when_unclipped_and_zero_when_offscreen() -> None:
    cam = bedroom_rig()[0]
    frac_visible = cam.visible_fraction((3.0, 2.5), height_m=1.75)
    assert 0.0 < frac_visible <= 1.0

    frac_hidden = cam.visible_fraction((100.0, 100.0), height_m=1.75)
    assert frac_hidden == 0.0


# --- to_calib -------------------------------------------------------------------


def test_to_calib_intrinsics_match_camera_k() -> None:
    cam = VirtualCamera("cam_x", (0.3, 0.3, 2.2), yaw_deg=45.0, pitch_deg=28.0)
    calib = cam.to_calib(floor_extent_m=(0.0, 0.0, 6.0, 5.0))
    assert calib.camera_id == "cam_x"
    assert np.allclose(calib.intrinsics.K, cam.K)
    assert calib.height_m == pytest.approx(cam.position_m[2])


def test_to_calib_ground_homography_is_exact_projective_inverse() -> None:
    cam = VirtualCamera("cam_x", (0.3, 0.3, 2.2), yaw_deg=45.0, pitch_deg=28.0)
    calib = cam.to_calib(floor_extent_m=(0.0, 0.0, 6.0, 5.0))
    assert np.allclose(
        calib.ground.H, cam.H_img2world
    ), "to_calib()'s GroundPlane must carry the camera's own exact analytic homography"
