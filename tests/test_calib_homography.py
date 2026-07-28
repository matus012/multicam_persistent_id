"""Tests for mcreid.calib.homography — ground homography fitting + AprilTag geometry."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from mcreid.calib.homography import (
    TagPlacement,
    check_non_degenerate,
    correspondences_from_tags,
    fit_ground_homography,
    ground_plane_from_correspondences,
)
from mcreid.calib.schema import Intrinsics

FloatArray = npt.NDArray[np.float64]

_IMAGE_QUAD = np.array([[100.0, 100.0], [500.0, 100.0], [500.0, 400.0], [100.0, 400.0]])
_WORLD_QUAD = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]])


def _stub_intrinsics() -> Intrinsics:
    return Intrinsics(
        fx=900.0,
        fy=900.0,
        cx=640.0,
        cy=360.0,
        dist_coeffs=[0.0] * 5,
        image_width=1280,
        image_height=720,
        rms_reproj_px=0.0,
        n_views=0,
    )


# --- fit_ground_homography ------------------------------------------------------


def test_fit_ground_homography_exact_four_points() -> None:
    H, rms, n_inliers = fit_ground_homography(_IMAGE_QUAD, _WORLD_QUAD)
    assert rms < 1e-9, f"4-point exact fit should have ~0 residual, got {rms}"
    assert n_inliers == 4

    homo = np.hstack([_IMAGE_QUAD, np.ones((4, 1))])
    proj = homo @ H.T
    world = proj[:, :2] / proj[:, 2:3]
    assert np.allclose(world, _WORLD_QUAD, atol=1e-9)


def test_fit_ground_homography_ransac_recovers_noisy_correspondences() -> None:
    H_true, _, _ = fit_ground_homography(_IMAGE_QUAD, _WORLD_QUAD)

    rng = np.random.default_rng(0)
    # Kept small (rather than e.g. 30): check_non_degenerate enumerates every
    # triangle (O(n^3)) and cv2.cross emits a NumPy 2.0 deprecation warning per
    # triangle, so a larger n here is mostly warning noise, not test signal.
    n = 12
    img_pts = rng.uniform([100.0, 100.0], [500.0, 400.0], size=(n, 2))
    homo = np.hstack([img_pts, np.ones((n, 1))])
    proj = homo @ H_true.T
    world_clean = proj[:, :2] / proj[:, 2:3]
    world_noisy = world_clean + rng.normal(scale=0.01, size=world_clean.shape)

    H_est, rms, n_inliers = fit_ground_homography(img_pts, world_noisy, ransac_thresh_m=0.05)
    assert rms < 0.05, f"RANSAC fit rms too high: {rms}"
    assert n_inliers >= int(0.7 * n), f"expected most points to be inliers, got {n_inliers}/{n}"
    assert np.allclose(H_est, H_true, atol=0.2), "recovered homography should be close to truth"


def test_fit_ground_homography_rejects_mismatched_counts() -> None:
    with pytest.raises(ValueError, match="count mismatch"):
        fit_ground_homography(_IMAGE_QUAD, _WORLD_QUAD[:3])


# --- check_non_degenerate --------------------------------------------------------


def test_check_non_degenerate_raises_on_collinear_points() -> None:
    collinear = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    with pytest.raises(ValueError, match="collinear"):
        check_non_degenerate(collinear, "test_pts")


def test_check_non_degenerate_raises_on_too_few_points() -> None:
    with pytest.raises(ValueError, match=">= 4"):
        check_non_degenerate(np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]), "test_pts")


def test_check_non_degenerate_accepts_well_spread_points() -> None:
    check_non_degenerate(_WORLD_QUAD, "test_pts")  # must not raise


def test_check_non_degenerate_raises_on_near_collinear_quad() -> None:
    # Points hug the y=x diagonal (a non-zero-area bbox) closely enough that
    # the largest triangle is under 2% of the bounding-box area.
    almost_collinear = np.array([[0.0, 0.0], [1.0, 1.01], [2.0, 2.02], [3.0, 2.99]])
    with pytest.raises(ValueError, match="near-collinear"):
        check_non_degenerate(almost_collinear, "test_pts")


# --- TagPlacement -----------------------------------------------------------------


def test_tag_placement_rejects_bad_params() -> None:
    with pytest.raises(ValueError, match="tag_id"):
        TagPlacement(tag_id=-1, center_xy_m=(0.0, 0.0), size_m=0.2)
    with pytest.raises(ValueError, match="size_m"):
        TagPlacement(tag_id=1, center_xy_m=(0.0, 0.0), size_m=0.0)


def test_tag_placement_world_corners_ordering_at_yaw_zero() -> None:
    tag = TagPlacement(tag_id=1, center_xy_m=(0.0, 0.0), size_m=1.0, yaw_deg=0.0)
    corners = tag.world_corners()
    assert corners.shape == (4, 2)
    tl, tr, br, bl = corners
    assert np.allclose(tl, [-0.5, 0.5])
    assert np.allclose(tr, [0.5, 0.5])
    assert np.allclose(br, [0.5, -0.5])
    assert np.allclose(bl, [-0.5, -0.5])


def test_tag_placement_world_corners_yaw_90_rotates_correctly() -> None:
    tag0 = TagPlacement(tag_id=1, center_xy_m=(1.0, 2.0), size_m=1.0, yaw_deg=0.0)
    tag90 = TagPlacement(tag_id=1, center_xy_m=(1.0, 2.0), size_m=1.0, yaw_deg=90.0)
    corners0 = tag0.world_corners() - np.array([1.0, 2.0])
    corners90 = tag90.world_corners() - np.array([1.0, 2.0])

    rot90 = np.array([[0.0, -1.0], [1.0, 0.0]])
    expected = corners0 @ rot90.T
    assert np.allclose(
        corners90, expected, atol=1e-9
    ), "yaw=90 must rotate every corner by +90 degrees about the tag centre"


# --- correspondences_from_tags -----------------------------------------------------


def test_correspondences_from_tags_matches_detections_to_placements() -> None:
    placements = [
        TagPlacement(tag_id=1, center_xy_m=(1.0, 1.0), size_m=0.2),
        TagPlacement(tag_id=2, center_xy_m=(3.0, 2.0), size_m=0.2),
    ]
    detections = {
        1: np.array([[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]]),
        2: np.array([[110.0, 10.0], [120.0, 10.0], [120.0, 20.0], [110.0, 20.0]]),
    }
    image_pts, world_pts = correspondences_from_tags(detections, placements, min_tags=2)
    assert image_pts.shape == (8, 2)
    assert world_pts.shape == (8, 2)
    assert np.allclose(world_pts[:4], placements[0].world_corners())
    assert np.allclose(world_pts[4:], placements[1].world_corners())


def test_correspondences_from_tags_raises_when_too_few_matched() -> None:
    placements = [
        TagPlacement(tag_id=1, center_xy_m=(1.0, 1.0), size_m=0.2),
        TagPlacement(tag_id=2, center_xy_m=(3.0, 2.0), size_m=0.2),
    ]
    detections = {1: np.array([[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]])}
    with pytest.raises(RuntimeError, match="only 1 of 2"):
        correspondences_from_tags(detections, placements, min_tags=2)


def test_correspondences_from_tags_raises_on_duplicate_tag_id() -> None:
    placements = [
        TagPlacement(tag_id=1, center_xy_m=(1.0, 1.0), size_m=0.2),
        TagPlacement(tag_id=1, center_xy_m=(3.0, 2.0), size_m=0.2),
    ]
    detections = {1: np.array([[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]])}
    with pytest.raises(ValueError, match="duplicate tag_id"):
        correspondences_from_tags(detections, placements, min_tags=1)


# --- ground_plane_from_correspondences ------------------------------------------


def test_ground_plane_from_correspondences_end_to_end() -> None:
    intr = _stub_intrinsics()
    gp = ground_plane_from_correspondences(intr, _IMAGE_QUAD, _WORLD_QUAD, method="four_point")
    assert gp.method == "four_point"
    assert gp.rms_error_m < 1e-6
    assert gp.n_correspondences == 4
    assert gp.floor_extent_m is not None


def test_ground_plane_from_correspondences_rejects_unknown_method() -> None:
    intr = _stub_intrinsics()
    with pytest.raises(ValueError, match="unknown method"):
        ground_plane_from_correspondences(intr, _IMAGE_QUAD, _WORLD_QUAD, method="bogus")
