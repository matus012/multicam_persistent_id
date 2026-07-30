"""Ground-plane homography estimation: 4-point floor correspondences or AprilTags.

Corner-order convention (LOCKED — must match how the tags are physically laid out):
    A tag lies flat on the floor, print-side up. Its local frame is +x right,
    +y toward the top of the printed tag. ``yaw_deg`` rotates that local frame
    about world +Z. cv2.aruco returns corners as (TL, TR, BR, BL) in the tag's
    own frame, which in local metres is:
        TL (-s/2, +s/2)  TR (+s/2, +s/2)  BR (+s/2, -s/2)  BL (-s/2, -s/2)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

from mcreid.calib.geometry import apply_homography, horizon_sign, undistort_points
from mcreid.calib.schema import CameraCalib, GroundPlane, Intrinsics
from mcreid.utils.logging import get_logger

logger = get_logger(__name__)

FloatArray = npt.NDArray[np.float64]

APRILTAG_DICT = cv2.aruco.DICT_APRILTAG_36H11

# A quadrilateral whose smallest triangle area falls below this fraction of its
# bounding box is too close to degenerate to fit a stable homography.
_MIN_TRIANGLE_AREA_RATIO = 0.02


def _as_pts2(pts: npt.ArrayLike, name: str) -> FloatArray:
    arr = np.asarray(pts, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{name} must be (N, 2), got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def check_non_degenerate(pts: FloatArray, name: str) -> None:
    """Reject near-collinear point sets, which yield unstable homographies."""
    if pts.shape[0] < 4:
        raise ValueError(f"{name}: need >= 4 points, got {pts.shape[0]}")
    span = pts.max(axis=0) - pts.min(axis=0)
    bbox_area = float(span[0] * span[1])
    if bbox_area <= 0.0:
        raise ValueError(f"{name}: all points collinear (zero-area bounding box)")

    # Largest triangle over the point set must be a meaningful fraction of the bbox.
    # np.cross on 2-D vectors is deprecated in NumPy 2, so the z-component of the
    # cross product is written out directly.
    best = 0.0
    n = pts.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a, b, c = pts[i], pts[j], pts[k]
                u, v = b - a, c - a
                area = 0.5 * abs(float(u[0] * v[1] - u[1] * v[0]))
                best = max(best, area)
    ratio = best / bbox_area
    if ratio < _MIN_TRIANGLE_AREA_RATIO:
        raise ValueError(
            f"{name}: points are near-collinear (max triangle / bbox = {ratio:.4f} "
            f"< {_MIN_TRIANGLE_AREA_RATIO}). Spread the correspondences across the floor."
        )


def _floor_extent(world_pts: FloatArray, margin_m: float) -> tuple[float, float, float, float]:
    lo = world_pts.min(axis=0) - margin_m
    hi = world_pts.max(axis=0) + margin_m
    return (float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1]))


def fit_ground_homography(
    image_pts: npt.ArrayLike,
    world_pts: npt.ArrayLike,
    ransac_thresh_m: float = 0.05,
) -> tuple[FloatArray, float, int]:
    """Fit image->world(Z=0) homography from correspondences.

    Args:
        image_pts: (N, 2) *undistorted* pixel coordinates.
        world_pts: (N, 2) floor coordinates in metres.
        ransac_thresh_m: RANSAC inlier threshold, metres. Ignored for N == 4
            (exact fit, nothing to reject).

    Returns:
        (H, rms_error_m, n_inliers)
    """
    src = _as_pts2(image_pts, "image_pts")
    dst = _as_pts2(world_pts, "world_pts")
    if src.shape[0] != dst.shape[0]:
        raise ValueError(f"correspondence count mismatch: {src.shape[0]} vs {dst.shape[0]}")
    check_non_degenerate(src, "image_pts")
    check_non_degenerate(dst, "world_pts")

    if src.shape[0] == 4:
        raw = cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))
        inliers = np.ones(4, dtype=bool)
    else:
        raw, mask = cv2.findHomography(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh_m
        )
        if raw is None:
            raise RuntimeError("cv2.findHomography failed — correspondences are inconsistent")
        inliers = np.asarray(mask, dtype=bool).ravel()
        if inliers.sum() < 4:
            raise RuntimeError(f"RANSAC kept only {int(inliers.sum())} inliers (need >= 4)")

    H: FloatArray = np.asarray(raw, dtype=np.float64)
    if abs(H[2, 2]) < 1e-12:
        raise RuntimeError("degenerate homography (H[2,2] ~ 0)")
    H = H / H[2, 2]

    projected, valid = apply_homography(H, src)
    if not valid.all():
        raise RuntimeError("some correspondence points map to infinity — degenerate geometry")
    residual = np.linalg.norm(projected[inliers] - dst[inliers], axis=1)
    rms = float(np.sqrt(np.mean(residual**2)))
    logger.info(
        "ground homography: %d/%d inliers, RMS = %.4f m (max %.4f m)",
        int(inliers.sum()),
        src.shape[0],
        rms,
        float(residual.max()),
    )
    return H, rms, int(inliers.sum())


def ground_plane_from_correspondences(
    intrinsics: Intrinsics,
    image_pts_raw: npt.ArrayLike,
    world_pts: npt.ArrayLike,
    method: str = "four_point",
    margin_m: float = 1.0,
    ransac_thresh_m: float = 0.05,
) -> GroundPlane:
    """Undistort raw pixels, fit the ground homography, wrap in a `GroundPlane`."""
    if method not in {"four_point", "apriltag", "synthetic"}:
        raise ValueError(f"unknown method {method!r}")

    # A throwaway calib carrying only intrinsics, so undistort_points can be reused.
    src_raw = _as_pts2(image_pts_raw, "image_pts_raw")
    dst = _as_pts2(world_pts, "world_pts")
    stub = CameraCalib(
        camera_id="_stub",
        intrinsics=intrinsics,
        ground=GroundPlane.from_matrix(np.eye(3), "synthetic", 0.0, 4),
    )
    src = undistort_points(stub, src_raw)

    H, rms, n_inliers = fit_ground_homography(src, dst, ransac_thresh_m=ransac_thresh_m)
    # Fail fast rather than shipping a homography that maps the visible floor
    # to the wrong side of the horizon.
    horizon_sign(H, intrinsics.image_size)

    return GroundPlane.from_matrix(
        H=H,
        method=method,  # type: ignore[arg-type]
        rms_error_m=rms,
        n_correspondences=n_inliers,
        floor_extent_m=_floor_extent(dst, margin_m),
    )


@dataclass(frozen=True)
class TagPlacement:
    """Where one AprilTag physically sits on the floor."""

    tag_id: int
    center_xy_m: tuple[float, float]
    size_m: float
    yaw_deg: float = 0.0

    def __post_init__(self) -> None:
        if self.tag_id < 0:
            raise ValueError(f"tag_id must be non-negative, got {self.tag_id}")
        if self.size_m <= 0.0:
            raise ValueError(f"size_m must be positive, got {self.size_m}")

    def world_corners(self) -> FloatArray:
        """(4, 2) world corners in cv2.aruco order (TL, TR, BR, BL)."""
        half = self.size_m / 2.0
        local = np.array(
            [[-half, half], [half, half], [half, -half], [-half, -half]], dtype=np.float64
        )
        theta = math.radians(self.yaw_deg)
        rot = np.array(
            [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
            dtype=np.float64,
        )
        return local @ rot.T + np.asarray(self.center_xy_m, dtype=np.float64)


def detect_apriltags(
    image: npt.NDArray[np.uint8],
    dictionary_id: int = APRILTAG_DICT,
) -> dict[int, FloatArray]:
    """Detect AprilTags. Returns {tag_id: (4, 2) image corners in (TL,TR,BR,BL)}."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_id)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    corners, ids, _rejected = detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return {}
    out: dict[int, FloatArray] = {}
    for tag_corners, tag_id in zip(corners, ids.ravel(), strict=True):
        out[int(tag_id)] = np.asarray(tag_corners, dtype=np.float64).reshape(4, 2)
    return out


def correspondences_from_tags(
    detections: dict[int, FloatArray],
    placements: list[TagPlacement],
    min_tags: int = 2,
) -> tuple[FloatArray, FloatArray]:
    """Match detected tags to their known floor placements.

    Returns (image_pts, world_pts), 4 corner correspondences per matched tag.
    """
    by_id = {p.tag_id: p for p in placements}
    dupes = len(placements) - len(by_id)
    if dupes:
        raise ValueError("duplicate tag_id in placements")

    matched = sorted(set(detections) & set(by_id))
    missing = sorted(set(by_id) - set(detections))
    if len(matched) < min_tags:
        raise RuntimeError(
            f"only {len(matched)} of {len(by_id)} placed tags detected "
            f"(missing {missing}); need >= {min_tags}. Re-shoot with better lighting "
            "or larger tags."
        )
    if missing:
        logger.warning("tags placed but not detected in this view: %s", missing)

    image_pts = np.vstack([detections[t] for t in matched])
    world_pts = np.vstack([by_id[t].world_corners() for t in matched])
    return image_pts, world_pts


def ground_plane_from_apriltags(
    intrinsics: Intrinsics,
    image: npt.NDArray[np.uint8],
    placements: list[TagPlacement],
    min_tags: int = 2,
    margin_m: float = 1.0,
    ransac_thresh_m: float = 0.05,
) -> GroundPlane:
    """End-to-end: detect tags in one frame, fit the ground homography."""
    detections = detect_apriltags(image)
    logger.info("detected %d AprilTag(s): %s", len(detections), sorted(detections))
    image_pts, world_pts = correspondences_from_tags(detections, placements, min_tags=min_tags)
    return ground_plane_from_correspondences(
        intrinsics,
        image_pts,
        world_pts,
        method="apriltag",
        margin_m=margin_m,
        ransac_thresh_m=ransac_thresh_m,
    )
