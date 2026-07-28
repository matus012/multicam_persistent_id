"""Image <-> ground-plane projection.

The one subtlety that bites every homography-based multi-view tracker: a
homography maps the image's *horizon line* to infinity. Pixels on the far side
of the horizon still produce finite, plausible-looking world coordinates — with
a flipped sign. Feeding those into the fusion stage yields ghost tracks behind
the camera. Every mapping here therefore returns a validity mask and writes NaN
into rejected rows instead of silently returning garbage.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import numpy.typing as npt

from mcreid.calib.schema import CameraCalib
from mcreid.utils.logging import get_logger

logger = get_logger(__name__)

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

# Homogeneous scale below which a point is treated as "on the horizon".
_W_EPS = 1e-9


def _as_pts2(pts: npt.ArrayLike) -> FloatArray:
    """Coerce to an (N, 2) float64 array, accepting a single (2,) point."""
    arr = np.asarray(pts, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"expected (N, 2) points, got shape {arr.shape}")
    return arr


def horizon_sign(H: FloatArray, image_size: tuple[int, int]) -> float:  # noqa: N803
    """Sign of the homogeneous scale on the *visible floor* side of the horizon.

    Reference pixel is the bottom-centre of the image: for any camera mounted
    above the floor and looking at it (the only mount this project supports —
    see capture_guide.md), that pixel images floor in front of the camera.
    """
    width, height = image_size
    ref = np.array([width / 2.0, height - 1.0, 1.0], dtype=np.float64)
    scale = float(H[2] @ ref)
    if abs(scale) < _W_EPS:
        raise ValueError(
            "bottom-centre pixel lies on the horizon line — the ground homography is "
            "degenerate or the camera is not looking at the floor"
        )
    return math.copysign(1.0, scale)


def apply_homography(
    H: FloatArray,  # noqa: N803
    pts: npt.ArrayLike,
    valid_sign: float | None = None,
) -> tuple[FloatArray, BoolArray]:
    """Apply a 3x3 homography to (N, 2) points.

    Args:
        H: 3x3 homography.
        pts: (N, 2) source points.
        valid_sign: if given, points whose homogeneous scale does not carry this
            sign are marked invalid (wrong side of the horizon).

    Returns:
        (out, valid) where ``out`` is (N, 2) with NaN in invalid rows.
    """
    src = _as_pts2(pts)
    H = np.asarray(H, dtype=np.float64)
    if H.shape != (3, 3):
        raise ValueError(f"H must be 3x3, got {H.shape}")

    homo = np.hstack([src, np.ones((src.shape[0], 1), dtype=np.float64)])
    proj = homo @ H.T  # (N, 3)
    scale = proj[:, 2]

    valid = np.abs(scale) > _W_EPS
    if valid_sign is not None:
        valid &= np.sign(scale) == math.copysign(1.0, valid_sign)

    out = np.full((src.shape[0], 2), np.nan, dtype=np.float64)
    safe = valid & (np.abs(scale) > _W_EPS)
    out[safe] = proj[safe, :2] / scale[safe, None]
    return out, valid


def undistort_points(cam: CameraCalib, pts_img: npt.ArrayLike) -> FloatArray:
    """Undistort raw pixel coords into the ideal-pinhole pixel frame the ground
    homography was fitted in. No-op when the camera has negligible distortion."""
    src = _as_pts2(pts_img)
    if not cam.undistort_maps_needed():
        return src
    K = cam.intrinsics.K
    undist = cv2.undistortPoints(
        src.reshape(-1, 1, 2).astype(np.float64), K, cam.intrinsics.dist, P=K
    )
    return np.asarray(undist, dtype=np.float64).reshape(-1, 2)


def distort_points(cam: CameraCalib, pts_img: npt.ArrayLike) -> FloatArray:
    """Inverse of :func:`undistort_points`: ideal-pinhole pixels -> raw pixels."""
    src = _as_pts2(pts_img)
    if not cam.undistort_maps_needed():
        return src
    K = cam.intrinsics.K
    # Back-project to normalised camera coords on Z=1, then re-apply distortion.
    normalised = np.linalg.solve(K, np.hstack([src, np.ones((src.shape[0], 1))]).T).T
    obj = np.hstack([normalised[:, :2], np.ones((src.shape[0], 1))]).astype(np.float64)
    zero = np.zeros(3, dtype=np.float64)
    projected, _ = cv2.projectPoints(obj, zero, zero, K, cam.intrinsics.dist)
    return np.asarray(projected, dtype=np.float64).reshape(-1, 2)


def image_to_ground(
    cam: CameraCalib,
    pts_img: npt.ArrayLike,
    undistort: bool = True,
) -> tuple[FloatArray, BoolArray]:
    """Project image points onto the world floor plane (Z=0), in metres.

    Returns (world_xy, valid). Invalid rows (beyond the horizon) are NaN.
    """
    src = _as_pts2(pts_img)
    ideal = undistort_points(cam, src) if undistort else src
    H = cam.ground.H
    sign = horizon_sign(H, cam.intrinsics.image_size)
    return apply_homography(H, ideal, valid_sign=sign)


def ground_to_image(
    cam: CameraCalib,
    pts_world: npt.ArrayLike,
    distort: bool = True,
) -> tuple[FloatArray, BoolArray]:
    """Project world floor points back into raw image pixels.

    Returns (pixels, valid). Points that would land behind the camera are NaN.
    """
    src = _as_pts2(pts_world)
    H = cam.ground.H
    H_inv = cam.ground.H_inv

    # Valid side in world space = image of the bottom-centre reference pixel.
    width, height = cam.intrinsics.image_size
    ref_world, ref_ok = apply_homography(H, [[width / 2.0, height - 1.0]])
    if not bool(ref_ok[0]):
        raise ValueError("degenerate ground homography: reference pixel maps to infinity")
    ref_scale = float(H_inv[2] @ np.array([ref_world[0, 0], ref_world[0, 1], 1.0]))
    if abs(ref_scale) < _W_EPS:
        raise ValueError("degenerate inverse ground homography")
    sign = math.copysign(1.0, ref_scale)

    ideal, valid = apply_homography(H_inv, src, valid_sign=sign)
    if not distort:
        return ideal, valid

    out = np.full_like(ideal, np.nan)
    if valid.any():
        out[valid] = distort_points(cam, ideal[valid])
    return out, valid


def feet_point(bbox_xyxy: npt.ArrayLike) -> FloatArray:
    """Ground-contact point of a person box: bottom-centre.

    Args:
        bbox_xyxy: (N, 4) or (4,) boxes as (x1, y1, x2, y2).

    Returns:
        (N, 2) pixel coordinates.
    """
    boxes = np.asarray(bbox_xyxy, dtype=np.float64)
    if boxes.ndim == 1:
        boxes = boxes.reshape(1, -1)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"expected (N, 4) xyxy boxes, got shape {boxes.shape}")
    if np.any(boxes[:, 2] < boxes[:, 0]) or np.any(boxes[:, 3] < boxes[:, 1]):
        raise ValueError("xyxy boxes must satisfy x2 >= x1 and y2 >= y1")
    return np.stack([(boxes[:, 0] + boxes[:, 2]) * 0.5, boxes[:, 3]], axis=1)


def ground_covariance(
    cam: CameraCalib,
    pts_img: npt.ArrayLike,
    sigma_px: float = 4.0,
    delta_px: float = 1.0,
    model_sigma_m: float = 0.15,
) -> FloatArray:
    """Propagate isotropic pixel noise to ground-plane covariance, per point.

    A foot point 8 m away carries an order of magnitude more world-space
    uncertainty than one at 2 m; a plain Euclidean association gate over-trusts
    the far one. The Jacobian is estimated by central differences (cheap, and
    exact enough at these scales).

    ``model_sigma_m`` adds an isotropic floor in *world* units, and it is not
    optional in practice: the dominant error in a bbox-derived ground point is
    not pixel noise but the fact that the bottom-centre of a detection box is
    only an approximation of where the person actually touches the floor
    (perspective, footwear, shadows, boxes truncated at the frame edge). On the
    toy generator the pixel-noise-only model underestimates the true projection
    error by roughly 5x, which collapses the Mahalanobis gate and shatters
    tracks into one ID per frame. Measure it before changing it.

    Returns:
        (N, 2, 2) covariance matrices in m^2. Rows for invalid points are NaN.
    """
    if sigma_px <= 0.0:
        raise ValueError(f"sigma_px must be positive, got {sigma_px}")
    if model_sigma_m < 0.0:
        raise ValueError(f"model_sigma_m must be non-negative, got {model_sigma_m}")
    src = _as_pts2(pts_img)
    n = src.shape[0]

    base, valid = image_to_ground(cam, src)
    jac = np.zeros((n, 2, 2), dtype=np.float64)
    for axis in range(2):
        step = np.zeros((1, 2), dtype=np.float64)
        step[0, axis] = delta_px
        plus, ok_p = image_to_ground(cam, src + step)
        minus, ok_m = image_to_ground(cam, src - step)
        valid &= ok_p & ok_m
        jac[:, :, axis] = (plus - minus) / (2.0 * delta_px)

    cov = (sigma_px**2) * jac @ np.transpose(jac, (0, 2, 1))
    if model_sigma_m > 0.0:
        cov += (model_sigma_m**2) * np.eye(2, dtype=np.float64)[None, :, :]
    cov[~valid] = np.nan
    cov[~np.isfinite(base).all(axis=1)] = np.nan
    return cov
