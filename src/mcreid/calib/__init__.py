"""Camera calibration: intrinsics, ground-plane homography, projection."""

from mcreid.calib.geometry import (
    apply_homography,
    distort_points,
    feet_point,
    ground_covariance,
    ground_to_image,
    horizon_sign,
    image_to_ground,
    undistort_points,
)
from mcreid.calib.homography import (
    TagPlacement,
    correspondences_from_tags,
    detect_apriltags,
    fit_ground_homography,
    ground_plane_from_apriltags,
    ground_plane_from_correspondences,
)
from mcreid.calib.intrinsics import (
    CheckerboardSpec,
    calibrate_intrinsics_from_corners,
    calibrate_intrinsics_from_dir,
    find_checkerboard,
)
from mcreid.calib.schema import SCHEMA_VERSION, CameraCalib, GroundPlane, Intrinsics, RigCalib

__all__ = [
    "SCHEMA_VERSION",
    "CameraCalib",
    "CheckerboardSpec",
    "GroundPlane",
    "Intrinsics",
    "RigCalib",
    "TagPlacement",
    "apply_homography",
    "calibrate_intrinsics_from_corners",
    "calibrate_intrinsics_from_dir",
    "correspondences_from_tags",
    "detect_apriltags",
    "distort_points",
    "feet_point",
    "find_checkerboard",
    "fit_ground_homography",
    "ground_covariance",
    "ground_plane_from_apriltags",
    "ground_plane_from_correspondences",
    "ground_to_image",
    "horizon_sign",
    "image_to_ground",
    "undistort_points",
]
