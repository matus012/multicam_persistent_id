"""Serializable calibration schema (pydantic) — the `calib.json` contract.

One `CameraCalib` per physical camera; a `RigCalib` bundles the N cameras that
share a single world frame. Matrices are stored as nested lists so the JSON is
human-diffable; ``.matrix`` properties hand back numpy views.

World frame convention (LOCKED):
    - Right-handed, metres.
    - Z = 0 is the floor plane. X/Y span the floor.
    - The ground homography maps *image pixels* -> *world (X, Y) on Z=0*.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Mat3 = Annotated[list[list[float]], Field(min_length=3, max_length=3)]
FloatArray = npt.NDArray[np.float64]

SCHEMA_VERSION = 1


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Intrinsics(_Base):
    """Pinhole intrinsics + Brown-Conrady distortion, as produced by
    ``cv2.calibrateCamera``."""

    fx: float = Field(gt=0.0)
    fy: float = Field(gt=0.0)
    cx: float = Field(gt=0.0)
    cy: float = Field(gt=0.0)
    # (k1, k2, p1, p2, k3) — OpenCV order.
    dist_coeffs: list[float] = Field(min_length=5, max_length=5)
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    rms_reproj_px: float = Field(ge=0.0, description="cv2.calibrateCamera RMS, pixels")
    n_views: int = Field(ge=0, description="checkerboard views used")

    @field_validator("dist_coeffs")
    @classmethod
    def _finite_dist(cls, v: list[float]) -> list[float]:
        if not all(np.isfinite(v)):
            raise ValueError("dist_coeffs must be finite")
        return v

    @model_validator(mode="after")
    def _principal_point_inside(self) -> Intrinsics:
        if not (0.0 < self.cx < self.image_width and 0.0 < self.cy < self.image_height):
            raise ValueError(
                f"principal point ({self.cx:.1f}, {self.cy:.1f}) outside image "
                f"{self.image_width}x{self.image_height} — calibration almost certainly diverged"
            )
        return self

    @property
    def K(self) -> FloatArray:  # noqa: N802 - camera-matrix convention
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @property
    def dist(self) -> FloatArray:
        return np.asarray(self.dist_coeffs, dtype=np.float64)

    @property
    def image_size(self) -> tuple[int, int]:
        """(width, height) — OpenCV order."""
        return (self.image_width, self.image_height)

    @classmethod
    def from_matrices(
        cls,
        K: FloatArray,  # noqa: N803
        dist: FloatArray,
        image_size: tuple[int, int],
        rms_reproj_px: float,
        n_views: int,
    ) -> Intrinsics:
        K = np.asarray(K, dtype=np.float64)
        if K.shape != (3, 3):
            raise ValueError(f"K must be 3x3, got {K.shape}")
        flat = np.asarray(dist, dtype=np.float64).ravel()
        if flat.size < 5:
            flat = np.pad(flat, (0, 5 - flat.size))
        return cls(
            fx=float(K[0, 0]),
            fy=float(K[1, 1]),
            cx=float(K[0, 2]),
            cy=float(K[1, 2]),
            dist_coeffs=[float(x) for x in flat[:5]],
            image_width=int(image_size[0]),
            image_height=int(image_size[1]),
            rms_reproj_px=float(rms_reproj_px),
            n_views=int(n_views),
        )


class GroundPlane(_Base):
    """Image <-> floor (Z=0) homography.

    ``H_img2world`` maps homogeneous *undistorted* pixel coords to world metres.
    Stored normalised so H[2, 2] == 1.
    """

    H_img2world: Mat3
    method: Literal["apriltag", "four_point", "synthetic"]
    rms_error_m: float = Field(ge=0.0, description="round-trip / correspondence residual, metres")
    n_correspondences: int = Field(ge=4)
    # Floor rectangle the BEV canvas should cover: (x_min, y_min, x_max, y_max) metres.
    floor_extent_m: tuple[float, float, float, float] | None = None

    @field_validator("H_img2world")
    @classmethod
    def _valid_homography(cls, v: Mat3) -> Mat3:
        H = np.asarray(v, dtype=np.float64)
        if H.shape != (3, 3):
            raise ValueError(f"homography must be 3x3, got {H.shape}")
        if not np.all(np.isfinite(H)):
            raise ValueError("homography contains non-finite entries")
        det = float(np.linalg.det(H))
        if abs(det) < 1e-12:
            raise ValueError(f"homography is singular (det={det:.3e})")
        return v

    @model_validator(mode="after")
    def _extent_ordered(self) -> GroundPlane:
        if self.floor_extent_m is not None:
            x0, y0, x1, y1 = self.floor_extent_m
            if not (x1 > x0 and y1 > y0):
                raise ValueError(f"floor_extent_m must be (x_min, y_min, x_max, y_max), got {self.floor_extent_m}")
        return self

    @property
    def H(self) -> FloatArray:  # noqa: N802
        return np.asarray(self.H_img2world, dtype=np.float64)

    @property
    def H_inv(self) -> FloatArray:  # noqa: N802
        return np.linalg.inv(self.H)

    @classmethod
    def from_matrix(
        cls,
        H: FloatArray,  # noqa: N803
        method: Literal["apriltag", "four_point", "synthetic"],
        rms_error_m: float,
        n_correspondences: int,
        floor_extent_m: tuple[float, float, float, float] | None = None,
    ) -> GroundPlane:
        H = np.asarray(H, dtype=np.float64)
        if H.shape != (3, 3):
            raise ValueError(f"H must be 3x3, got {H.shape}")
        if abs(H[2, 2]) < 1e-12:
            raise ValueError("cannot normalise homography with H[2,2] ~ 0")
        H = H / H[2, 2]
        return cls(
            H_img2world=[[float(x) for x in row] for row in H],
            method=method,
            rms_error_m=float(rms_error_m),
            n_correspondences=int(n_correspondences),
            floor_extent_m=floor_extent_m,
        )


class CameraCalib(_Base):
    """Everything the fusion stage needs about one camera."""

    camera_id: str = Field(min_length=1)
    intrinsics: Intrinsics
    ground: GroundPlane
    # Optional metadata — mount description, so the capture is reproducible.
    height_m: float | None = Field(default=None, gt=0.0)
    notes: str = ""

    def undistort_maps_needed(self) -> bool:
        """True when distortion is non-negligible and points must be undistorted
        before the ground homography is applied."""
        return bool(np.any(np.abs(self.intrinsics.dist) > 1e-6))


class RigCalib(_Base):
    """N cameras sharing one world frame."""

    schema_version: int = SCHEMA_VERSION
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    world_notes: str = "Z=0 floor plane, metres, right-handed."
    cameras: list[CameraCalib] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_ids(self) -> RigCalib:
        ids = [c.camera_id for c in self.cameras]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"duplicate camera_id(s): {sorted(dupes)}")
        return self

    @property
    def camera_ids(self) -> list[str]:
        return [c.camera_id for c in self.cameras]

    def get(self, camera_id: str) -> CameraCalib:
        for cam in self.cameras:
            if cam.camera_id == camera_id:
                return cam
        raise KeyError(f"camera_id {camera_id!r} not in rig ({self.camera_ids})")

    def floor_extent(self) -> tuple[float, float, float, float]:
        """Union of per-camera floor extents — the BEV canvas bounds."""
        extents = [c.ground.floor_extent_m for c in self.cameras if c.ground.floor_extent_m]
        if not extents:
            raise ValueError("no camera declares floor_extent_m; cannot size the BEV canvas")
        return (
            min(e[0] for e in extents),
            min(e[1] for e in extents),
            max(e[2] for e in extents),
            max(e[3] for e in extents),
        )

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> RigCalib:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"calibration file not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        version = data.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"calib schema version mismatch: file={version}, expected={SCHEMA_VERSION}"
            )
        return cls.model_validate(data)
