"""Analytic pinhole camera used to synthesise multi-view toy sequences.

World convention (matches `mcreid.calib.schema`): +X / +Y span the floor, +Z is
up, Z=0 is the floor, metres.

Camera convention: x_cam = right, y_cam = down, z_cam = forward (OpenCV).
``yaw_deg`` rotates the optical axis about world +Z from the +X axis;
``pitch_deg`` is positive *downwards*, so a ceiling-mounted camera looking at
the floor has a positive pitch.

The ground homography this class emits is the exact analytic inverse of its own
projection, which is what makes calibration round-trip tests meaningful: any
error the test sees comes from `mcreid.calib`, not from the fixture.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from mcreid.calib.schema import CameraCalib, GroundPlane, Intrinsics

FloatArray = npt.NDArray[np.float64]

_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)


@dataclass(frozen=True)
class VirtualCamera:
    """A posed pinhole camera with a known-exact ground homography."""

    camera_id: str
    position_m: tuple[float, float, float]
    yaw_deg: float
    pitch_deg: float
    hfov_deg: float = 70.0
    image_size: tuple[int, int] = (1280, 720)
    dist_coeffs: tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if self.position_m[2] <= 0.0:
            raise ValueError(f"camera must be above the floor, got z={self.position_m[2]}")
        if not 0.0 < self.hfov_deg < 179.0:
            raise ValueError(f"hfov_deg out of range: {self.hfov_deg}")
        if not 0.0 < self.pitch_deg < 90.0:
            raise ValueError(
                f"pitch_deg must be in (0, 90) — the camera has to look down at the floor, "
                f"got {self.pitch_deg}"
            )
        width, height = self.image_size
        if width <= 0 or height <= 0:
            raise ValueError(f"invalid image_size {self.image_size}")

    # --- geometry ---------------------------------------------------------

    @property
    def K(self) -> FloatArray:  # noqa: N802
        width, height = self.image_size
        fx = (width / 2.0) / math.tan(math.radians(self.hfov_deg) / 2.0)
        # Square pixels.
        return np.array(
            [[fx, 0.0, width / 2.0], [0.0, fx, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64
        )

    @property
    def R(self) -> FloatArray:  # noqa: N802
        """World -> camera rotation; rows are (right, down, forward)."""
        yaw = math.radians(self.yaw_deg)
        pitch = math.radians(self.pitch_deg)
        forward = np.array(
            [math.cos(yaw) * math.cos(pitch), math.sin(yaw) * math.cos(pitch), -math.sin(pitch)],
            dtype=np.float64,
        )
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, _UP)
        norm = np.linalg.norm(right)
        if norm < 1e-9:  # pragma: no cover - excluded by the pitch validator
            raise ValueError("camera looks straight down; yaw is undefined")
        right /= norm
        down = np.cross(forward, right)
        return np.stack([right, down, forward], axis=0)

    @property
    def C(self) -> FloatArray:  # noqa: N802
        return np.asarray(self.position_m, dtype=np.float64)

    @property
    def P(self) -> FloatArray:  # noqa: N802
        """3x4 projection matrix, world -> homogeneous pixels."""
        R = self.R
        t = -R @ self.C
        return self.K @ np.hstack([R, t.reshape(3, 1)])

    @property
    def H_world2img(self) -> FloatArray:  # noqa: N802
        """Exact ground-plane (Z=0) homography, world metres -> pixels."""
        P = self.P
        H = P[:, [0, 1, 3]]
        return H / H[2, 2]

    @property
    def H_img2world(self) -> FloatArray:  # noqa: N802
        H = np.linalg.inv(self.H_world2img)
        return H / H[2, 2]

    # --- projection -------------------------------------------------------

    def project(self, points_world: npt.ArrayLike) -> tuple[FloatArray, npt.NDArray[np.bool_]]:
        """Project (N, 3) world points. Returns (pixels (N,2), in_front_of_camera)."""
        pts = np.asarray(points_world, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts.reshape(1, -1)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"expected (N, 3) world points, got {pts.shape}")

        cam = (pts - self.C) @ self.R.T
        depth = cam[:, 2]
        in_front = depth > 1e-6

        out = np.full((pts.shape[0], 2), np.nan, dtype=np.float64)
        K = self.K
        safe = cam[in_front]
        out[in_front] = np.stack(
            [
                K[0, 0] * safe[:, 0] / safe[:, 2] + K[0, 2],
                K[1, 1] * safe[:, 1] / safe[:, 2] + K[1, 2],
            ],
            axis=1,
        )
        return out, in_front

    def person_bbox(
        self,
        foot_xy: npt.ArrayLike,
        height_m: float,
        width_m: float = 0.55,
    ) -> FloatArray | None:
        """Axis-aligned image box for an upright person standing at ``foot_xy``.

        Returns None when the person is behind the camera or fully outside the
        frame. Boxes that straddle the border are clipped (mirroring what a real
        detector reports).
        """
        foot = np.asarray(foot_xy, dtype=np.float64).reshape(2)
        pillar = np.array([[foot[0], foot[1], 0.0], [foot[0], foot[1], height_m]])
        pix, in_front = self.project(pillar)
        if not in_front.all():
            return None

        foot_px, head_px = pix[0], pix[1]
        px_height = abs(foot_px[1] - head_px[1])
        if px_height < 1.0:
            return None
        # Width from the true metric width, scaled by the same depth as the body.
        px_width = px_height * (width_m / height_m)
        cx = 0.5 * (foot_px[0] + head_px[0])
        box = np.array(
            [
                cx - px_width / 2.0,
                min(foot_px[1], head_px[1]),
                cx + px_width / 2.0,
                max(foot_px[1], head_px[1]),
            ],
            dtype=np.float64,
        )

        img_w, img_h = self.image_size
        if box[2] <= 0 or box[0] >= img_w or box[3] <= 0 or box[1] >= img_h:
            return None
        clipped = np.array(
            [
                max(box[0], 0.0),
                max(box[1], 0.0),
                min(box[2], float(img_w)),
                min(box[3], float(img_h)),
            ]
        )
        if clipped[2] - clipped[0] < 2.0 or clipped[3] - clipped[1] < 4.0:
            return None
        return clipped

    def visible_fraction(self, foot_xy: npt.ArrayLike, height_m: float) -> float:
        """Fraction of the person's box that survives frame clipping (0..1)."""
        foot = np.asarray(foot_xy, dtype=np.float64).reshape(2)
        pillar = np.array([[foot[0], foot[1], 0.0], [foot[0], foot[1], height_m]])
        pix, in_front = self.project(pillar)
        if not in_front.all():
            return 0.0
        full_height = abs(pix[0, 1] - pix[1, 1])
        box = self.person_bbox(foot, height_m)
        if box is None or full_height < 1e-6:
            return 0.0
        return float(np.clip((box[3] - box[1]) / full_height, 0.0, 1.0))

    # --- calibration export ----------------------------------------------

    def to_calib(
        self,
        floor_extent_m: tuple[float, float, float, float] | None = None,
    ) -> CameraCalib:
        """Ground-truth `CameraCalib` — exactly consistent with this camera's optics."""
        width, height = self.image_size
        K = self.K
        intr = Intrinsics(
            fx=float(K[0, 0]),
            fy=float(K[1, 1]),
            cx=float(K[0, 2]),
            cy=float(K[1, 2]),
            dist_coeffs=list(self.dist_coeffs),
            image_width=width,
            image_height=height,
            rms_reproj_px=0.0,
            n_views=0,
        )
        ground = GroundPlane.from_matrix(
            H=self.H_img2world,
            method="synthetic",
            rms_error_m=0.0,
            n_correspondences=4,
            floor_extent_m=floor_extent_m,
        )
        return CameraCalib(
            camera_id=self.camera_id,
            intrinsics=intr,
            ground=ground,
            height_m=float(self.position_m[2]),
            notes=f"synthetic camera yaw={self.yaw_deg}deg pitch={self.pitch_deg}deg",
        )
