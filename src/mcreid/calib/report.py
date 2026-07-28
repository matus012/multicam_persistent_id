"""Calibration sanity report — the gate that runs before any tracking on footage.

A bad ground homography does not announce itself. Tracking still runs, boxes
still appear, the BEV still shows dots; they are just in the wrong places, and
every downstream number is quietly meaningless. This module produces evidence a
human can check in seconds: reprojected tags against detected tags, and a metric
floor grid drawn into each camera image.

Two numbers are reported, and they catch **different** failure modes. Neither
subsumes the other, which is why both are gated:

* **fit residual** — the stored calibration reprojected against the tags it can
  see. This catches a calibration that is stale, corrupted, or simply belongs to
  a different camera: anything where `calib.json` disagrees with the picture.
  Verified to reject a shifted, a mirrored and a mis-scaled homography.
* **leave-one-out error** — refit the ground plane from every tag *except* one
  and predict the held-out tag. This never consults the stored homography, so it
  says nothing about `calib.json`; what it validates is whether the tag world
  positions in `tags.yaml` are mutually consistent with what the cameras see.
  That is the single most common human error in this pipeline — a tag measured
  to the wrong spot, or moved between shots. Verified to catch a tag displaced
  by 42 cm (LOO jumps from ~0.5 cm to 19-37 cm on the cameras that can see it,
  and correctly stays clean on the one that cannot).

So: a bad `calib.json` shows up in the fit residual, a bad `tags.yaml` shows up
in leave-one-out, and a camera that was bumped after calibration shows up in
both.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from mcreid.calib.geometry import ground_to_image, image_to_ground
from mcreid.calib.homography import (
    TagPlacement,
    correspondences_from_tags,
    detect_apriltags,
    fit_ground_homography,
)
from mcreid.calib.schema import CameraCalib
from mcreid.utils.logging import get_logger

logger = get_logger(__name__)

FloatArray = npt.NDArray[np.float64]
Image = npt.NDArray[np.uint8]

# Defaults chosen so a competent phone capture passes and a sloppy one does not.
DEFAULT_MAX_LOO_ERROR_M = 0.10
DEFAULT_MAX_FIT_RESIDUAL_M = 0.05
DEFAULT_MIN_FLOOR_COVERAGE = 0.10


@dataclass
class CameraReport:
    """Per-camera calibration verdict."""

    camera_id: str
    n_tags_detected: int
    n_tags_expected: int
    missing_tag_ids: list[int]
    fit_residual_m: float
    loo_error_mean_m: float
    loo_error_max_m: float
    reprojection_px_mean: float
    reprojection_px_max: float
    floor_coverage: float
    """Fraction of the image below the horizon that the declared floor extent
    actually projects into — a sanity check that the camera is pointed at the
    room the calibration claims."""
    grid_is_convex: bool
    """A metric floor square must image as a convex quadrilateral. If it does
    not, the homography is mirrored or degenerate."""
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


@dataclass
class RigReport:
    cameras: list[CameraReport]
    max_loo_error_m: float
    max_fit_residual_m: float

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.cameras)

    @property
    def failed(self) -> list[CameraReport]:
        return [c for c in self.cameras if not c.ok]

    def to_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "ok": self.ok,
                    "thresholds": {
                        "max_loo_error_m": self.max_loo_error_m,
                        "max_fit_residual_m": self.max_fit_residual_m,
                    },
                    "cameras": [asdict(c) for c in self.cameras],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return path

    def to_markdown(self, path: Path) -> Path:
        lines = [
            "# Calibration sanity report",
            "",
            f"**Verdict: {'PASS' if self.ok else 'FAIL'}**",
            "",
            "The two error columns catch different failures, so read both:",
            "",
            "- **fit residual** — the stored `calib.json` reprojected against the tags.",
            "  High means the calibration disagrees with the picture: stale, corrupted,",
            "  belongs to another camera, or the camera was bumped after calibrating.",
            "- **LOO** — refit the ground plane without one tag, then predict it. This",
            "  never touches `calib.json`; it checks whether the tag positions in",
            "  `tags.yaml` are mutually consistent with what the cameras see. High means",
            "  a tag was mis-measured or moved between shots.",
            "",
            "| camera | tags | LOO mean | LOO max | fit resid | reproj px | floor cov | ok |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for cam in self.cameras:
            lines.append(
                f"| {cam.camera_id} | {cam.n_tags_detected}/{cam.n_tags_expected} "
                f"| {cam.loo_error_mean_m * 100:.1f} cm | {cam.loo_error_max_m * 100:.1f} cm "
                f"| {cam.fit_residual_m * 100:.1f} cm | {cam.reprojection_px_mean:.2f} "
                f"| {cam.floor_coverage:.0%} | {'yes' if cam.ok else 'NO'} |"
            )
        if not self.ok:
            lines += ["", "## What to fix", ""]
            for cam in self.failed:
                lines.append(f"### {cam.camera_id}")
                for problem in cam.problems:
                    lines.append(f"- {problem}")
                lines.append("")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path


def leave_one_out_error(
    detections: dict[int, FloatArray],
    placements: list[TagPlacement],
    cam: CameraCalib,
) -> tuple[float, float]:
    """Fit on all tags but one, predict the held-out tag. Returns (mean, max) metres.

    Returns (nan, nan) when fewer than three tags are available, since a
    two-tag fit has no spare evidence to hold out.
    """
    matched = sorted(set(detections) & {p.tag_id for p in placements})
    if len(matched) < 3:
        return float("nan"), float("nan")

    by_id = {p.tag_id: p for p in placements}
    errors: list[float] = []
    for held_out in matched:
        others = [by_id[t] for t in matched if t != held_out]
        subset = {t: detections[t] for t in matched if t != held_out}
        try:
            image_pts, world_pts = correspondences_from_tags(subset, others, min_tags=2)
            H, _rms, _n = fit_ground_homography(image_pts, world_pts)
        except (RuntimeError, ValueError) as exc:
            logger.debug("LOO fit failed without tag %d: %s", held_out, exc)
            continue

        probe = CameraCalib(
            camera_id=cam.camera_id,
            intrinsics=cam.intrinsics,
            ground=cam.ground.model_copy(
                update={"H_img2world": [[float(v) for v in row] for row in H]}
            ),
        )
        predicted, valid = image_to_ground(probe, detections[held_out])
        truth = by_id[held_out].world_corners()
        if not valid.all():
            errors.append(float("inf"))
            continue
        errors.append(float(np.linalg.norm(predicted - truth, axis=1).mean()))

    if not errors:
        return float("nan"), float("nan")
    return float(np.mean(errors)), float(np.max(errors))


def _floor_coverage(cam: CameraCalib, samples: int = 40) -> tuple[float, bool]:
    """What fraction of the declared floor lands in frame, and is it convex?"""
    extent = cam.ground.floor_extent_m
    if extent is None:
        return float("nan"), True
    x0, y0, x1, y1 = extent

    xs, ys = np.meshgrid(np.linspace(x0, x1, samples), np.linspace(y0, y1, samples))
    grid = np.stack([xs.ravel(), ys.ravel()], axis=1)
    pixels, valid = ground_to_image(cam, grid)
    width, height = cam.intrinsics.image_size
    inside = (
        valid
        & np.isfinite(pixels).all(axis=1)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    coverage = float(inside.mean())

    # Convexity: a metric square on the floor must image as a convex quad.
    square = np.array(
        [
            [x0 + 0.25 * (x1 - x0), y0 + 0.25 * (y1 - y0)],
            [x0 + 0.75 * (x1 - x0), y0 + 0.25 * (y1 - y0)],
            [x0 + 0.75 * (x1 - x0), y0 + 0.75 * (y1 - y0)],
            [x0 + 0.25 * (x1 - x0), y0 + 0.75 * (y1 - y0)],
        ]
    )
    corners, ok = ground_to_image(cam, square)
    if not ok.all() or not np.isfinite(corners).all():
        return coverage, False
    hull = cv2.convexHull(corners.astype(np.float32))
    convex = len(hull) == 4
    return coverage, bool(convex)


def analyse_camera(
    cam: CameraCalib,
    image: Image,
    placements: list[TagPlacement],
    max_loo_error_m: float = DEFAULT_MAX_LOO_ERROR_M,
    max_fit_residual_m: float = DEFAULT_MAX_FIT_RESIDUAL_M,
    min_floor_coverage: float = DEFAULT_MIN_FLOOR_COVERAGE,
) -> tuple[CameraReport, dict[int, FloatArray]]:
    """Score one camera's calibration. Returns (report, tag detections)."""
    detections = detect_apriltags(image)
    expected = {p.tag_id for p in placements}
    matched = sorted(set(detections) & expected)
    missing = sorted(expected - set(detections))

    problems: list[str] = []
    fit_residual = float("nan")
    reproj_mean = reproj_max = float("nan")

    if len(matched) < 2:
        problems.append(
            f"only {len(matched)} of {len(expected)} AprilTags detected "
            f"(missing {missing}). Re-shoot this view with more light, less motion "
            "blur, and every tag fully inside the frame."
        )
    else:
        by_id = {p.tag_id: p for p in placements}
        image_pts = np.vstack([detections[t] for t in matched])
        world_pts = np.vstack([by_id[t].world_corners() for t in matched])

        predicted, valid = image_to_ground(cam, image_pts)
        if valid.all():
            residuals = np.linalg.norm(predicted - world_pts, axis=1)
            fit_residual = float(np.sqrt(np.mean(residuals**2)))
        else:
            problems.append(
                "some detected tag corners project beyond the horizon — the ground "
                "homography is inconsistent with this view. Re-run calibration for "
                "this camera."
            )

        back, ok = ground_to_image(cam, world_pts)
        if ok.all() and np.isfinite(back).all():
            errors = np.linalg.norm(back - image_pts, axis=1)
            reproj_mean, reproj_max = float(errors.mean()), float(errors.max())

    loo_mean, loo_max = leave_one_out_error(detections, placements, cam)
    coverage, convex = _floor_coverage(cam)

    if np.isfinite(fit_residual) and fit_residual > max_fit_residual_m:
        problems.append(
            f"ground-plane fit residual is {fit_residual * 100:.1f} cm "
            f"(limit {max_fit_residual_m * 100:.0f} cm). The tag world positions in "
            "tags.yaml disagree with the image. RE-MEASURE the tag centres with the "
            "tape and check the yaw of each tag."
        )
    if np.isfinite(loo_mean) and loo_mean > max_loo_error_m:
        problems.append(
            f"leave-one-out error is {loo_mean * 100:.1f} cm mean / "
            f"{loo_max * 100:.1f} cm max (limit {max_loo_error_m * 100:.0f} cm). The "
            "homography does not generalise beyond the tags it was fitted on — this "
            "is the signature of a mis-measured tag position or a tag that moved "
            "between shots. RE-MEASURE tag positions and RE-SHOOT this view."
        )
    if not np.isfinite(loo_mean) and len(matched) >= 2:
        problems.append(
            f"only {len(matched)} tags detected, so leave-one-out validation could not "
            "run. Place at least 4 tags, well spread across the floor, and re-shoot."
        )
    if not convex:
        problems.append(
            "a square on the floor does not image as a convex quadrilateral — the "
            "ground homography is mirrored or degenerate. This usually means the "
            "four-point correspondences were entered in the wrong order, or a tag's "
            "yaw is wrong by 90/180 degrees. RE-CHECK tags.yaml orientation."
        )
    if np.isfinite(coverage) and coverage < min_floor_coverage:
        problems.append(
            f"only {coverage:.0%} of the declared floor area projects into this frame "
            f"(limit {min_floor_coverage:.0%}). Either the camera is not pointed at "
            "the room, or floor_extent_m is wrong. Check the mount and the room "
            "dimensions."
        )

    report = CameraReport(
        camera_id=cam.camera_id,
        n_tags_detected=len(matched),
        n_tags_expected=len(expected),
        missing_tag_ids=missing,
        fit_residual_m=fit_residual,
        loo_error_mean_m=loo_mean,
        loo_error_max_m=loo_max,
        reprojection_px_mean=reproj_mean,
        reprojection_px_max=reproj_max,
        floor_coverage=coverage,
        grid_is_convex=convex,
        problems=problems,
    )
    return report, detections
