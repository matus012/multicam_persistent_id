"""Calibration overlays: floor grid and tag reprojection, drawn into a camera view.

These images are the human-checkable half of the calibration gate. A number can
be borderline; a grid that visibly slides off the floor cannot be argued with.
"""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt

from mcreid.calib.geometry import ground_to_image
from mcreid.calib.homography import TagPlacement
from mcreid.calib.schema import CameraCalib
from mcreid.viz.palette import TEXT_COLOR

Image = npt.NDArray[np.uint8]
FloatArray = npt.NDArray[np.float64]

GRID_COLOR = (0, 220, 220)
AXIS_X_COLOR = (60, 60, 255)
AXIS_Y_COLOR = (60, 255, 60)
DETECTED_COLOR = (0, 255, 0)
REPROJECTED_COLOR = (0, 0, 255)
_MAX_PIXEL = 1e4


def _polyline(
    canvas: Image,
    points: FloatArray,
    valid: npt.NDArray[np.bool_],
    colour: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    for i in range(1, len(points)):
        if not (valid[i - 1] and valid[i]):
            continue
        p0, p1 = points[i - 1], points[i]
        if not (np.isfinite(p0).all() and np.isfinite(p1).all()):
            continue
        if max(abs(p0).max(), abs(p1).max()) > _MAX_PIXEL:
            continue
        cv2.line(
            canvas,
            (int(p0[0]), int(p0[1])),
            (int(p1[0]), int(p1[1])),
            colour,
            thickness,
            cv2.LINE_AA,
        )


def draw_floor_grid(
    image: Image,
    cam: CameraCalib,
    step_m: float = 0.5,
    samples: int = 40,
    label_axes: bool = True,
) -> Image:
    """Project a metric floor grid into the camera image.

    If the grid does not lie flat on the visible floor, the ground homography is
    wrong — no further diagnosis needed.
    """
    canvas = image.copy()
    extent = cam.ground.floor_extent_m
    if extent is None:
        cv2.putText(
            canvas,
            "no floor_extent_m in calibration",
            (12, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return canvas
    x0, y0, x1, y1 = extent

    for x in np.arange(np.ceil(x0 / step_m) * step_m, x1 + 1e-9, step_m):
        line = np.stack([np.full(samples, x), np.linspace(y0, y1, samples)], axis=1)
        pixels, valid = ground_to_image(cam, line)
        _polyline(canvas, pixels, valid, GRID_COLOR)
    for y in np.arange(np.ceil(y0 / step_m) * step_m, y1 + 1e-9, step_m):
        line = np.stack([np.linspace(x0, x1, samples), np.full(samples, y)], axis=1)
        pixels, valid = ground_to_image(cam, line)
        _polyline(canvas, pixels, valid, GRID_COLOR)

    if label_axes:
        origin = np.array([[0.0, 0.0]])
        x_axis = np.stack([np.linspace(0.0, 1.0, samples), np.zeros(samples)], axis=1)
        y_axis = np.stack([np.zeros(samples), np.linspace(0.0, 1.0, samples)], axis=1)
        px, ok = ground_to_image(cam, x_axis)
        _polyline(canvas, px, ok, AXIS_X_COLOR, 3)
        py, ok_y = ground_to_image(cam, y_axis)
        _polyline(canvas, py, ok_y, AXIS_Y_COLOR, 3)
        po, ok_o = ground_to_image(cam, origin)
        if ok_o[0] and np.isfinite(po[0]).all() and abs(po[0]).max() < _MAX_PIXEL:
            cv2.circle(canvas, (int(po[0, 0]), int(po[0, 1])), 6, TEXT_COLOR, -1, cv2.LINE_AA)

    cv2.putText(
        canvas,
        f"{cam.camera_id}  floor grid {step_m:g} m  (red=+X, green=+Y)",
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )
    return canvas


def draw_tag_reprojection(
    image: Image,
    cam: CameraCalib,
    detections: dict[int, FloatArray],
    placements: list[TagPlacement],
) -> Image:
    """Detected tag corners (green) against corners predicted by the calibration (red).

    Green and red should sit on top of each other. Any visible separation is the
    calibration error, in pixels, at a location you can point at.
    """
    canvas = image.copy()
    by_id = {p.tag_id: p for p in placements}

    for tag_id, corners in sorted(detections.items()):
        cv2.polylines(
            canvas, [corners.astype(np.int32)], True, DETECTED_COLOR, 2, cv2.LINE_AA
        )
        centre = corners.mean(axis=0)
        cv2.putText(
            canvas,
            str(tag_id),
            (int(centre[0]) - 8, int(centre[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            DETECTED_COLOR,
            2,
            cv2.LINE_AA,
        )

        placement = by_id.get(tag_id)
        if placement is None:
            continue
        predicted, valid = ground_to_image(cam, placement.world_corners())
        if not valid.all() or not np.isfinite(predicted).all():
            continue
        if abs(predicted).max() > _MAX_PIXEL:
            continue
        cv2.polylines(
            canvas, [predicted.astype(np.int32)], True, REPROJECTED_COLOR, 2, cv2.LINE_AA
        )
        for observed, expected in zip(corners, predicted, strict=True):
            cv2.line(
                canvas,
                (int(observed[0]), int(observed[1])),
                (int(expected[0]), int(expected[1])),
                (0, 165, 255),
                1,
                cv2.LINE_AA,
            )

    missing = sorted(set(by_id) - set(detections))
    banner = f"{cam.camera_id}  green=detected  red=reprojected from calibration"
    if missing:
        banner += f"   MISSING TAGS: {missing}"
    cv2.putText(
        canvas, banner, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.58, TEXT_COLOR, 2, cv2.LINE_AA
    )
    return canvas
