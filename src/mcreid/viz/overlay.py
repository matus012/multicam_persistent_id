"""Per-camera overlays: boxes labelled with the *global* ID.

The label deliberately shows the global ID, not the per-view local track ID —
the claim being demonstrated is cross-camera identity, so the number over the
person must be the same in all four panels.
"""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt

from mcreid.calib.geometry import feet_point
from mcreid.calib.schema import CameraCalib
from mcreid.fusion.types import ViewObservation
from mcreid.viz.palette import TEXT_COLOR, id_color

Image = npt.NDArray[np.uint8]


def draw_view(
    frame: Image,
    observations: list[ViewObservation],
    global_ids: dict[int, int],
    camera_id: str,
    frame_index: int | None = None,
    occluded: bool = False,
) -> Image:
    """Draw boxes + global-ID labels for one camera.

    Args:
        frame: BGR image, modified on a copy.
        observations: this camera's view observations for the frame.
        global_ids: local_track_id -> global_id. Locals with no global
            assignment are drawn thin and unlabelled rather than hidden, so the
            viewer can see the tracker considering a detection it did not fuse.
        occluded: annotate the panel as fully blocked this frame.
    """
    canvas = frame.copy()
    for obs in observations:
        box = np.asarray(obs.bbox_xyxy, dtype=np.float64)
        p0 = (int(round(box[0])), int(round(box[1])))
        p1 = (int(round(box[2])), int(round(box[3])))
        gid = global_ids.get(obs.local_track_id)

        if gid is None:
            cv2.rectangle(canvas, p0, p1, (110, 110, 110), 1, cv2.LINE_AA)
            continue

        colour = id_color(gid)
        cv2.rectangle(canvas, p0, p1, colour, 2, cv2.LINE_AA)

        label = f"ID {gid}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        top = max(p0[1] - th - 8, 0)
        cv2.rectangle(canvas, (p0[0], top), (p0[0] + tw + 8, top + th + 8), colour, -1)
        cv2.putText(
            canvas,
            label,
            (p0[0] + 4, top + th + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

        foot = feet_point(box)[0]
        cv2.drawMarker(
            canvas,
            (int(round(foot[0])), int(round(foot[1]))),
            colour,
            cv2.MARKER_CROSS,
            12,
            2,
            cv2.LINE_AA,
        )

    banner = camera_id if frame_index is None else f"{camera_id}  f{frame_index}"
    cv2.putText(
        canvas, banner, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, TEXT_COLOR, 2, cv2.LINE_AA
    )
    if occluded:
        cv2.putText(
            canvas,
            "BLOCKED",
            (10, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 80, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.rectangle(
            canvas, (2, 2), (canvas.shape[1] - 3, canvas.shape[0] - 3), (0, 80, 255), 3
        )
    return canvas


def draw_horizon(
    canvas: Image, cam: CameraCalib, colour: tuple[int, int, int] = (0, 0, 180)
) -> Image:
    """Debug aid: draw the horizon line, above which ground projection is invalid."""
    H = cam.ground.H
    a, b, c = H[2]
    width = canvas.shape[1]
    if abs(b) < 1e-9:
        return canvas
    pts = [(x, (-c - a * x) / b) for x in (0.0, float(width - 1))]
    cv2.line(
        canvas,
        (int(pts[0][0]), int(pts[0][1])),
        (int(pts[1][0]), int(pts[1][1])),
        colour,
        1,
        cv2.LINE_AA,
    )
    return canvas
