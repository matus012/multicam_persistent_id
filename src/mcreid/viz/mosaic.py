"""Compose the demo frame: 2x2 camera grid beside the BEV canvas."""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt

from mcreid.viz.palette import FLOOR_COLOR, TEXT_COLOR

Image = npt.NDArray[np.uint8]


def _fit(image: Image, size: tuple[int, int]) -> Image:
    """Resize preserving aspect ratio, letterboxing onto a canvas of ``size``."""
    target_w, target_h = size
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    resized = cv2.resize(
        image, (max(int(w * scale), 1), max(int(h * scale), 1)), interpolation=cv2.INTER_AREA
    )
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    canvas[:] = FLOOR_COLOR
    y = (target_h - resized.shape[0]) // 2
    x = (target_w - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def compose(
    views: dict[str, Image],
    bev: Image,
    camera_order: list[str],
    panel_size: tuple[int, int] = (480, 270),
    caption: str = "",
) -> Image:
    """Build one demo frame.

    Layout: a 2x2 grid of camera panels on the left, the BEV on the right at the
    grid's full height, with an optional caption strip underneath.
    """
    if not camera_order:
        raise ValueError("camera_order must be non-empty")
    if len(camera_order) > 4:
        raise ValueError(f"layout supports at most 4 cameras, got {len(camera_order)}")

    pw, ph = panel_size
    grid = np.zeros((ph * 2, pw * 2, 3), dtype=np.uint8)
    grid[:] = FLOOR_COLOR
    for i, camera_id in enumerate(camera_order):
        row, col = divmod(i, 2)
        panel = views.get(camera_id)
        if panel is None:
            continue
        grid[row * ph : (row + 1) * ph, col * pw : (col + 1) * pw] = _fit(panel, panel_size)

    bev_panel = _fit(bev, (ph * 2, ph * 2))
    frame = np.hstack([grid, bev_panel])

    if caption:
        strip = np.zeros((46, frame.shape[1], 3), dtype=np.uint8)
        strip[:] = (18, 18, 20)
        cv2.putText(
            strip, caption, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, TEXT_COLOR, 1, cv2.LINE_AA
        )
        frame = np.vstack([frame, strip])
    return frame
