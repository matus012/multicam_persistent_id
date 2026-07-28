"""Human-watchable demo composition.

The evaluation mosaic packs seven 1080p views into a strip, which is fine for
debugging and useless as a demo — people end up ~30 px tall and the ID labels
are unreadable. This module optimises for a viewer instead:

* three camera panels at a size where a person is actually legible, plus the BEV
* ID labels drawn large, with an outline so they survive a busy background
* the same colour for a global ID in every panel and on the map, because the
  whole claim of the project is that those numbers agree across cameras
* a caption naming the person currently being handed between cameras
"""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt

from mcreid.fusion.types import GlobalTrackSnapshot, TrackState, ViewObservation
from mcreid.viz.palette import TEXT_COLOR, id_color

Image = npt.NDArray[np.uint8]

_FONT = cv2.FONT_HERSHEY_DUPLEX


def draw_demo_view(
    frame: Image,
    observations: list[ViewObservation],
    global_ids: dict[int, int],
    camera_id: str,
    panel_size: tuple[int, int],
    highlight: int | None = None,
) -> Image:
    """Draw boxes and large global-ID labels, then scale to ``panel_size``.

    Boxes are drawn at full resolution and the panel is scaled afterwards, so
    line weights and text stay proportional instead of turning into mush.
    """
    canvas = frame.copy()
    target_w, target_h = panel_size
    scale = target_w / canvas.shape[1]
    # Compensate so that, after downscaling, strokes land at a readable size.
    thickness = max(int(round(3 / scale)), 2)
    font_scale = 1.1 / scale

    for obs in observations:
        gid = global_ids.get(obs.local_track_id)
        box = np.asarray(obs.bbox_xyxy, dtype=np.float64)
        p0 = (int(box[0]), int(box[1]))
        p1 = (int(box[2]), int(box[3]))
        if gid is None:
            cv2.rectangle(canvas, p0, p1, (120, 120, 120), max(thickness // 2, 1))
            continue

        colour = id_color(gid)
        emphasis = thickness + 3 if gid == highlight else thickness
        cv2.rectangle(canvas, p0, p1, colour, emphasis)

        label = str(gid)
        (tw, th), _ = cv2.getTextSize(label, _FONT, font_scale, thickness)
        top = max(p0[1] - th - int(14 / scale), 0)
        cv2.rectangle(
            canvas,
            (p0[0], top),
            (p0[0] + tw + int(16 / scale), top + th + int(14 / scale)),
            colour,
            -1,
        )
        cv2.putText(
            canvas,
            label,
            (p0[0] + int(8 / scale), top + th + int(4 / scale)),
            _FONT,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )

    panel: Image = np.asarray(
        cv2.resize(canvas, panel_size, interpolation=cv2.INTER_AREA), dtype=np.uint8
    )
    cv2.rectangle(panel, (0, 0), (panel.shape[1] - 1, panel.shape[0] - 1), (70, 70, 70), 2)
    _banner(panel, camera_id)
    return panel


def _banner(panel: Image, text: str) -> None:
    (tw, th), _ = cv2.getTextSize(text, _FONT, 0.7, 2)
    cv2.rectangle(panel, (0, 0), (tw + 18, th + 16), (18, 18, 20), -1)
    cv2.putText(panel, text, (9, th + 7), _FONT, 0.7, TEXT_COLOR, 2, cv2.LINE_AA)


def compose_demo(
    panels: dict[str, Image],
    bev: Image,
    camera_order: list[str],
    panel_size: tuple[int, int],
    caption: str = "",
    subcaption: str = "",
) -> Image:
    """Three camera panels (top row + bottom left) with the BEV filling the rest."""
    pw, ph = panel_size
    grid = np.full((ph * 2, pw * 2, 3), 16, dtype=np.uint8)

    slots = [(0, 0), (0, 1), (1, 0)]
    for camera_id, (row, col) in zip(camera_order[:3], slots, strict=False):
        panel = panels.get(camera_id)
        if panel is not None:
            grid[row * ph : (row + 1) * ph, col * pw : (col + 1) * pw] = panel

    bev_panel = cv2.resize(bev, (pw, ph), interpolation=cv2.INTER_AREA)
    cv2.rectangle(
        bev_panel, (0, 0), (bev_panel.shape[1] - 1, bev_panel.shape[0] - 1), (70, 70, 70), 2
    )
    grid[ph:, pw:] = bev_panel

    if not caption and not subcaption:
        return grid
    strip = np.full((78 if subcaption else 52, grid.shape[1], 3), 18, dtype=np.uint8)
    if caption:
        cv2.putText(strip, caption, (16, 33), _FONT, 0.78, TEXT_COLOR, 2, cv2.LINE_AA)
    if subcaption:
        cv2.putText(strip, subcaption, (16, 64), _FONT, 0.62, (150, 220, 255), 1, cv2.LINE_AA)
    return np.vstack([grid, strip])


def find_handoff_segments(
    snapshots_per_frame: list[list[GlobalTrackSnapshot]],
    min_length: int,
) -> list[tuple[int, int, int, int]]:
    """Find windows where a track is visibly handed between camera sets.

    A "handoff" is a frame where a confirmed track's supporting-camera set gains
    a camera it did not have before while keeping its ID. Returns
    ``(score, start, end, global_id)`` sorted best first — score counts handoffs
    for that ID inside the window, which is what makes a demo segment worth
    watching.
    """
    seen_cameras: dict[int, set[str]] = {}
    events: dict[int, list[int]] = {}

    for frame, snaps in enumerate(snapshots_per_frame):
        for snap in snaps:
            if snap.state is not TrackState.CONFIRMED or not snap.supporting_cameras:
                continue
            known = seen_cameras.setdefault(snap.global_id, set())
            fresh = set(snap.supporting_cameras) - known
            if fresh and known:
                events.setdefault(snap.global_id, []).append(frame)
            known.update(snap.supporting_cameras)

    scored: list[tuple[int, int, int, int]] = []
    total = len(snapshots_per_frame)
    for gid, frames in events.items():
        for pivot in frames:
            start = max(0, pivot - min_length // 2)
            end = min(total, start + min_length)
            start = max(0, end - min_length)
            score = sum(1 for f in frames if start <= f < end)
            # Only worth showing if the ID is actually on screen for the window.
            present = sum(
                1
                for f in range(start, end)
                for s in snapshots_per_frame[f]
                if s.global_id == gid
            )
            if present < min_length * 0.6:
                continue
            scored.append((score, start, end, gid))

    scored.sort(reverse=True)
    deduped: list[tuple[int, int, int, int]] = []
    for item in scored:
        if all(abs(item[1] - kept[1]) >= min_length // 2 for kept in deduped):
            deduped.append(item)
    return deduped
