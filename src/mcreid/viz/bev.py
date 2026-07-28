"""Bird's-eye-view canvas: the floor plan with moving global-ID dots.

This is the artefact the demo is judged on. A coasting track — one whose target
is currently hidden from every camera — is drawn as a hollow ring with a dashed
trail rather than being dropped, so the viewer can watch the ID *persist* while
the cardboard is up. Hiding it would make the hardest part of the system
invisible in the very GIF that is supposed to prove it works.
"""

from __future__ import annotations

from collections import defaultdict, deque

import cv2
import numpy as np
import numpy.typing as npt

from mcreid.calib.schema import RigCalib
from mcreid.fusion.types import GlobalTrackSnapshot, TrackState
from mcreid.viz.palette import (
    CAMERA_COLOR,
    FLOOR_COLOR,
    GRID_COLOR,
    TEXT_COLOR,
    dim,
    id_color,
)

FloatArray = npt.NDArray[np.float64]
Image = npt.NDArray[np.uint8]


class BevRenderer:
    """Renders global tracks onto a metric floor plan."""

    def __init__(
        self,
        rig: RigCalib,
        canvas_size: tuple[int, int] = (640, 640),
        margin_m: float = 0.4,
        trail_length: int = 45,
        grid_step_m: float = 1.0,
    ) -> None:
        self.rig = rig
        self.canvas_size = canvas_size
        self.trail_length = trail_length
        self.grid_step_m = grid_step_m

        x0, y0, x1, y1 = rig.floor_extent()
        self.extent = (x0 - margin_m, y0 - margin_m, x1 + margin_m, y1 + margin_m)
        width_m = self.extent[2] - self.extent[0]
        depth_m = self.extent[3] - self.extent[1]
        if width_m <= 0 or depth_m <= 0:
            raise ValueError(f"degenerate floor extent: {self.extent}")

        # Uniform scale so the floor plan keeps its aspect ratio.
        self.px_per_m = min(canvas_size[0] / width_m, canvas_size[1] / depth_m)
        self._trails: dict[int, deque[tuple[int, int]]] = defaultdict(
            lambda: deque(maxlen=trail_length)
        )
        self._coasting: dict[int, deque[bool]] = defaultdict(
            lambda: deque(maxlen=trail_length)
        )

    def to_pixels(self, world_xy: npt.ArrayLike) -> tuple[int, int]:
        """World metres -> canvas pixels. World +Y points up, so the row axis flips."""
        pt = np.asarray(world_xy, dtype=np.float64).reshape(2)
        col = (pt[0] - self.extent[0]) * self.px_per_m
        row = (self.extent[3] - pt[1]) * self.px_per_m
        return int(round(col)), int(round(row))

    def _blank(self) -> Image:
        canvas = np.full(
            (self.canvas_size[1], self.canvas_size[0], 3), FLOOR_COLOR[0], dtype=np.uint8
        )
        canvas[:] = FLOOR_COLOR

        x0, y0, x1, y1 = self.extent
        start = np.ceil(x0 / self.grid_step_m) * self.grid_step_m
        for x in np.arange(start, x1, self.grid_step_m):
            c0 = self.to_pixels((x, y0))
            c1 = self.to_pixels((x, y1))
            cv2.line(canvas, c0, c1, GRID_COLOR, 1, cv2.LINE_AA)
        start = np.ceil(y0 / self.grid_step_m) * self.grid_step_m
        for y in np.arange(start, y1, self.grid_step_m):
            c0 = self.to_pixels((x0, y))
            c1 = self.to_pixels((x1, y))
            cv2.line(canvas, c0, c1, GRID_COLOR, 1, cv2.LINE_AA)
        return canvas

    def draw_cameras(self, canvas: Image, positions: dict[str, tuple[float, float]]) -> None:
        """Mark physical camera positions on the plan (optional, cosmetic)."""
        for camera_id, xy in positions.items():
            px = self.to_pixels(xy)
            cv2.drawMarker(canvas, px, CAMERA_COLOR, cv2.MARKER_TRIANGLE_UP, 14, 2, cv2.LINE_AA)
            cv2.putText(
                canvas,
                camera_id,
                (px[0] + 10, px[1] + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                CAMERA_COLOR,
                1,
                cv2.LINE_AA,
            )

    def render(
        self,
        snapshots: list[GlobalTrackSnapshot],
        frame: int | None = None,
        camera_positions: dict[str, tuple[float, float]] | None = None,
    ) -> Image:
        """Draw one BEV frame."""
        canvas = self._blank()
        if camera_positions:
            self.draw_cameras(canvas, camera_positions)

        live_ids = {s.global_id for s in snapshots}
        for gid in list(self._trails):
            if gid not in live_ids:
                del self._trails[gid]
                self._coasting.pop(gid, None)

        for snap in snapshots:
            coasting = snap.state is TrackState.COASTING
            colour = id_color(snap.global_id)
            px = self.to_pixels(snap.world_xy)
            self._trails[snap.global_id].append(px)
            self._coasting[snap.global_id].append(coasting)

            trail = list(self._trails[snap.global_id])
            flags = list(self._coasting[snap.global_id])
            for i in range(1, len(trail)):
                # Dashes while coasting: the trail visibly changes texture the
                # moment the target stops being observed.
                if flags[i] and i % 2 == 0:
                    continue
                fade = 0.25 + 0.75 * (i / max(len(trail) - 1, 1))
                cv2.line(canvas, trail[i - 1], trail[i], dim(colour, fade), 2, cv2.LINE_AA)

            if coasting:
                cv2.circle(canvas, px, 11, colour, 2, cv2.LINE_AA)
                cv2.circle(canvas, px, 3, colour, -1, cv2.LINE_AA)
            else:
                cv2.circle(canvas, px, 9, colour, -1, cv2.LINE_AA)
                cv2.circle(canvas, px, 9, (255, 255, 255), 1, cv2.LINE_AA)

            speed = float(np.linalg.norm(snap.velocity_mps))
            if speed > 0.15:
                tip = snap.world_xy + snap.velocity_mps * 0.5
                cv2.arrowedLine(
                    canvas, px, self.to_pixels(tip), colour, 2, cv2.LINE_AA, tipLength=0.3
                )

            label = f"ID {snap.global_id}"
            if coasting:
                label += f"  occluded {snap.frames_since_measurement}f"
            elif snap.supporting_cameras:
                label += f"  [{len(snap.supporting_cameras)} cam]"
            cv2.putText(
                canvas,
                label,
                (px[0] + 13, px[1] - 9),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                colour,
                2 if not coasting else 1,
                cv2.LINE_AA,
            )

        header = "BEV  (metres)"
        if frame is not None:
            header += f"   frame {frame}"
        cv2.putText(
            canvas, header, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, cv2.LINE_AA
        )
        cv2.putText(
            canvas,
            f"{self.grid_step_m:g} m grid",
            (10, self.canvas_size[1] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            GRID_COLOR,
            1,
            cv2.LINE_AA,
        )
        return canvas

    def reset(self) -> None:
        self._trails.clear()
        self._coasting.clear()
