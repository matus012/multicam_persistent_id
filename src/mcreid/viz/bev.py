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

from mcreid.calib.geometry import image_to_ground
from mcreid.calib.schema import CameraCalib, RigCalib
from mcreid.fusion.types import GlobalTrackSnapshot, TrackState
from mcreid.viz.palette import (
    CAMERA_COLOR,
    FLOOR_COLOR,
    GRID_COLOR,
    TEXT_COLOR,
    camera_color,
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
        max_labelled_tracks: int = 12,
    ) -> None:
        self.rig = rig
        self.canvas_size = canvas_size
        self.trail_length = trail_length
        self.grid_step_m = grid_step_m
        self.max_labelled_tracks = max_labelled_tracks
        """Above this many live tracks the map is too dense to label everything;
        labelling is reduced to the tracks that carry information."""

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

    def draw_cameras(
        self,
        canvas: Image,
        positions: dict[str, tuple[float, float]],
        camera_order: list[str] | None = None,
    ) -> None:
        """Mark physical camera positions on the plan (optional, cosmetic).

        With ``camera_order`` each mount takes its tile colour, matching the
        frustum wash; without it they all take the legacy single colour.
        """
        order = camera_order or []
        for camera_id, xy in positions.items():
            colour = camera_color(order.index(camera_id)) if camera_id in order else CAMERA_COLOR
            px = self.to_pixels(xy)
            cv2.drawMarker(canvas, px, colour, cv2.MARKER_TRIANGLE_UP, 18, 3, cv2.LINE_AA)
            cv2.putText(
                canvas,
                camera_id,
                (px[0] + 12, px[1] + 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                colour,
                2,
                cv2.LINE_AA,
            )

    def ground_footprint(self, camera: CameraCalib, samples: int = 24) -> FloatArray:
        """The ground region this camera actually sees, in world metres.

        Sampled along the image border and back-projected with the real inverse
        homography, rather than drawn as a symmetric cone from the mount point.
        A pitched camera's floor footprint is a trapezoid whose far edge runs to
        the horizon, and a cone drawn from the FOV angle alone would claim
        coverage the camera does not have — on a map whose whole purpose is to
        show which camera can see where, that is a lie with pixels.

        Border points above the horizon back-project to invalid or absurd
        coordinates; they are dropped, and the convex hull of what survives is
        returned. Empty if the camera sees no floor at all.
        """
        width, height = camera.intrinsics.image_size
        edge = np.linspace(0.0, 1.0, samples)
        border = np.concatenate(
            [
                np.stack([edge * width, np.full(samples, height - 1.0)], axis=1),  # bottom
                np.stack([edge * width, np.zeros(samples)], axis=1),  # top
                np.stack([np.zeros(samples), edge * height], axis=1),  # left
                np.stack([np.full(samples, width - 1.0), edge * height], axis=1),  # right
            ]
        )
        world, ok = image_to_ground(camera, border)
        good = np.asarray(ok, dtype=bool) & np.isfinite(world).all(axis=1)
        # A near-horizon ray lands kilometres away and would blow the hull out to
        # nothing useful; clamp to a generous multiple of the room instead.
        x0, y0, x1, y1 = self.extent
        reach = 4.0 * max(x1 - x0, y1 - y0)
        centre = np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0])
        good &= np.linalg.norm(world - centre, axis=1) <= reach
        points = world[good]
        if len(points) < 3:
            return np.empty((0, 2), dtype=np.float64)
        hull = cv2.convexHull(points.astype(np.float32).reshape(-1, 1, 2))

        # Clip to the room. A pitched camera's footprint runs to the horizon,
        # metres past any wall, and washing that raw polygon over the plan buries
        # the floor grid under colour and implies coverage of floor that is not
        # in the room at all. Both polygons are convex, so the intersection is.
        rx0, ry0, rx1, ry1 = self.rig.floor_extent()
        room = np.array(
            [[rx0, ry0], [rx1, ry0], [rx1, ry1], [rx0, ry1]], dtype=np.float32
        ).reshape(-1, 1, 2)
        area, clipped = cv2.intersectConvexConvex(hull, room)
        if clipped is None or area <= 0.0:
            return np.empty((0, 2), dtype=np.float64)
        return np.asarray(clipped, dtype=np.float64).reshape(-1, 2)

    def draw_camera_frustums(
        self, canvas: Image, camera_order: list[str] | None = None, alpha: float = 0.10
    ) -> None:
        """Wash each camera's floor coverage in that camera's tile colour.

        Colour is keyed to position in ``camera_order`` so the wash under a
        person and the border of the tile they are visible in are the same
        colour — that pairing is the whole point of putting the map on screen.
        """
        order = camera_order if camera_order is not None else list(self.rig.camera_ids)
        # Blend each camera separately rather than filling one shared overlay:
        # a single overlay lets the last camera's fill overwrite the others, so
        # floor seen by two cameras renders as if only one covered it, and the
        # first camera in the list disappears entirely. Blending in sequence
        # makes overlap read as a deeper mix, which is the useful signal — where
        # coverage doubles is exactly where a handoff can happen.
        for index, camera_id in enumerate(order):
            camera = self.rig.get(camera_id)
            polygon = self.ground_footprint(camera)
            if len(polygon) < 3:
                continue
            overlay = canvas.copy()
            pixels = np.array([self.to_pixels(p) for p in polygon], dtype=np.int32)
            cv2.fillConvexPoly(overlay, pixels, camera_color(index), cv2.LINE_AA)
            cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0.0, dst=canvas)
        for index, camera_id in enumerate(order):
            camera = self.rig.get(camera_id)
            polygon = self.ground_footprint(camera)
            if len(polygon) < 3:
                continue
            pixels = np.array([self.to_pixels(p) for p in polygon], dtype=np.int32)
            cv2.polylines(canvas, [pixels], True, camera_color(index), 2, cv2.LINE_AA)

    def render(
        self,
        snapshots: list[GlobalTrackSnapshot],
        frame: int | None = None,
        camera_positions: dict[str, tuple[float, float]] | None = None,
        camera_order: list[str] | None = None,
    ) -> Image:
        """Draw one BEV frame."""
        canvas = self._blank()
        if camera_order is not None:
            self.draw_camera_frustums(canvas, camera_order)
        if camera_positions:
            self.draw_cameras(canvas, camera_positions, camera_order)

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

            # In a crowd, per-track annotations overlap into an unreadable mat.
            # Above the threshold, label only what the viewer needs: the ID, and
            # a marker for tracks that are fused across cameras or coasting.
            crowded = len(snapshots) > self.max_labelled_tracks
            if crowded and not coasting and len(snap.supporting_cameras) < 2:
                continue
            label = str(snap.global_id)
            if coasting:
                label += f" occl {snap.frames_since_measurement}f"
            elif len(snap.supporting_cameras) >= 2:
                label += f" [{len(snap.supporting_cameras)}cam]"
            elif not crowded:
                label += " [1cam]"
            cv2.putText(
                canvas,
                label,
                (px[0] + 13, px[1] - 9),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
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
