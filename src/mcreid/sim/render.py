"""Draw synthetic camera frames for a `ToyScene`.

There is no real footage yet, and G-M1-2 is blocked on it. Rendering the toy
scene into actual images means the full demo path — per-view overlay, BEV,
mosaic, video/GIF export — is exercised and reviewable *now*, so when the real
clips arrive only the detector front-end is new.

These frames are intentionally schematic (floor grid + person rectangles), not
photorealistic. Nothing downstream consumes pixels: the toy detections come from
the generator's analytic projection, and this module only visualises them.
"""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt

from mcreid.calib.geometry import ground_to_image
from mcreid.calib.schema import CameraCalib
from mcreid.sim.toy import ToyScene
from mcreid.viz.palette import FLOOR_COLOR, GRID_COLOR

Image = npt.NDArray[np.uint8]


def render_floor(
    cam: CameraCalib, extent: tuple[float, float, float, float], step_m: float = 1.0
) -> Image:
    """Blank camera frame with the floor grid projected into it."""
    width, height = cam.intrinsics.image_size
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = FLOOR_COLOR

    x0, y0, x1, y1 = extent
    xs = np.arange(np.ceil(x0 / step_m) * step_m, x1 + 1e-9, step_m)
    ys = np.arange(np.ceil(y0 / step_m) * step_m, y1 + 1e-9, step_m)
    samples = 24

    segments: list[np.ndarray] = []
    for x in xs:
        segments.append(np.stack([np.full(samples, x), np.linspace(y0, y1, samples)], axis=1))
    for y in ys:
        segments.append(np.stack([np.linspace(x0, x1, samples), np.full(samples, y)], axis=1))

    for line in segments:
        pixels, valid = ground_to_image(cam, line)
        for i in range(1, samples):
            if not (valid[i - 1] and valid[i]):
                continue
            p0 = pixels[i - 1]
            p1 = pixels[i]
            if not (np.isfinite(p0).all() and np.isfinite(p1).all()):
                continue
            # Guard against the huge coordinates a near-horizon projection makes.
            if max(abs(p0).max(), abs(p1).max()) > 1e4:
                continue
            cv2.line(
                canvas,
                (int(p0[0]), int(p0[1])),
                (int(p1[0]), int(p1[1])),
                GRID_COLOR,
                1,
                cv2.LINE_AA,
            )
    return canvas


class ToySceneRenderer:
    """Renders per-camera frames for a toy scene, caching the static floor."""

    def __init__(self, scene: ToyScene, grid_step_m: float = 1.0) -> None:
        self.scene = scene
        extent = scene.config.floor_extent_m
        self._floors = {
            cam.camera_id: render_floor(cam, extent, grid_step_m) for cam in scene.rig.cameras
        }

    def frame(self, camera_id: str, frame_index: int) -> Image:
        """Synthetic view for one camera at one frame (people drawn as slabs)."""
        canvas = self._floors[camera_id].copy()
        for det in self.scene.frame_detections(frame_index)[camera_id]:
            box = np.asarray(det.bbox_xyxy, dtype=np.float64)
            p0 = (int(box[0]), int(box[1]))
            p1 = (int(box[2]), int(box[3]))
            shade = (90, 90, 96) if det.gt_agent_id is not None else (60, 45, 45)
            cv2.rectangle(canvas, p0, p1, shade, -1)
            cv2.rectangle(canvas, p0, p1, (150, 150, 158), 1, cv2.LINE_AA)
        return canvas

    def all_frames(self, frame_index: int) -> dict[str, Image]:
        return {cam: self.frame(cam, frame_index) for cam in self._floors}
