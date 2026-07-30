"""Single-camera live tracking session.

The whole per-view stack — detection, tracking, ReID, occlusion coasting and
dormant-gallery long-gap re-identification — on one webcam, in real time.

Deliberately separated from the CLI so the frame loop is testable without a
camera, a GPU, or torch: `LiveSession.process` takes a frame and returns an
annotated frame plus state, and both the detector backend and the clock are
injectable.

**Calibration is optional.** With one camera there is no cross-view fusion to do,
so the ground plane is not needed to keep an identity — only to draw a metric
map. Without `--homography` the session builds a *pixel-plane* calibration: a
pure scale from pixels to pseudo-metres, so the Kalman filter, coasting and
gallery all operate in consistent units. Those distances are **not physical**,
the BEV panel is omitted, and nothing downstream claims otherwise.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
import numpy.typing as npt

from mcreid.calib.schema import CameraCalib, GroundPlane, Intrinsics, RigCalib
from mcreid.diagnostics.shadow import ShadowProbe
from mcreid.fusion.global_id import FusionConfig, GlobalIDManager
from mcreid.fusion.types import GlobalTrackSnapshot, TrackState, ViewObservation
from mcreid.utils.logging import get_logger
from mcreid.viz.palette import TEXT_COLOR, id_color

logger = get_logger(__name__)

Image = npt.NDArray[np.uint8]
FloatArray = npt.NDArray[np.float64]

_FONT = cv2.FONT_HERSHEY_DUPLEX
LIVE_CAMERA_ID = "live"


class ViewBackend(Protocol):
    """Anything that turns a frame into per-view observations."""

    def step(self, image: Image, frame: int) -> list[ViewObservation]: ...


def pixel_plane_calibration(
    width: int, height: int, span_m: float = 6.0
) -> CameraCalib:
    """A stand-in calibration for an uncalibrated single camera.

    Maps pixels to pseudo-metres by a single scale so that the fusion stage's
    metre-denominated gates stay in a sane numeric range. ``span_m`` is what the
    frame height is *assumed* to cover. It is a unit convention, not a
    measurement: positions produced under it are not physical, which is why the
    BEV panel is suppressed unless a real homography is supplied.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid frame size {width}x{height}")
    if span_m <= 0:
        raise ValueError(f"span_m must be positive, got {span_m}")

    scale = span_m / height
    homography = [[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]]
    return CameraCalib(
        camera_id=LIVE_CAMERA_ID,
        intrinsics=Intrinsics(
            fx=float(width),
            fy=float(width),
            cx=width / 2.0,
            cy=height / 2.0,
            dist_coeffs=[0.0] * 5,
            image_width=width,
            image_height=height,
            rms_reproj_px=0.0,
            n_views=0,
        ),
        ground=GroundPlane.from_matrix(
            H=np.asarray(homography, dtype=np.float64),
            method="synthetic",
            rms_error_m=0.0,
            n_correspondences=4,
            floor_extent_m=(0.0, 0.0, width * scale, height * scale),
        ),
        notes="pixel-plane stand-in: distances are scaled pixels, not metres",
    )


def load_homography_calibration(path: Path, width: int, height: int) -> CameraCalib:
    """Build a real ground calibration from a 4-point YAML.

    Expected shape::

        image_points: [[x, y], [x, y], [x, y], [x, y]]   # pixels
        world_points: [[x, y], [x, y], [x, y], [x, y]]   # metres on the floor
    """
    import yaml

    from mcreid.calib.homography import ground_plane_from_correspondences

    if not path.is_file():
        raise FileNotFoundError(f"homography file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("image_points", "world_points"):
        if key not in data:
            raise ValueError(f"{path}: missing required key {key!r}")

    image_points = np.asarray(data["image_points"], dtype=np.float64)
    world_points = np.asarray(data["world_points"], dtype=np.float64)
    if image_points.shape != world_points.shape or image_points.shape[0] < 4:
        raise ValueError(
            f"{path}: need matching lists of >= 4 image/world points, got "
            f"{image_points.shape} and {world_points.shape}"
        )

    intrinsics = Intrinsics(
        fx=float(width),
        fy=float(width),
        cx=width / 2.0,
        cy=height / 2.0,
        dist_coeffs=[0.0] * 5,
        image_width=width,
        image_height=height,
        rms_reproj_px=0.0,
        n_views=0,
    )
    ground = ground_plane_from_correspondences(
        intrinsics, image_points, world_points, method="four_point"
    )
    logger.info("ground homography from %s: residual %.1f cm", path, ground.rms_error_m * 100)
    return CameraCalib(camera_id=LIVE_CAMERA_ID, intrinsics=intrinsics, ground=ground)


@dataclass
class IdentityTimeline:
    """When each global ID was first seen, and how long it has been held."""

    first_seen: dict[int, float] = field(default_factory=dict)
    last_seen: dict[int, float] = field(default_factory=dict)
    reacquired_gap: dict[int, float] = field(default_factory=dict)

    def observe(self, global_id: int, now: float, gap_threshold: float = 1.0) -> None:
        previous = self.last_seen.get(global_id)
        if previous is not None and now - previous >= gap_threshold:
            # Same ID returning after an absence: that is a re-acquisition, and
            # the gap is the headline number for the long-gap machinery.
            self.reacquired_gap[global_id] = now - previous
        self.first_seen.setdefault(global_id, now)
        self.last_seen[global_id] = now

    def held_seconds(self, global_id: int, now: float) -> float:
        return now - self.first_seen.get(global_id, now)


@dataclass(frozen=True)
class LiveConfig:
    span_m: float = 6.0
    clip_seconds: float = 8.0
    """How much recent video the rolling buffer keeps for the save hotkey."""
    reacquire_gap_s: float = 1.0
    show_bev: bool = True
    bev_size: int = 360


class LiveSession:
    """Stateful single-camera tracking session."""

    def __init__(
        self,
        backend: ViewBackend,
        calibration: CameraCalib,
        metric: bool,
        config: LiveConfig | None = None,
        fusion_config: FusionConfig | None = None,
        shadow: ShadowProbe | None = None,
    ) -> None:
        self.backend = backend
        self.config = config or LiveConfig()
        self.metric = metric
        """True only when a real homography was supplied; gates the BEV panel."""
        self.rig = RigCalib(cameras=[calibration], world_notes="single live camera")
        self.manager = GlobalIDManager(self.rig, fusion_config)
        self.shadow = shadow
        """Optional measurement-only recorder. None on every normal run."""
        self.timeline = IdentityTimeline()
        self.frame_index = -1
        self._fps = deque[float](maxlen=30)
        self._dt = deque[float](maxlen=30)
        self.last_now = 0.0
        self._max_coasting_lines = 6
        self.clip: deque[Image] = deque(maxlen=1)
        self._bev: Any | None = None
        if self.metric and self.config.show_bev:
            from mcreid.viz.bev import BevRenderer

            self._bev = BevRenderer(
                self.rig,
                canvas_size=(self.config.bev_size, self.config.bev_size),
                grid_step_m=1.0,
                trail_length=30,
            )

    def set_clip_capacity(self, fps: float) -> None:
        """Size the rolling buffer once the real capture rate is known."""
        frames = max(int(self.config.clip_seconds * max(fps, 1.0)), 1)
        self.clip = deque(self.clip, maxlen=frames)

    @property
    def fps(self) -> float:
        """Tracking throughput: detection + fusion + render, per second.

        This is *processing time alone*. It is the right number for "can the
        stack keep up", and the wrong one for "how fast is the session running"
        — see :attr:`wall_fps`.
        """
        return float(np.mean(self._fps)) if self._fps else 0.0

    @property
    def wall_fps(self) -> float:
        """End-to-end loop rate, capture and display included.

        Measured from the caller's own clock, so it counts the webcam read and
        the imshow that :attr:`fps` excludes — together roughly a third of the
        loop at 720p. This is the honest session rate, and the rate a saved
        clip must be written at if it is to play back at life speed.
        """
        return 1.0 / float(np.mean(self._dt)) if self._dt else 0.0

    @property
    def reported_ids(self) -> list[int]:
        """Global IDs that ever reached CONFIRMED — the identities a viewer saw.

        Not the same as ``manager.n_ids_issued``, which counts every birth
        including tentative tracks that a one-frame spurious detection creates
        and the lifecycle deletes three frames later. Quoting the minted count
        as "people seen" overstates it by a large factor.
        """
        return sorted(self.timeline.first_seen)

    def process(self, frame: Image, now: float, dt: float) -> tuple[Image, dict[str, Any]]:
        """Track one frame. Returns the annotated frame and a state summary."""
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")
        self.frame_index += 1
        self.last_now = now
        self._dt.append(dt)
        started = time.perf_counter()

        observations = self.backend.step(frame, self.frame_index)
        snapshots = self.manager.step(observations, self.frame_index, dt)
        if self.shadow is not None:
            self.shadow.observe(self.manager, self.frame_index, now)
        assignment = self.manager.last_assignment
        for snap in snapshots:
            if snap.state is not TrackState.COASTING:
                self.timeline.observe(snap.global_id, now, self.config.reacquire_gap_s)

        annotated = self._render(frame, observations, assignment, snapshots, now)
        self.clip.append(annotated)
        self._fps.append(1.0 / max(time.perf_counter() - started, 1e-6))

        return annotated, {
            "frame": self.frame_index,
            "observations": len(observations),
            "tracks": len(snapshots),
            "coasting": sum(1 for s in snapshots if s.state is TrackState.COASTING),
            "dormant": len(self.manager.dormant),
            "resurrected": self.manager.dormant.n_resurrected,
            "reported_ids": len(self.timeline.first_seen),
            "fps": self.fps,
            "wall_fps": self.wall_fps,
        }

    def _render(
        self,
        frame: Image,
        observations: list[ViewObservation],
        assignment: dict[tuple[str, int], int],
        snapshots: list[GlobalTrackSnapshot],
        now: float,
    ) -> Image:
        canvas = frame.copy()
        states = {s.global_id: s for s in snapshots}

        for obs in observations:
            gid = assignment.get((LIVE_CAMERA_ID, obs.local_track_id))
            box = np.asarray(obs.bbox_xyxy, dtype=np.float64)
            p0 = (int(box[0]), int(box[1]))
            p1 = (int(box[2]), int(box[3]))
            if gid is None:
                cv2.rectangle(canvas, p0, p1, (120, 120, 120), 1)
                continue

            snap = states.get(gid)
            coasting = snap is not None and snap.state is TrackState.COASTING
            colour = id_color(gid)
            cv2.rectangle(canvas, p0, p1, colour, 2 if coasting else 3)

            state_name = snap.state.value.upper() if snap else "TENTATIVE"
            held = self.timeline.held_seconds(gid, now)
            label = f"ID {gid}  {state_name}  {held:.0f}s"
            (tw, th), _ = cv2.getTextSize(label, _FONT, 0.6, 2)
            top = max(p0[1] - th - 12, 0)
            cv2.rectangle(canvas, (p0[0], top), (p0[0] + tw + 12, top + th + 12), colour, -1)
            cv2.putText(
                canvas, label, (p0[0] + 6, top + th + 4), _FONT, 0.6, (0, 0, 0), 2, cv2.LINE_AA
            )

        # Tracks that are coasting have no box this frame; say so explicitly
        # rather than letting them vanish from the display. Stacked upward from
        # the bottom, one line each — a shared y made them overprint each other
        # into an unreadable smear as soon as two were occluded at once.
        coasting_tracks = [s for s in snapshots if s.state is TrackState.COASTING]
        for row, occluded in enumerate(coasting_tracks[: self._max_coasting_lines]):
            cv2.putText(
                canvas,
                f"ID {occluded.global_id} occluded {occluded.frames_since_measurement}f",
                (12, canvas.shape[0] - 14 - 22 * row),
                _FONT,
                0.55,
                id_color(occluded.global_id),
                1,
                cv2.LINE_AA,
            )
        if len(coasting_tracks) > self._max_coasting_lines:
            cv2.putText(
                canvas,
                f"+{len(coasting_tracks) - self._max_coasting_lines} more occluded",
                (12, canvas.shape[0] - 14 - 22 * self._max_coasting_lines),
                _FONT,
                0.5,
                (150, 150, 150),
                1,
                cv2.LINE_AA,
            )

        canvas = self._draw_banner(canvas, snapshots, now)
        if self._bev is not None:
            canvas = self._attach_bev(canvas, snapshots)
        return canvas

    def _draw_banner(
        self, canvas: Image, snapshots: list[GlobalTrackSnapshot], now: float
    ) -> Image:
        live = [s for s in snapshots if s.state is not TrackState.COASTING]
        longest = max(
            (self.timeline.held_seconds(s.global_id, now) for s in live), default=0.0
        )
        held_id = None
        if live:
            held_id = max(live, key=lambda s: self.timeline.held_seconds(s.global_id, now))

        parts = [f"{self.wall_fps:4.1f} FPS", f"tracks {len(snapshots)}"]
        if held_id is not None:
            parts.append(f"ID {held_id.global_id} held {longest:.0f}s")
        if self.timeline.reacquired_gap:
            gid, gap = max(self.timeline.reacquired_gap.items(), key=lambda kv: kv[1])
            parts.append(f"ID {gid} reacquired after {gap:.1f}s gap")
        if self.manager.dormant.n_resurrected:
            parts.append(f"resurrections {self.manager.dormant.n_resurrected}")
        if not self.metric:
            parts.append("uncalibrated (no BEV)")

        strip = np.full((40, canvas.shape[1], 3), 18, dtype=np.uint8)
        cv2.putText(
            strip, "   |   ".join(parts), (12, 27), _FONT, 0.58, TEXT_COLOR, 1, cv2.LINE_AA
        )
        hint = "q quit   s save clip"
        (tw, _), _ = cv2.getTextSize(hint, _FONT, 0.5, 1)
        cv2.putText(
            strip, hint, (strip.shape[1] - tw - 12, 27), _FONT, 0.5, (140, 140, 140), 1,
            cv2.LINE_AA,
        )
        return np.vstack([canvas, strip])

    def _attach_bev(self, canvas: Image, snapshots: list[GlobalTrackSnapshot]) -> Image:
        assert self._bev is not None
        panel = self._bev.render(snapshots, self.frame_index)
        target_h = canvas.shape[0]
        scale = target_h / panel.shape[0]
        panel = cv2.resize(
            panel, (int(panel.shape[1] * scale), target_h), interpolation=cv2.INTER_AREA
        )
        return np.hstack([canvas, np.asarray(panel, dtype=np.uint8)])

    def save_clip(self, out_dir: Path, fps: float) -> Path | None:
        """Write the rolling buffer to an mp4. Returns None if nothing buffered.

        ``fps`` must be the rate the frames were *captured* at
        (:attr:`wall_fps`), not the processing throughput: the buffer holds one
        entry per loop iteration, so writing it at the faster processing rate
        makes the clip play back time-compressed.
        """
        if not self.clip:
            return None
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"live_{int(time.time())}.mp4"
        height, width = self.clip[0].shape[:2]
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter.fourcc(*"mp4v"), max(fps, 1.0), (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError(f"could not open video writer for {path}")
        try:
            for frame in self.clip:
                writer.write(frame)
        finally:
            writer.release()
        logger.info("saved %d frames -> %s", len(self.clip), path)
        return path
