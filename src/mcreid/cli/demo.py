"""`mcreid-demo` — run the tracker and export the 4-view + BEV demo.

Modes:
    synthetic   scripted toy scene, no footage or GPU required (works today)
    recorded    four video files + a calib.json  (needs footage; G-M1-2)
    live        webcams                          (needs footage/hardware; G-M1-2)
"""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import typer

from mcreid.calib.schema import RigCalib
from mcreid.eval.id_metrics import evaluate_id_consistency
from mcreid.fusion.global_id import FusionConfig
from mcreid.fusion.types import TrackState, ViewObservation
from mcreid.pipeline import MultiViewPipeline
from mcreid.sim.render import ToySceneRenderer
from mcreid.sim.toy import cardboard_scene, crossing_scene, generate_scene
from mcreid.track.per_view import Detection
from mcreid.utils.logging import get_logger, setup_logging
from mcreid.utils.seed import DEFAULT_SEED, seed_everything
from mcreid.viz.bev import BevRenderer
from mcreid.viz.mosaic import compose
from mcreid.viz.overlay import draw_view

logger = get_logger(__name__)
app = typer.Typer(add_completion=False, help="Multi-camera persistent-ID tracking demo.")


class Scenario(str, Enum):
    CARDBOARD = "cardboard"
    CROSSING = "crossing"


def _write_video(frames: list[np.ndarray], path: Path, fps: float) -> Path:
    if not frames:
        raise ValueError("no frames to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {path}")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()
    return path


def _write_gif(frames: list[np.ndarray], path: Path, fps: float, stride: int, width: int) -> Path:
    try:
        import imageio.v2 as imageio
    except ImportError as exc:  # pragma: no cover - dev extra
        raise RuntimeError("GIF export needs the dev extra: uv pip install -e '.[dev]'") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    selected = frames[::stride]
    scale = width / selected[0].shape[1]
    resized: list[Any] = [
        cv2.cvtColor(
            cv2.resize(f, (width, int(f.shape[0] * scale)), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2RGB,
        )
        for f in selected
    ]
    imageio.mimsave(path, resized, duration=stride / fps, loop=0)
    return path


@app.command()
def synthetic(
    scenario: Scenario = typer.Option(Scenario.CARDBOARD, help="Which scripted scene to run."),
    out_dir: Path = typer.Option(Path("outputs/demo"), help="Where to write video/GIF."),
    seed: int = typer.Option(DEFAULT_SEED, help="RNG seed."),
    fps: float = typer.Option(30.0, help="Scene frame rate."),
    n_frames: int = typer.Option(420, help="Clip length in frames."),
    gif: bool = typer.Option(True, help="Also export a README-sized GIF."),
    gif_stride: int = typer.Option(3, help="Keep every Nth frame in the GIF."),
    gif_width: int = typer.Option(900, help="GIF width in pixels."),
    log_level: str = typer.Option("INFO"),
) -> None:
    """Run the tracker on a scripted scene and export the demo mosaic.

    This is the one command that works before any footage exists — it is how the
    full demo path stays reviewable while G-M1-2 is blocked on the real clips.
    """
    setup_logging(log_level)
    seed_everything(seed)

    config = (
        cardboard_scene(n_frames=n_frames, fps=fps, seed=seed)
        if scenario is Scenario.CARDBOARD
        else crossing_scene(n_frames=n_frames, fps=fps, seed=seed)
    )
    scene = generate_scene(config)
    renderer = ToySceneRenderer(scene)
    pipeline = MultiViewPipeline(scene.rig, FusionConfig())
    bev = BevRenderer(scene.rig)

    camera_positions = {
        vcam.camera_id: (vcam.position_m[0], vcam.position_m[1]) for vcam in config.cameras
    }

    dt = 1.0 / config.fps
    frames: list[np.ndarray] = []
    snapshots_per_frame = []

    for frame_index in range(scene.n_frames):
        detections = {
            camera_id: [Detection(d.bbox_xyxy, d.score, d.embedding) for d in dets]
            for camera_id, dets in scene.frame_detections(frame_index).items()
        }
        snapshots = pipeline.step(detections, frame_index, dt)
        snapshots_per_frame.append(snapshots)

        assignment = pipeline.manager.last_assignment
        views = {}
        for camera_id, tracker in pipeline.trackers.items():
            observations: list[ViewObservation] = [
                ViewObservation(
                    camera_id=camera_id,
                    frame=frame_index,
                    local_track_id=t.track_id,
                    bbox_xyxy=t.box.copy(),
                    embedding=t.embedding.copy(),
                    score=float(np.clip(t.score, 0.0, 1.0)),
                )
                for t in tracker.tracks
                if t.confirmed and t.time_since_update == 0
            ]
            local_to_global = {
                obs.local_track_id: assignment[(camera_id, obs.local_track_id)]
                for obs in observations
                if (camera_id, obs.local_track_id) in assignment
            }
            blocked = not scene.frame_detections(frame_index)[camera_id]
            views[camera_id] = draw_view(
                renderer.frame(camera_id, frame_index),
                observations,
                local_to_global,
                camera_id,
                frame_index,
                occluded=blocked,
            )

        coasting = [s for s in snapshots if s.state is TrackState.COASTING]
        caption = f"{scenario.value}  |  global IDs live: {len(snapshots)}"
        if coasting:
            longest = max(s.frames_since_measurement for s in coasting)
            caption += f"  |  COASTING through occlusion ({longest} frames, ID held)"
        frames.append(
            compose(
                views,
                bev.render(snapshots, frame_index, camera_positions),
                camera_order=list(scene.rig.camera_ids),
                caption=caption,
            )
        )

    report = evaluate_id_consistency(
        gt_world=scene.gt_world,
        gt_visible=scene.gt_visible,
        results=snapshots_per_frame,
        n_ids_issued=pipeline.n_ids_issued,
    )
    typer.echo(report.summary())

    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = _write_video(frames, out_dir / f"{scenario.value}.mp4", config.fps)
    typer.echo(f"wrote {video_path}")
    if gif:
        gif_path = _write_gif(
            frames, out_dir / f"{scenario.value}.gif", config.fps, gif_stride, gif_width
        )
        typer.echo(f"wrote {gif_path}")

    if scenario is Scenario.CARDBOARD:
        switches = report.total_id_switches
        survived = report.longest_coast_survived.get(1, 0)
        verdict = "PASS" if switches == 0 else "FAIL"
        typer.echo(
            f"\ncardboard criterion: {verdict} "
            f"(ID switches={switches}, longest total occlusion survived={survived} frames "
            f"= {survived / config.fps:.2f} s)"
        )
        if switches != 0:
            raise typer.Exit(code=1)


@app.command()
def recorded(
    calib: Path = typer.Option(..., help="Path to calib.json produced by mcreid-calibrate."),
    footage: Path = typer.Option(..., help="Directory holding one video per camera."),
    out_dir: Path = typer.Option(Path("outputs/demo")),
    log_level: str = typer.Option("INFO"),
) -> None:
    """Run on recorded 4-view footage. Requires the perception extra.

    Validates the calibration and footage layout today; the detector front-end
    lands with G-M1-2, which is blocked on the capture session. Everything
    downstream of detection is already exercised by `mcreid-demo synthetic`.
    """
    setup_logging(log_level)
    rig = RigCalib.load(calib)
    logger.info("loaded rig with %d camera(s): %s", len(rig.cameras), rig.camera_ids)

    if not footage.is_dir():
        raise typer.BadParameter(f"footage directory not found: {footage}")
    missing = [c for c in rig.camera_ids if not list(footage.glob(f"{c}.*"))]
    if missing:
        raise typer.BadParameter(f"no video found for camera(s) {missing} in {footage}")
    logger.info("footage layout OK for %s", rig.camera_ids)

    typer.echo(
        "Calibration and footage layout validated.\n"
        "The recorded-mode detector front-end (YOLO11 + BoT-SORT + ReID) lands with "
        "G-M1-2, which is blocked on the capture session — see capture_guide.md.\n"
        "Run `mcreid-demo synthetic` for the full end-to-end path today."
    )
    raise typer.Exit(code=2)


@app.callback()
def main() -> None:
    """Multi-camera persistent-ID tracking."""
    logging.getLogger("mcreid").setLevel(logging.INFO)


if __name__ == "__main__":  # pragma: no cover
    app()
