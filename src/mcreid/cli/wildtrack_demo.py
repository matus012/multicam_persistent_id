"""`mcreid-wildtrack-demo` — the human-watchable WILDTRACK demo.

Renders real camera frames with detection boxes and global-ID labels, three
views plus the bird's-eye map, and picks the segment containing the most
cross-camera handoffs for the GIF. The point of the artefact is that a viewer
can watch one number and one colour follow a person from camera to camera.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import typer

from mcreid.eval.wildtrack import load_rig
from mcreid.fusion.global_id import FusionConfig, GlobalIDManager
from mcreid.fusion.types import TrackState, ViewObservation
from mcreid.track.gpu_view import GpuPerViewBackend, GpuViewConfig
from mcreid.track.reid_models import DEFAULT_EMBEDDER
from mcreid.utils.logging import get_logger, setup_logging
from mcreid.utils.seed import DEFAULT_SEED, seed_everything
from mcreid.viz.bev import BevRenderer
from mcreid.viz.demo_mosaic import compose_demo, draw_demo_view, find_handoff_segments

logger = get_logger(__name__)
app = typer.Typer(add_completion=False, help="Render the WILDTRACK demo video and GIF.")

Image = npt.NDArray[np.uint8]


def _write_video(frames: list[Image], path: Path, fps: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {path}")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()
    return path


def _write_gif(frames: list[Image], path: Path, fps: float, width: int, max_mb: float) -> Path:
    """Write a GIF, shrinking until it fits ``max_mb``."""
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt_width in (width, int(width * 0.85), int(width * 0.7), int(width * 0.55)):
        scale = attempt_width / frames[0].shape[1]
        resized: list[Any] = [
            cv2.cvtColor(
                cv2.resize(
                    f,
                    (attempt_width, int(f.shape[0] * scale)),
                    interpolation=cv2.INTER_AREA,
                ),
                cv2.COLOR_BGR2RGB,
            )
            for f in frames
        ]
        imageio.mimsave(path, resized, duration=1.0 / fps, loop=0)
        size_mb = path.stat().st_size / 1e6
        logger.info("GIF at width %d: %.1f MB", attempt_width, size_mb)
        if size_mb <= max_mb:
            return path
    logger.warning("GIF still %.1f MB after shrinking", path.stat().st_size / 1e6)
    return path


@app.command()
def render(
    root: Path = typer.Option(Path("data/wildtrack_full"), help="WILDTRACK root."),
    out_dir: Path = typer.Option(Path("reports"), help="Output directory."),
    start: int = typer.Option(0, help="First frame slot."),
    n_frames: int = typer.Option(120, help="Frames to process."),
    cameras: str = typer.Option("", help="Comma-separated camera ids; default = first 3."),
    fps: float = typer.Option(2.0, help="Source annotated-frame rate."),
    playback_fps: float = typer.Option(6.0, help="Playback rate of the exported video."),
    panel_w: int = typer.Option(640, help="Width of each camera panel."),
    gif_seconds: float = typer.Option(10.0, help="Length of the exported GIF segment."),
    gif_width: int = typer.Option(1000, help="GIF width in pixels."),
    gif_max_mb: float = typer.Option(15.0, help="Hard size ceiling for the GIF."),
    embedder: str = typer.Option(DEFAULT_EMBEDDER),
    weights: Path = typer.Option(Path("weights/yolo11x.pt")),
    seed: int = typer.Option(DEFAULT_SEED),
    log_level: str = typer.Option("INFO"),
) -> None:
    """Run the tracker and export a watchable demo video plus a README GIF."""
    setup_logging(log_level)
    seed_everything(seed)
    if not root.is_dir():
        raise typer.BadParameter(f"{root} not found")

    rig = load_rig(root / "calibrations")
    chosen = [c.strip() for c in cameras.split(",") if c.strip()] or list(rig.camera_ids)[:3]
    unknown = set(chosen) - set(rig.camera_ids)
    if unknown:
        raise typer.BadParameter(f"unknown cameras {sorted(unknown)}; have {rig.camera_ids}")

    index_of = {c: i for i, c in enumerate(rig.camera_ids)}
    paths = {
        camera_id: sorted((root / "Image_subsets" / f"C{index_of[camera_id] + 1}").glob("*.png"))
        for camera_id in chosen
    }
    available = min(len(v) for v in paths.values())
    n_frames = min(n_frames, available - start)

    backends = {
        camera_id: GpuPerViewBackend(
            camera_id, GpuViewConfig(weights=weights, embedder=embedder)
        )
        for camera_id in chosen
    }
    # Fuse only over the cameras being shown, so the BEV and the panels describe
    # the same system — a track supported by an off-screen camera would look
    # like it appeared from nowhere.
    subset = rig.model_copy(update={"cameras": [rig.get(c) for c in chosen]})
    manager = GlobalIDManager(subset, FusionConfig())
    bev = BevRenderer(subset, canvas_size=(760, 760), grid_step_m=2.0, trail_length=14)
    panel_size = (panel_w, int(panel_w * 9 / 16))
    dt = 1.0 / fps

    cached: list[dict[str, Any]] = []
    snapshots_per_frame = []
    timings = []

    typer.echo(f"tracking {n_frames} frames over {chosen} ...")
    for offset in range(n_frames):
        slot = start + offset
        index = int(paths[chosen[0]][slot].stem)
        started = time.perf_counter()

        views: list[ViewObservation] = []
        images: dict[str, Image] = {}
        for camera_id in chosen:
            raw = cv2.imread(str(paths[camera_id][slot]), cv2.IMREAD_COLOR)
            if raw is None:
                raise OSError(f"could not read {paths[camera_id][slot]}")
            image: Image = np.asarray(raw, dtype=np.uint8)
            images[camera_id] = image
            views.extend(backends[camera_id].step(image, index))

        snaps = manager.step(views, index, dt)
        timings.append(time.perf_counter() - started)
        snapshots_per_frame.append(snaps)
        cached.append(
            {
                "index": index,
                "images": images,
                "views": views,
                "assignment": dict(manager.last_assignment),
                "snaps": snaps,
            }
        )
        if offset % 20 == 0:
            typer.echo(f"  {offset}/{n_frames}  live IDs: {len(snaps)}")

    segment_len = max(int(gif_seconds * fps), 8)
    segments = find_handoff_segments(snapshots_per_frame, segment_len)
    if segments:
        score, seg_start, seg_end, hero = segments[0]
        typer.echo(
            f"best handoff segment: frames {seg_start}-{seg_end}, global ID {hero}, "
            f"{score} handoff event(s)"
        )
    else:
        score, seg_start, seg_end, hero = 0, 0, min(segment_len, n_frames), -1
        typer.echo("no clean cross-camera handoff found; exporting the opening segment")

    frames_out: list[Image] = []
    for position, item in enumerate(cached):
        highlight = hero if seg_start <= position < seg_end else None
        panels = {}
        for camera_id in chosen:
            observations = [v for v in item["views"] if v.camera_id == camera_id]
            mapping = {
                obs.local_track_id: item["assignment"][(camera_id, obs.local_track_id)]
                for obs in observations
                if (camera_id, obs.local_track_id) in item["assignment"]
            }
            panels[camera_id] = draw_demo_view(
                item["images"][camera_id],
                observations,
                mapping,
                camera_id,
                panel_size,
                highlight=highlight,
            )
        multi = [s for s in item["snaps"] if len(s.supporting_cameras) >= 2]
        coasting = [s for s in item["snaps"] if s.state is TrackState.COASTING]
        caption = (
            f"WILDTRACK f{item['index']}   global IDs live: {len(item['snaps'])}   "
            f"fused across 2+ cameras: {len(multi)}   coasting: {len(coasting)}"
        )
        sub = "geometric fusion + off-the-shelf ReID (zero training by us)"
        if highlight is not None:
            sub = f"watch ID {hero} keep its number and colour across cameras   |   " + sub
        frames_out.append(
            compose_demo(
                panels,
                bev.render(item["snaps"], item["index"]),
                chosen,
                panel_size,
                caption=caption,
                subcaption=sub,
            )
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    video = _write_video(frames_out, out_dir / "wildtrack_demo.mp4", playback_fps)
    typer.echo(f"wrote {video}  ({video.stat().st_size / 1e6:.1f} MB)")

    gif_frames = frames_out[seg_start:seg_end]
    gif = _write_gif(
        gif_frames, out_dir / "wildtrack_demo.gif", playback_fps, gif_width, gif_max_mb
    )
    typer.echo(f"wrote {gif}  ({gif.stat().st_size / 1e6:.1f} MB, {len(gif_frames)} frames)")

    for probe in (seg_start, (seg_start + seg_end) // 2, seg_end - 1):
        cv2.imwrite(str(out_dir / f"demo_check_{probe:04d}.png"), frames_out[probe])
    typer.echo(f"wrote 3 check frames to {out_dir} — open them and confirm people are legible")

    (out_dir / "wildtrack_demo.json").write_text(
        json.dumps(
            {
                "cameras": chosen,
                "frames": n_frames,
                "embedder": embedder,
                "gif_segment": [seg_start, seg_end],
                "highlight_global_id": hero,
                "handoff_events_in_segment": score,
                "median_ms_per_frame": float(np.median(timings)) * 1000.0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":  # pragma: no cover
    app()
