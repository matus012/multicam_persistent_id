"""`mcreid-hpc-demo` — the narrated showcase render.

One command, one mp4. Three events in order — cross-camera handoff, coast
through total occlusion, resurrection after a dead gap — each detected from the
pipeline's own output and frozen on screen long enough to read.

Nothing here is animation. The scene is procedural, every frame goes through the
real `MultiViewPipeline`, and the captions are anchored to frames where the
manager actually did the thing being claimed. If an event stops happening, the
command fails rather than narrating it anyway.

    uv run mcreid-hpc-demo --out reports/hpc_demo.mp4
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import typer

from mcreid.fusion.global_id import FusionConfig
from mcreid.fusion.types import GlobalTrackSnapshot, ViewObservation
from mcreid.pipeline import MultiViewPipeline
from mcreid.sim.render import ToySceneRenderer
from mcreid.sim.toy import generate_scene, hpc_demo_scene
from mcreid.track.per_view import Detection
from mcreid.utils.logging import setup_logging
from mcreid.utils.seed import seed_everything
from mcreid.viz.bev import BevRenderer
from mcreid.viz.overlay import draw_view
from mcreid.viz.palette import camera_color
from mcreid.viz.story import (
    EventMark,
    caption_bar,
    detect_coast,
    detect_handoff,
    detect_resurrection,
    stamp_watermark,
    text_card,
)

logger = logging.getLogger(__name__)
app = typer.Typer(add_completion=False, help="Render the narrated HPC showcase video.")

Image = npt.NDArray[np.uint8]

WATERMARK = "SYNTHETIC SCENE - no real footage"
TILE = (640, 360)

# Audited claims only, quoted from the current README. The demoted rows keep
# their demoted wording: this card is the project's own claim surface, and a
# showcase that restores a number the README has already walked back is exactly
# the failure the demotions were for.
CEILING_CLAIMS: list[tuple[str, str]] = [
    ("Hero keeps its ID through 2.5 s of four-camera blackout", "4/5 seeds"),
    ("BEV dot alive through the blackout", "75/75 frames, 5/5 seeds"),
    # These two rows report OPPOSITE quantities and must never be formatted
    # alike. 12/15 is how often the identity is recovered; 9/15 is how often the
    # adversarial case FAILS. Printed in the same shape, a viewer reads both as
    # success rates and the worse number looks like the better one — which would
    # re-inflate, on the claim card, the exact result the README just demoted.
    ("Long-gap re-ID, 75 s absence: ID recovered", "3/3 gate seeds, 12/15 sweep"),
    ("Adversarial long-gap: stranger CAPTURES the ID", "0/3 gate seeds, 9/15 sweep"),
    ("Ground homography image->world round-trip", "RMS < 1e-6 m"),
    ("Runtime, seven 1080p streams, RTX 4060 Laptop", "7.7 FPS/cam, 1.10 aggregate"),
]

HPC_ASK: list[tuple[str, str]] = [
    ("Zero training by us today: YOLO11x on COCO, OSNet on MSMT17", ""),
    ("Neither has seen any evaluation data used here", ""),
    ("", ""),
    ("Domain ReID fine-tune on captured multi-camera footage", "80-120 H200 GPU-hours"),
]


def _tile(panel: Image, colour: tuple[int, int, int], label: str) -> Image:
    """One camera tile: fit to TILE, colour-matched border, big label."""
    out: Image = cv2.resize(panel, TILE, interpolation=cv2.INTER_AREA).astype(np.uint8)
    cv2.rectangle(out, (0, 0), (TILE[0] - 1, TILE[1] - 1), colour, 4)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.82, 2)
    cv2.rectangle(out, (10, 10), (22 + tw, 26 + th), (12, 12, 14), -1)
    cv2.putText(
        out, label, (16, 20 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.82, colour, 2, cv2.LINE_AA
    )
    return out


def _mosaic(tiles: list[Image], bev: Image) -> Image:
    """2x2: three camera tiles plus the BEV, every tile a quarter of the frame."""
    if bev.shape[:2] == (TILE[1], TILE[0]):
        bev_tile: Image = bev.copy()
    else:
        bev_tile = cv2.resize(bev, TILE, interpolation=cv2.INTER_AREA).astype(np.uint8)
    cv2.rectangle(bev_tile, (0, 0), (TILE[0] - 1, TILE[1] - 1), (90, 90, 96), 4)
    top = np.hstack([tiles[0], tiles[1]])
    bottom = np.hstack([tiles[2], bev_tile])
    return np.vstack([top, bottom])


def _write_video(frames: list[Image], path: Path, fps: float) -> Path:
    if not frames:
        raise ValueError("no frames to write")
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


def _write_highlight_gif(
    narrated: list[Image],
    marks: list[EventMark],
    path: Path,
    fps: float,
    lead_s: float = 1.5,
    tail_s: float = 0.6,
    stride: int = 3,
    width: int = 720,
) -> Path:
    """A condensed GIF: a short run-up to each event, then its freeze.

    Not the whole clip. A 57 s render at any usable resolution is tens of MB as
    a GIF, and this repository whitelists embedded assets one file at a time
    precisely because a careless one cost it 5.9 MB of history. The README that
    embeds this says plainly that it is an excerpt and links the command that
    rebuilds the full video.
    """
    try:
        import imageio.v2 as imageio
    except ImportError as exc:  # pragma: no cover - dev extra
        raise RuntimeError("GIF export needs the dev extra: uv pip install -e '.[dev]'") from exc

    lead, tail = int(round(lead_s * fps)), int(round(tail_s * fps))
    keep: list[int] = []
    for mark in marks:
        # `narrated` still holds the freeze copies, so the window that follows an
        # event lands on its frozen caption rather than running past it.
        start = max(0, mark.rendered_index - lead)
        end = min(len(narrated), mark.rendered_index + int(round(mark.hold_s * fps)) + tail)
        keep.extend(range(start, end, stride))

    if not keep:
        raise ValueError("no highlight frames selected")
    scale = width / narrated[0].shape[1]
    height = int(narrated[0].shape[0] * scale)
    frames: list[Any] = [
        cv2.cvtColor(
            cv2.resize(narrated[i], (width, height), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2RGB,
        )
        for i in keep
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, duration=stride / fps, loop=0)
    return path


def _build_marks(
    snapshots: list[list[GlobalTrackSnapshot]], hero: int, config: FusionConfig, fps: float
) -> list[EventMark]:
    """Detect the three events. Raises if the pipeline did not produce one."""
    marks: list[EventMark] = []

    handoff = detect_handoff(snapshots, hero)
    if handoff is None:
        raise RuntimeError("no cross-camera handoff detected — nothing to narrate for event 1")
    frame, before, after = handoff
    marks.append(
        EventMark(
            frame=frame,
            title="1 - CROSS-CAMERA HANDOFF",
            detail=(
                f"supporting cameras {'+'.join(before) or 'none'} -> {'+'.join(after)}"
                f"   |   global ID {hero} unchanged"
            ),
            global_id=hero,
        )
    )

    coast = detect_coast(snapshots, hero, min_frames=int(round(fps)))
    if coast is None:
        raise RuntimeError("no surviving coast detected — nothing to narrate for event 2")
    start, end = coast
    marks.append(
        EventMark(
            frame=(start + end) // 2,
            title="2 - TOTAL OCCLUSION, ID HELD",
            detail=(
                f"hidden from every camera for {(end - start + 1) / fps:.1f} s   |   "
                f"coasting on the motion model, still ID {hero}"
            ),
            global_id=hero,
        )
    )

    resurrection = detect_resurrection(snapshots, hero, min_gap=config.reid_window_frames)
    if resurrection is None:
        raise RuntimeError("no resurrection detected — nothing to narrate for event 3")
    last_seen, back = resurrection
    gap_s = (back - last_seen) / fps
    window_s = config.reid_window_frames / fps
    marks.append(
        EventMark(
            frame=back,
            title="3 - RESURRECTED, SAME ID",
            detail=(
                f"unobserved {gap_s:.1f} s - the {window_s:.0f} s re-association window expired "
                f"{gap_s - window_s:.1f} s before the return   |   "
                f"recovered from the dormant gallery as ID {hero}"
            ),
            global_id=hero,
            hold_s=2.2,
        )
    )
    return marks


@app.command()
def render(
    out: Path = typer.Option(Path("reports/hpc_demo.mp4"), help="Output mp4 path."),
    gif: Path = typer.Option(
        None, help="Also write a condensed highlights GIF (README-sized) here."
    ),
    seed: int = typer.Option(42, help="RNG seed."),
    fps: float = typer.Option(30.0, help="Scene and output frame rate."),
    log_level: str = typer.Option("INFO", help="Logging level."),
) -> None:
    """Render the three-event showcase to an mp4."""
    setup_logging(log_level)
    seed_everything(seed)

    config = hpc_demo_scene(fps=fps, seed=seed, image_size=TILE)
    scene = generate_scene(config)
    fusion = FusionConfig()
    pipeline = MultiViewPipeline(scene.rig, fusion)
    renderer = ToySceneRenderer(scene)
    # Rendered AT tile size, not rendered square and resized into it: a 720x720
    # plan squeezed into a 16:9 tile is scaled 2x harder vertically than
    # horizontally, so the room, the frustums and the trails all come out
    # flattened. A metric map that is not to scale is worse than no map.
    bev = BevRenderer(scene.rig, canvas_size=TILE)
    order = list(config.camera_ids)
    positions = {c.camera_id: (c.position_m[0], c.position_m[1]) for c in config.cameras}

    dt = 1.0 / config.fps
    snapshots: list[list[GlobalTrackSnapshot]] = []
    panels: list[Image] = []

    for index in range(scene.n_frames):
        detections = {
            camera_id: [Detection(d.bbox_xyxy, d.score, d.embedding) for d in dets]
            for camera_id, dets in scene.frame_detections(index).items()
        }
        snaps = pipeline.step(detections, index, dt)
        snapshots.append(snaps)

        assignment = pipeline.manager.last_assignment
        tiles: list[Image] = []
        for position, camera_id in enumerate(order):
            tracker = pipeline.trackers[camera_id]
            observations = [
                ViewObservation(
                    camera_id=camera_id,
                    frame=index,
                    local_track_id=t.track_id,
                    bbox_xyxy=t.box.copy(),
                    embedding=t.embedding.copy(),
                    score=float(np.clip(t.score, 0.0, 1.0)),
                )
                for t in tracker.tracks
                if t.confirmed and t.time_since_update == 0
            ]
            view = draw_view(
                renderer.frame(camera_id, index),
                observations,
                {
                    o.local_track_id: assignment[(camera_id, o.local_track_id)]
                    for o in observations
                    if (camera_id, o.local_track_id) in assignment
                },
                camera_id,
                None,
                occluded=not scene.frame_detections(index)[camera_id],
            )
            tiles.append(_tile(view, camera_color(position), camera_id))
        panels.append(_mosaic(tiles, bev.render(snaps, index, positions, order)))

    # --- which ID is the hero? The one the pipeline gave agent 1, by position.
    gt = scene.gt_world[1]
    votes: dict[int, int] = {}
    for index, snaps in enumerate(snapshots):
        for snap in snaps:
            if float(np.linalg.norm(snap.world_xy - gt[index])) < 0.8:
                votes[snap.global_id] = votes.get(snap.global_id, 0) + 1
    if not votes:
        raise RuntimeError("no global track ever tracked the hero — nothing to render")
    hero = max(votes, key=lambda k: votes[k])
    marks = _build_marks(snapshots, hero, fusion, config.fps)

    reported = sorted({s.global_id for snaps in snapshots for s in snaps})
    logger.info("hero global id %d; all reported ids %s", hero, reported)
    if len(reported) > len(config.agents):
        logger.warning(
            "DUPLICATE IDENTITIES IN THE RENDER: %d ids reported for %d people (%s). "
            "This is real pipeline behaviour on this input and is NOT being hidden.",
            len(reported),
            len(config.agents),
            reported,
        )

    # --- assemble: title card, narrated body with freezes, end cards.
    width, height = panels[0].shape[1], panels[0].shape[0]
    bar_h = 172
    size = (width, height + bar_h)
    hold = int(round(2.6 * config.fps))

    video: list[Image] = [
        text_card(
            size,
            "mcreid - persistent ID across cameras",
            [
                ("Procedural synthetic scene, 3 cameras, 2 people", ""),
                ("Every frame runs the real multi-view pipeline", ""),
                ("Captions are detected from pipeline output, not scripted", ""),
            ],
            footer="No real or dataset footage appears in this video.",
        )
    ] * hold

    by_frame = {m.frame: m for m in marks}
    placed: list[EventMark] = []
    active: EventMark | None = None
    for index, panel in enumerate(panels):
        mark = by_frame.get(index)
        if mark is not None:
            active = mark
        title = active.title if active else ""
        detail = active.detail if active else ""
        frame = np.vstack([panel, caption_bar(width, title, detail, bar_h)])
        stamp_watermark(frame, WATERMARK)
        if mark is not None:
            placed.append(replace(mark, rendered_index=len(video)))
        video.append(frame)
        if mark is not None:
            video.extend([frame.copy()] * int(round(mark.hold_s * config.fps)))

    video.extend(
        [
            text_card(
                size,
                "Measured, and audited against the README",
                CEILING_CLAIMS,
                footer="Synthetic gates. Real four-camera footage is not captured yet.",
            )
        ]
        * int(round(7.0 * config.fps))
    )
    video.extend(
        [text_card(size, "What HPC buys", HPC_ASK, footer="TUKE PERUN, 208x NVIDIA H200.")]
        * int(round(5.0 * config.fps))
    )

    path = _write_video(video, out, config.fps)
    if gif is not None:
        gif_path = _write_highlight_gif(video, placed, gif, config.fps)
        typer.echo(f"wrote {gif_path}  |  {gif_path.stat().st_size / 1e6:.1f} MB")
    seconds = len(video) / config.fps
    size_mb = path.stat().st_size / 1e6
    typer.echo(f"wrote {path}  |  {len(video)} frames  {seconds:.1f} s  {size_mb:.1f} MB")
    for mark in marks:
        typer.echo(f"  event @ frame {mark.frame:5}: {mark.title} - {mark.detail}")
    if len(reported) > len(config.agents):
        typer.echo(f"  WARNING: {len(reported)} global ids for {len(config.agents)} people")


if __name__ == "__main__":
    app()
