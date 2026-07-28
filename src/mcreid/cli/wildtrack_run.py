"""`mcreid-wildtrack` — run the full pipeline on real WILDTRACK footage.

Three commands, meant to be run in this order:

    calib-report   cross-check the WILDTRACK -> calib.json converter
    run            per-view detection + tracking + fusion over a clip, with
                   annotated per-camera video and a BEV, plus honest metrics
    (mcreid-eval)  the MODA/MODP protocol row for G-M1-3

Everything here is labelled *geometric baseline, zero training*: a COCO-pretrained
detector, an ImageNet-pretrained appearance trunk, and geometry. Nothing in this
pipeline has seen WILDTRACK.
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

from mcreid.eval.id_metrics import evaluate_id_consistency
from mcreid.eval.wildtrack import load_annotations, load_rig
from mcreid.eval.wildtrack_report import check_conversion
from mcreid.fusion.associate import AssociationConfig
from mcreid.fusion.dormant import DormantConfig
from mcreid.fusion.global_id import FusionConfig, GlobalIDManager
from mcreid.fusion.types import TrackState, ViewObservation
from mcreid.track.gpu_view import GpuPerViewBackend, GpuViewConfig
from mcreid.utils.logging import get_logger, setup_logging
from mcreid.utils.seed import DEFAULT_SEED, seed_everything
from mcreid.viz.bev import BevRenderer
from mcreid.viz.calib_overlay import draw_floor_grid
from mcreid.viz.mosaic import compose
from mcreid.viz.overlay import draw_view

logger = get_logger(__name__)
app = typer.Typer(add_completion=False, help="Run the tracker on real WILDTRACK footage.")

Image = npt.NDArray[np.uint8]
FloatArray = npt.NDArray[np.float64]

# WILDTRACK's Image_subsets dirs are C1..C7 in the same order as the calibration
# files (CVLab1-4, IDIAP1-3). The converter keeps that order.
IMAGE_DIR_PREFIX = "C"
ANNOTATED_FRAME_STRIDE = 5  # annotations are every 5th frame index


def _frame_paths(root: Path, camera_index: int) -> list[Path]:
    directory = root / "Image_subsets" / f"{IMAGE_DIR_PREFIX}{camera_index + 1}"
    if not directory.is_dir():
        raise typer.BadParameter(f"missing image directory: {directory}")
    return sorted(directory.glob("*.png"))


def _frame_number(path: Path) -> int:
    """WILDTRACK's real frame index, parsed from the filename.

    Files are 00000000.png, 00000005.png, ... — the annotated subset of a 60 fps
    capture. Using a *positional* index to look up annotations silently pairs
    image i with annotation i instead of annotation 5i, which lines up perfectly
    at frame 0 and drifts from there.
    """
    return int(path.stem)


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


def _write_gif(frames: list[Image], path: Path, fps: float, stride: int, width: int) -> Path:
    import imageio.v2 as imageio

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


@app.command("calib-report")
def calib_report(
    root: Path = typer.Option(Path("data/wildtrack_full"), help="WILDTRACK root."),
    out_dir: Path = typer.Option(Path("reports/wildtrack/calib"), help="Report output."),
    max_frames: int = typer.Option(20, help="Annotated frames to sample."),
    grid_step_m: float = typer.Option(1.0, help="Floor grid spacing in the overlay."),
    log_level: str = typer.Option("INFO"),
) -> None:
    """Validate the WILDTRACK -> calib.json converter (not the dataset itself)."""
    setup_logging(log_level)
    if not root.is_dir():
        raise typer.BadParameter(
            f"{root} not found. Run: python scripts/download_wildtrack.py fetch"
        )

    rig = load_rig(root / "calibrations")
    typer.echo(f"converted {len(rig.cameras)} cameras: {rig.camera_ids}")
    x0, y0, x1, y1 = rig.floor_extent()
    typer.echo(f"floor extent: x [{x0:.1f}, {x1:.1f}] m, y [{y0:.1f}, {y1:.1f}] m")

    report = check_conversion(rig, root / "annotations_positions", max_frames=max_frames)
    for cam in report.cameras:
        status = "OK" if cam.ok else "FAIL"
        typer.echo(
            f"  [{status}] {cam.camera_id}: {cam.n_samples} samples, "
            f"median {cam.median_px:.1f} px, p90 {cam.p90_px:.1f} px"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    for index, calib in enumerate(rig.cameras):
        frames = _frame_paths(root, index)
        raw = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
        if raw is None:
            continue
        overlay = draw_floor_grid(
            np.asarray(raw, dtype=np.uint8), calib, step_m=grid_step_m
        )
        cv2.imwrite(str(out_dir / f"{calib.camera_id}_grid.png"), overlay)

    report.to_json(out_dir / "summary.json")
    markdown = report.to_markdown(out_dir / "summary.md")
    typer.echo(f"\nwrote floor-grid overlays and {markdown}")
    typer.echo(
        "NOTE: this validates OUR CONVERTER against WILDTRACK's own annotations. "
        "It cannot detect an error in WILDTRACK's published calibration, because "
        "both sides of the comparison come from WILDTRACK."
    )
    if not report.ok:
        raise typer.Exit(code=1)


@app.command()
def run(
    root: Path = typer.Option(Path("data/wildtrack_full"), help="WILDTRACK root."),
    out_dir: Path = typer.Option(Path("reports/wildtrack/run"), help="Where to write output."),
    start: int = typer.Option(0, help="First frame index."),
    n_frames: int = typer.Option(120, help="How many frames to process."),
    fps: float = typer.Option(2.0, help="Annotated-frame rate (WILDTRACK samples at 2 fps)."),
    weights: Path = typer.Option(Path("weights/yolo11x.pt"), help="Detector weights."),
    imgsz: int = typer.Option(1280, help="Detector input size."),
    conf: float = typer.Option(0.25, help="Detection confidence floor."),
    match_radius_m: float = typer.Option(1.0, help="GT<->prediction match radius."),
    geometry_only: bool = typer.Option(
        False,
        help=(
            "Disable the appearance gate entirely and fuse on ground geometry alone. "
            "Isolates what the geometric baseline is worth when the untrained "
            "embedder contributes nothing."
        ),
    ),
    export_video: bool = typer.Option(True, help="Write annotated mosaic video + GIF."),
    seed: int = typer.Option(DEFAULT_SEED),
    log_level: str = typer.Option("INFO"),
) -> None:
    """Run detection + tracking + fusion over a WILDTRACK clip and score it."""
    setup_logging(log_level)
    seed_everything(seed)
    if not root.is_dir():
        raise typer.BadParameter(
            f"{root} not found. Run: python scripts/download_wildtrack.py fetch"
        )

    rig = load_rig(root / "calibrations")
    annotations = load_annotations(
        root / "annotations_positions", camera_ids=rig.camera_ids
    )
    per_camera_paths = [_frame_paths(root, i) for i in range(len(rig.cameras))]
    available = min(len(paths) for paths in per_camera_paths)
    n_frames = min(n_frames, available - start)
    if n_frames <= 0:
        raise typer.BadParameter(f"no frames available from index {start} (have {available})")

    backends = {
        cam.camera_id: GpuPerViewBackend(
            cam.camera_id,
            GpuViewConfig(weights=weights, imgsz=imgsz, conf_threshold=conf),
        )
        for cam in rig.cameras
    }
    if geometry_only:
        # Measured on WILDTRACK: the untrained ImageNet embedder separates
        # same-person-cross-camera (0.377) from different-person-cross-camera
        # (0.408) by 0.03 cosine. That is not a threshold-tuning problem — the
        # distributions overlap — so this ablation removes appearance from the
        # decision entirely and lets ground geometry stand on its own.
        association = AssociationConfig(
            weight_geometry=1.0,
            weight_appearance=0.0,
            max_appearance_distance=2.0,
        )
        fusion_config = FusionConfig(
            association=association,
            merge_appearance_distance=2.0,
            revive_appearance_distance=2.0,
            dormant=DormantConfig(enabled=False),
        )
        typer.echo("GEOMETRY-ONLY ablation: appearance gate disabled")
    else:
        fusion_config = FusionConfig()
    manager = GlobalIDManager(rig, fusion_config)
    bev = BevRenderer(rig, canvas_size=(720, 720), grid_step_m=2.0)
    dt = 1.0 / fps

    mosaics: list[Image] = []
    snapshots_per_frame = []
    detection_counts: list[int] = []
    timings: list[float] = []

    typer.echo(f"processing {n_frames} frames x {len(rig.cameras)} cameras ...")
    for offset in range(n_frames):
        slot = start + offset
        index = _frame_number(per_camera_paths[0][slot])
        started = time.perf_counter()

        views: list[ViewObservation] = []
        images: dict[str, Image] = {}
        for cam_index, cam in enumerate(rig.cameras):
            path = per_camera_paths[cam_index][slot]
            raw = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if raw is None:
                raise OSError(f"could not read {path}")
            image: Image = np.asarray(raw, dtype=np.uint8)
            images[cam.camera_id] = image
            views.extend(backends[cam.camera_id].step(image, index))

        snapshots = manager.step(views, index, dt)
        timings.append(time.perf_counter() - started)
        snapshots_per_frame.append(snapshots)
        detection_counts.append(len(views))

        if export_video:
            assignment = manager.last_assignment
            panels = {}
            for cam in rig.cameras:
                observations = [v for v in views if v.camera_id == cam.camera_id]
                mapping = {
                    obs.local_track_id: assignment[(cam.camera_id, obs.local_track_id)]
                    for obs in observations
                    if (cam.camera_id, obs.local_track_id) in assignment
                }
                panels[cam.camera_id] = draw_view(
                    images[cam.camera_id], observations, mapping, cam.camera_id, index
                )
            caption = (
                f"WILDTRACK frame {index} | live global IDs: {len(snapshots)} | "
                f"detections: {len(views)} | geometric baseline, zero training"
            )
            mosaics.append(
                compose(
                    panels,
                    bev.render(snapshots, index),
                    camera_order=list(rig.camera_ids)[:4],
                    caption=caption,
                )
            )
        if offset % 20 == 0:
            typer.echo(
                f"  frame {index}: {len(views)} view tracks, {len(snapshots)} global IDs, "
                f"{timings[-1] * 1000:.0f} ms"
            )

    # --- ground truth over the same frames ---
    frame_indices = [_frame_number(per_camera_paths[0][start + i]) for i in range(n_frames)]
    person_ids = sorted(
        {r.person_id for f in frame_indices for r in annotations.get(f, [])}
    )
    gt_world = {
        pid: np.full((n_frames, 2), np.nan, dtype=np.float64) for pid in person_ids
    }
    gt_visible = {
        pid: np.zeros((n_frames, len(rig.cameras)), dtype=bool) for pid in person_ids
    }
    for slot, frame in enumerate(frame_indices):
        for record in annotations.get(frame, []):
            gt_world[record.person_id][slot] = record.world_xy
            for cam_index, camera_id in enumerate(rig.camera_ids):
                if record.bboxes.get(camera_id) is not None:
                    gt_visible[record.person_id][slot, cam_index] = True

    report = evaluate_id_consistency(
        gt_world=gt_world,
        gt_visible=gt_visible,
        results=snapshots_per_frame,
        n_ids_issued=manager.n_ids_issued,
        match_radius_m=match_radius_m,
    )

    median_ms = float(np.median(timings)) * 1000.0
    # `global_ids_minted` counts every birth, including tentative tracks that die
    # before ever being reported. In a crowd that number is dominated by
    # short-lived candidates and badly overstates identity churn, so the count of
    # IDs that were actually *shown* is reported alongside it.
    visible_ids = {s.global_id for snaps in snapshots_per_frame for s in snaps}
    live_per_frame = [len(snaps) for snaps in snapshots_per_frame]
    summary = {
        "label": "geometric baseline, zero training",
        "mode": "geometry_only" if geometry_only else "geometry+appearance",
        "frames": n_frames,
        "cameras": len(rig.cameras),
        "ground_truth_people": len(person_ids),
        "global_ids_minted": manager.n_ids_issued,
        "global_ids_ever_reported": len(visible_ids),
        "mean_live_ids_per_frame": float(np.mean(live_per_frame)),
        "total_id_switches": report.total_id_switches,
        "id_switches_per_person": report.total_id_switches / max(len(person_ids), 1),
        "mean_detections_per_frame": float(np.mean(detection_counts)),
        "position_rmse_m": report.position_rmse_m,
        "coverage_visible": {k: round(v, 4) for k, v in report.coverage_visible.items()},
        "false_positive_tracks": report.false_positive_tracks,
        "median_ms_per_frame_all_cameras": median_ms,
        "aggregate_fps": 1000.0 / median_ms if median_ms else float("nan"),
        "per_camera_fps": len(rig.cameras) * 1000.0 / median_ms if median_ms else float("nan"),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    typer.echo("\n" + json.dumps(summary, indent=2))
    typer.echo(f"\nID switches per GT person: {dict(report.id_switches)}")

    if export_video and mosaics:
        video = _write_video(mosaics, out_dir / "wildtrack.mp4", fps)
        typer.echo(f"wrote {video}")
        gif = _write_gif(mosaics, out_dir / "wildtrack.gif", fps, 2, 1100)
        typer.echo(f"wrote {gif}")

    coasting = sum(
        1 for snaps in snapshots_per_frame for s in snaps if s.state is TrackState.COASTING
    )
    typer.echo(f"coasting snapshots: {coasting}")


if __name__ == "__main__":  # pragma: no cover
    app()
