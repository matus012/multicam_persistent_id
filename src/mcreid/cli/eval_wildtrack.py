"""`mcreid-eval` -- WILDTRACK evaluation for the existing geometric,
zero-training `MultiViewPipeline`.

This is a *geometric baseline*: WILDTRACK's ground-truth per-view bounding
boxes are fed through the unmodified fusion pipeline (calibration -> ground
projection -> Kalman/Hungarian association -> global IDs) using a constant,
uninformative appearance embedding (``_DUMMY_EMBEDDING``) instead of a
trained ReID model. No detector, no ReID network, and no WILDTRACK-specific
training happens anywhere on this path -- the resulting MODA/MODP/identity
numbers measure the calibration + association geometry alone, and must not
be read as comparable to trained multi-view detectors/trackers.

The per-view tracker (`mcreid.track.per_view.PerViewTracker`) assumes
frame-to-frame IoU continuity, which does not hold at WILDTRACK's 2 fps
annotation rate (it was designed for ~30 fps video). This command therefore
runs it with ``n_init=1`` so every detection is accepted immediately instead
of requiring IoU continuity across the resulting large per-frame gaps; the
actual cross-camera identity work is done by the ground-plane fusion stage,
not the per-view stage, in this baseline.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import typer

from mcreid.eval.id_metrics import evaluate_id_consistency
from mcreid.eval.wildtrack import (
    FRAME_SAMPLING_HZ,
    MultiviewDetectionMetrics,
    compute_moda_modp,
    load_annotations,
    load_rig,
)
from mcreid.fusion.global_id import FusionConfig
from mcreid.fusion.types import GlobalTrackSnapshot
from mcreid.pipeline import MultiViewPipeline
from mcreid.track.per_view import Detection, PerViewConfig
from mcreid.utils.logging import get_logger, setup_logging
from mcreid.utils.seed import DEFAULT_SEED, seed_everything

logger = get_logger(__name__)
app = typer.Typer(add_completion=False, help="WILDTRACK evaluation: MODA/MODP + identity metrics.")

_MARKDOWN_ROW_LABEL = "geometric baseline, no multi-view training"

# Published MVDet (Hou et al., ECCV 2020) WILDTRACK numbers. Left unset
# deliberately: this codebase has not verified a specific MODA/MODP figure
# against the paper closely enough to state it as fact, and several
# MVDet-family papers report different numbers for related baselines.
# Fill these in once the paper is checked -- never fabricate a benchmark
# number.
MVDET_WILDTRACK_MODA: float | None = None  # TODO: fill from paper
MVDET_WILDTRACK_MODP: float | None = None  # TODO: fill from paper

# Constant across every detection so the appearance term contributes nothing
# informative (distance to itself is always 0) -- association is driven
# purely by ground-plane geometry, which is the point of a "geometric
# baseline" that does not leak ground-truth identity through a fake embedding.
_DUMMY_EMBEDDING = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def _fmt_metric(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "TODO: fill from paper"


def _json_safe(values: dict[str, float | int]) -> dict[str, float | int | None]:
    """Replace NaN floats with ``null`` -- plain ``json.dumps`` emits a bare
    ``NaN`` token for them, which is not valid JSON and breaks strict parsers.
    NaN legitimately shows up here (e.g. MODP/RMSE are undefined with zero
    matches), so it is mapped to ``null`` rather than suppressed or faked.
    """
    return {
        key: (None if isinstance(value, float) and np.isnan(value) else value)
        for key, value in values.items()
    }


def _write_markdown_row(path: Path, metrics: MultiviewDetectionMetrics) -> None:
    """Write/append a markdown table row next to (TODO-guarded) published numbers."""
    header = "| Method | MODA | MODP | Precision | Recall |\n|---|---|---|---|---|\n"
    mvdet_row = (
        f"| MVDet (Hou et al., ECCV 2020), published | "
        f"{_fmt_metric(MVDET_WILDTRACK_MODA)} | {_fmt_metric(MVDET_WILDTRACK_MODP)} | - | - |\n"
    )
    ours_row = (
        f"| {_MARKDOWN_ROW_LABEL} | "
        f"{metrics.moda:.3f} | {metrics.modp:.3f} | {metrics.precision:.3f} | "
        f"{metrics.recall:.3f} |\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8") as fh:
        if write_header:
            fh.write(header)
            fh.write(mvdet_row)
        fh.write(ours_row)


@app.command()
def run(
    calib_root: Path = typer.Option(
        ...,
        help="WILDTRACK calibrations/ directory (contains intrinsic_zero/ and extrinsic/).",
    ),
    annotations: Path = typer.Option(
        ..., help="WILDTRACK annotations_positions/ directory."
    ),
    out_json: Path = typer.Option(
        Path("outputs/eval/wildtrack_results.json"), help="Result JSON path."
    ),
    out_markdown: Path = typer.Option(
        Path("outputs/eval/wildtrack_results.md"), help="Result markdown path."
    ),
    match_radius_m: float = typer.Option(
        0.5, help="Distance threshold (metres) for MODA/MODP and identity matching."
    ),
    max_frames: int = typer.Option(
        0, help="Debug: only evaluate the first N annotated frames (0 = all)."
    ),
    seed: int = typer.Option(DEFAULT_SEED),
    log_level: str = typer.Option("INFO"),
) -> None:
    """Run the geometric baseline pipeline over WILDTRACK and score it."""
    setup_logging(log_level)
    seed_everything(seed)

    if not calib_root.is_dir():
        typer.echo(
            f"WILDTRACK calibration directory not found: {calib_root}\n"
            "No dataset present. Fetch it with scripts/download_wildtrack.py -- "
            "run `python scripts/download_wildtrack.py info` for instructions."
        )
        raise typer.Exit(code=2)
    if not annotations.is_dir():
        typer.echo(
            f"WILDTRACK annotation directory not found: {annotations}\n"
            "No dataset present. Fetch it with scripts/download_wildtrack.py -- "
            "run `python scripts/download_wildtrack.py info` for instructions."
        )
        raise typer.Exit(code=2)

    try:
        rig = load_rig(calib_root)
    except (FileNotFoundError, ValueError, OSError, KeyError) as exc:
        typer.echo(f"failed to load WILDTRACK calibration from {calib_root}: {exc}")
        raise typer.Exit(code=2) from exc
    logger.info("loaded rig: %s", rig.camera_ids)

    try:
        frame_annotations = load_annotations(annotations, camera_ids=tuple(rig.camera_ids))
    except (FileNotFoundError, ValueError, KeyError) as exc:
        typer.echo(f"failed to load WILDTRACK annotations from {annotations}: {exc}")
        raise typer.Exit(code=2) from exc

    frames = sorted(frame_annotations)
    if max_frames > 0:
        frames = frames[:max_frames]
    if not frames:
        typer.echo(f"no annotated frames found in {annotations}")
        raise typer.Exit(code=2)
    logger.info("evaluating %d annotated frame(s)", len(frames))

    per_view_config = PerViewConfig(n_init=1)
    pipeline = MultiViewPipeline(rig, FusionConfig(), per_view_config)
    dt = 1.0 / FRAME_SAMPLING_HZ

    snapshots_per_frame: list[list[GlobalTrackSnapshot]] = []
    predictions: list[np.ndarray] = []
    ground_truth: list[np.ndarray] = []

    all_person_ids = sorted(
        {ann.person_id for frame in frames for ann in frame_annotations[frame]}
    )
    n_frames = len(frames)
    gt_world: dict[int, np.ndarray] = {
        pid: np.full((n_frames, 2), np.nan, dtype=np.float64) for pid in all_person_ids
    }
    gt_visible: dict[int, np.ndarray] = {
        pid: np.zeros((n_frames, len(rig.camera_ids)), dtype=bool) for pid in all_person_ids
    }

    for step_index, frame in enumerate(frames):
        anns = frame_annotations[frame]
        detections: dict[str, list[Detection]] = {cam: [] for cam in rig.camera_ids}
        for ann in anns:
            gt_world[ann.person_id][step_index] = ann.world_xy
            for cam_idx, camera_id in enumerate(rig.camera_ids):
                box = ann.bboxes.get(camera_id)
                if box is None:
                    continue
                gt_visible[ann.person_id][step_index, cam_idx] = True
                detections[camera_id].append(
                    Detection(bbox_xyxy=box, score=1.0, embedding=_DUMMY_EMBEDDING)
                )

        snapshots = pipeline.step(detections, step_index, dt)
        snapshots_per_frame.append(snapshots)
        predictions.append(
            np.array([s.world_xy for s in snapshots], dtype=np.float64)
            if snapshots
            else np.empty((0, 2), dtype=np.float64)
        )
        ground_truth.append(
            np.array([ann.world_xy for ann in anns], dtype=np.float64)
            if anns
            else np.empty((0, 2), dtype=np.float64)
        )

    detection_metrics = compute_moda_modp(predictions, ground_truth, threshold_m=match_radius_m)
    identity_report = evaluate_id_consistency(
        gt_world=gt_world,
        gt_visible=gt_visible,
        results=snapshots_per_frame,
        n_ids_issued=pipeline.n_ids_issued,
        match_radius_m=match_radius_m,
    )

    typer.echo(
        f"MODA={detection_metrics.moda:.3f}  MODP={detection_metrics.modp:.3f}  "
        f"precision={detection_metrics.precision:.3f}  recall={detection_metrics.recall:.3f}"
    )
    typer.echo(identity_report.summary())

    result = {
        "dataset": "WILDTRACK",
        "method": _MARKDOWN_ROW_LABEL,
        "n_frames": len(frames),
        "match_radius_m": match_radius_m,
        "detection": _json_safe(asdict(detection_metrics)),
        "identity": _json_safe(
            {
                "n_gt_agents": identity_report.n_gt_agents,
                "n_ids_issued": identity_report.n_ids_issued,
                "total_id_switches": identity_report.total_id_switches,
                "position_rmse_m": identity_report.position_rmse_m,
                "mean_position_error_m": identity_report.mean_position_error_m,
                "false_positive_tracks": identity_report.false_positive_tracks,
            }
        ),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    typer.echo(f"wrote {out_json}")

    _write_markdown_row(out_markdown, detection_metrics)
    typer.echo(f"wrote {out_markdown}")


if __name__ == "__main__":  # pragma: no cover
    app()
