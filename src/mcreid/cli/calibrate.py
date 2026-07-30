"""`mcreid-calibrate` — build a rig calib.json from checkerboard + AprilTag captures.

Expected capture layout:

    footage/calib/
        cam0/intrinsics/*.jpg     checkerboard frames, >= 8 usable views
        cam0/ground.jpg           one frame with every floor AprilTag visible
        cam1/...
        tags.yaml                 the physical tag placements

`tags.yaml`::

    tag_size_m: 0.20
    tags:
      - {id: 0, x: 0.50, y: 0.50, yaw_deg: 0}
      - {id: 1, x: 4.50, y: 0.60, yaw_deg: 0}
      - {id: 2, x: 4.40, y: 3.80, yaw_deg: 0}
      - {id: 3, x: 0.60, y: 3.70, yaw_deg: 0}
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import typer
import yaml

from mcreid.calib.homography import TagPlacement, ground_plane_from_apriltags
from mcreid.calib.intrinsics import (
    CheckerboardSpec,
    calibrate_intrinsics_from_dir,
    calibrate_intrinsics_from_video,
    sharpness,
)
from mcreid.calib.report import (
    DEFAULT_MAX_FIT_RESIDUAL_M,
    DEFAULT_MAX_LOO_ERROR_M,
    CameraReport,
    RigReport,
    analyse_camera,
)
from mcreid.calib.schema import CameraCalib, RigCalib
from mcreid.utils.logging import get_logger, setup_logging
from mcreid.viz.calib_overlay import draw_floor_grid, draw_tag_reprojection

logger = get_logger(__name__)
app = typer.Typer(add_completion=False, help="Camera calibration for the multi-view rig.")


def load_tag_placements(path: Path) -> list[TagPlacement]:
    """Parse tags.yaml into `TagPlacement`s, failing loudly on a bad layout."""
    if not path.is_file():
        raise typer.BadParameter(f"tag placement file not found: {path}")
    data: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise typer.BadParameter(f"{path}: expected a YAML mapping")

    for key in ("tag_size_m", "tags"):
        if key not in data:
            raise typer.BadParameter(f"{path}: missing required key {key!r}")
    size = float(data["tag_size_m"])
    entries = data["tags"]
    if not isinstance(entries, list) or len(entries) < 2:
        raise typer.BadParameter(f"{path}: 'tags' must be a list of >= 2 entries")

    placements: list[TagPlacement] = []
    for i, entry in enumerate(entries):
        for key in ("id", "x", "y"):
            if key not in entry:
                raise typer.BadParameter(f"{path}: tags[{i}] missing {key!r}")
        placements.append(
            TagPlacement(
                tag_id=int(entry["id"]),
                center_xy_m=(float(entry["x"]), float(entry["y"])),
                size_m=float(entry.get("size_m", size)),
                yaw_deg=float(entry.get("yaw_deg", 0.0)),
            )
        )
    logger.info("loaded %d tag placement(s) from %s", len(placements), path)
    return placements


VIDEO_SUFFIXES = (".mp4", ".mov", ".mkv", ".avi")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def find_intrinsics_source(camera_dir: Path) -> Path:
    """Locate a camera's checkerboard capture — a video or a directory of stills.

    The capture guide tells Matus to shoot `cam0/intrinsics.mp4`, because phones
    record video far more easily than they export bursts of stills. Both layouts
    are accepted so neither the guide nor an ad-hoc capture can be "wrong".
    """
    for suffix in VIDEO_SUFFIXES:
        candidate = camera_dir / f"intrinsics{suffix}"
        if candidate.is_file():
            return candidate
    directory = camera_dir / "intrinsics"
    if directory.is_dir():
        return directory
    raise FileNotFoundError(
        f"{camera_dir.name}: expected {camera_dir / 'intrinsics.mp4'} or "
        f"{directory}/ with checkerboard frames"
    )


def find_ground_frame(capture_dir: Path, camera_id: str) -> Path:
    """Locate the frame showing the floor AprilTags for one camera."""
    candidates = [
        *(capture_dir / "tags" / f"{camera_id}{s}" for s in (*IMAGE_SUFFIXES, *VIDEO_SUFFIXES)),
        *(capture_dir / camera_id / f"ground{s}" for s in (*IMAGE_SUFFIXES, *VIDEO_SUFFIXES)),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"{camera_id}: no AprilTag frame found. Expected "
        f"{capture_dir / 'tags' / (camera_id + '.jpg')} or "
        f"{capture_dir / camera_id / 'ground.jpg'}"
    )


def read_frame(path: Path) -> npt.NDArray[np.uint8]:
    """Read a still, or the sharpest of the first second of a video."""
    if path.suffix.lower() in IMAGE_SUFFIXES:
        raw = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if raw is None:
            raise OSError(f"could not read image: {path}")
        return np.asarray(raw, dtype=np.uint8)

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise OSError(f"could not open video: {path}")
    best: npt.NDArray[np.uint8] | None = None
    best_score = -1.0
    try:
        for _ in range(30):
            ok, frame = capture.read()
            if not ok:
                break
            image = np.asarray(frame, dtype=np.uint8)
            score = sharpness(image)
            if score > best_score:
                best_score, best = score, image
    finally:
        capture.release()
    if best is None:
        raise OSError(f"{path}: no readable frames")
    logger.info("%s: picked sharpest of the first 30 frames (score %.1f)", path.name, best_score)
    return best


@app.command()
def rig(
    capture_dir: Path = typer.Option(..., help="Directory holding per-camera calibration capture."),
    tags: Path = typer.Option(None, help="tags.yaml. Defaults to <capture_dir>/tags.yaml."),
    out: Path = typer.Option(Path("calib/rig.json"), help="Output calib.json path."),
    board_cols: int = typer.Option(9, help="Checkerboard INTERIOR corners across."),
    board_rows: int = typer.Option(6, help="Checkerboard INTERIOR corners down."),
    square_size_m: float = typer.Option(..., help="Measured checkerboard square size, metres."),
    min_views: int = typer.Option(8, help="Minimum usable checkerboard views per camera."),
    min_tags: int = typer.Option(2, help="Minimum AprilTags that must be detected per camera."),
    max_rms_px: float = typer.Option(
        1.5, help="Reject a camera whose intrinsics RMS exceeds this."
    ),
    max_ground_rms_m: float = typer.Option(
        0.05, help="Reject a camera whose ground homography residual exceeds this."
    ),
    log_level: str = typer.Option("INFO"),
) -> None:
    """Calibrate every camera under ``capture_dir`` and write one rig calib.json."""
    setup_logging(log_level)
    if not capture_dir.is_dir():
        raise typer.BadParameter(f"capture directory not found: {capture_dir}")

    placements = load_tag_placements(tags or capture_dir / "tags.yaml")
    spec = CheckerboardSpec(
        inner_corners=(board_cols, board_rows), square_size_m=square_size_m
    )

    camera_dirs = sorted(
        d for d in capture_dir.iterdir() if d.is_dir() and d.name not in {"tags"}
    )
    if not camera_dirs:
        raise typer.BadParameter(f"no per-camera subdirectories in {capture_dir}")

    tag_world = np.vstack([p.world_corners() for p in placements])
    extent = (
        float(tag_world[:, 0].min()) - 1.0,
        float(tag_world[:, 1].min()) - 1.0,
        float(tag_world[:, 0].max()) + 1.0,
        float(tag_world[:, 1].max()) + 1.0,
    )

    cameras: list[CameraCalib] = []
    failures: list[str] = []

    for camera_dir in camera_dirs:
        camera_id = camera_dir.name
        typer.echo(f"--- {camera_id} ---")

        try:
            source = find_intrinsics_source(camera_dir)
        except FileNotFoundError as exc:
            failures.append(str(exc))
            continue
        intrinsics = (
            calibrate_intrinsics_from_dir(source, spec, min_views=min_views)
            if source.is_dir()
            else calibrate_intrinsics_from_video(source, spec, min_views=min_views)
        )
        typer.echo(
            f"  intrinsics: {intrinsics.n_views} views from {source.name}, "
            f"RMS {intrinsics.rms_reproj_px:.3f} px"
        )
        if intrinsics.rms_reproj_px > max_rms_px:
            failures.append(
                f"{camera_id}: intrinsics RMS {intrinsics.rms_reproj_px:.3f} px "
                f"> {max_rms_px} px — re-shoot the checkerboard"
            )
            continue

        try:
            ground_frame = find_ground_frame(capture_dir, camera_id)
            image = read_frame(ground_frame)
        except (FileNotFoundError, OSError) as exc:
            failures.append(str(exc))
            continue
        if (image.shape[1], image.shape[0]) != intrinsics.image_size:
            failures.append(
                f"{camera_id}: ground frame is {image.shape[1]}x{image.shape[0]} but "
                f"intrinsics are {intrinsics.image_size} — same camera, same resolution"
            )
            continue

        ground = ground_plane_from_apriltags(
            intrinsics, image, placements, min_tags=min_tags, margin_m=1.0
        )
        ground = ground.model_copy(update={"floor_extent_m": extent})
        typer.echo(
            f"  ground plane: {ground.n_correspondences} correspondences, "
            f"RMS {ground.rms_error_m * 100:.1f} cm"
        )
        if ground.rms_error_m > max_ground_rms_m:
            failures.append(
                f"{camera_id}: ground RMS {ground.rms_error_m * 100:.1f} cm "
                f"> {max_ground_rms_m * 100:.1f} cm — re-measure the tag positions"
            )
            continue

        cameras.append(CameraCalib(camera_id=camera_id, intrinsics=intrinsics, ground=ground))

    if failures:
        typer.echo("\nCalibration failed for:")
        for line in failures:
            typer.echo(f"  - {line}")
    if not cameras:
        typer.echo("\nNo camera calibrated successfully.")
        raise typer.Exit(code=1)

    rig_calib = RigCalib(cameras=cameras)
    path = rig_calib.save(out)
    typer.echo(f"\nwrote {path} ({len(cameras)}/{len(camera_dirs)} cameras)")
    if failures:
        raise typer.Exit(code=1)


@app.command()
def report(
    calib: Path = typer.Option(..., help="calib.json produced by `mcreid-calibrate rig`."),
    capture_dir: Path = typer.Option(..., help="Capture directory holding the tag frames."),
    tags: Path = typer.Option(None, help="tags.yaml. Defaults to <capture_dir>/tags.yaml."),
    out_dir: Path = typer.Option(Path("reports/calib"), help="Where to write the report."),
    grid_step_m: float = typer.Option(0.5, help="Floor grid spacing in the overlays."),
    max_loo_error_m: float = typer.Option(
        DEFAULT_MAX_LOO_ERROR_M, help="Leave-one-out error limit, metres."
    ),
    max_fit_residual_m: float = typer.Option(
        DEFAULT_MAX_FIT_RESIDUAL_M, help="Ground-plane fit residual limit, metres."
    ),
    log_level: str = typer.Option("INFO"),
) -> None:
    """Calibration sanity check. RUN THIS BEFORE ANY TRACKING ON REAL FOOTAGE.

    For every camera: reprojects the AprilTags, renders a metric floor grid into
    the image, and validates the ground homography by leave-one-out. Writes
    annotated PNGs plus a markdown/JSON summary, and exits non-zero with
    specific re-measure/re-shoot instructions if anything fails.
    """
    setup_logging(log_level)
    rig_calib = RigCalib.load(calib)
    placements = load_tag_placements(tags or capture_dir / "tags.yaml")
    out_dir.mkdir(parents=True, exist_ok=True)

    reports: list[CameraReport] = []
    for cam in rig_calib.cameras:
        try:
            frame_path = find_ground_frame(capture_dir, cam.camera_id)
            image = read_frame(frame_path)
        except (FileNotFoundError, OSError) as exc:
            typer.echo(f"  {cam.camera_id}: {exc}")
            reports.append(
                CameraReport(
                    camera_id=cam.camera_id,
                    n_tags_detected=0,
                    n_tags_expected=len(placements),
                    missing_tag_ids=[p.tag_id for p in placements],
                    fit_residual_m=float("nan"),
                    loo_error_mean_m=float("nan"),
                    loo_error_max_m=float("nan"),
                    reprojection_px_mean=float("nan"),
                    reprojection_px_max=float("nan"),
                    floor_coverage=float("nan"),
                    grid_is_convex=False,
                    problems=[str(exc)],
                )
            )
            continue

        camera_report, detections = analyse_camera(
            cam,
            image,
            placements,
            max_loo_error_m=max_loo_error_m,
            max_fit_residual_m=max_fit_residual_m,
        )
        reports.append(camera_report)

        cv2.imwrite(
            str(out_dir / f"{cam.camera_id}_grid.png"),
            draw_floor_grid(image, cam, step_m=grid_step_m),
        )
        cv2.imwrite(
            str(out_dir / f"{cam.camera_id}_tags.png"),
            draw_tag_reprojection(image, cam, detections, placements),
        )
        status = "OK" if camera_report.ok else "FAIL"
        typer.echo(
            f"  [{status}] {cam.camera_id}: {camera_report.n_tags_detected}"
            f"/{camera_report.n_tags_expected} tags, "
            f"LOO {camera_report.loo_error_mean_m * 100:.1f} cm, "
            f"fit {camera_report.fit_residual_m * 100:.1f} cm, "
            f"floor coverage {camera_report.floor_coverage:.0%}"
        )

    rig_report = RigReport(
        cameras=reports,
        max_loo_error_m=max_loo_error_m,
        max_fit_residual_m=max_fit_residual_m,
    )
    rig_report.to_json(out_dir / "summary.json")
    markdown = rig_report.to_markdown(out_dir / "summary.md")
    typer.echo(f"\nwrote overlays and {markdown}")

    if not rig_report.ok:
        typer.echo("\n" + "=" * 72)
        typer.echo("CALIBRATION CHECK FAILED — do not run tracking on this footage yet.")
        typer.echo("=" * 72)
        for cam_report in rig_report.failed:
            typer.echo(f"\n{cam_report.camera_id}:")
            for problem in cam_report.problems:
                typer.echo(f"  - {problem}")
        typer.echo(
            f"\nLook at {out_dir}/<cam>_grid.png: the grid must lie flat on the floor.\n"
            f"Look at {out_dir}/<cam>_tags.png: green and red outlines must coincide."
        )
        raise typer.Exit(code=1)

    typer.echo("\nCalibration check PASSED for every camera. Safe to run tracking.")


@app.command()
def check(
    calib: Path = typer.Option(..., help="calib.json to inspect."),
    log_level: str = typer.Option("INFO"),
) -> None:
    """Print a calibration summary and re-verify its round-trip accuracy."""
    from mcreid.calib.geometry import ground_to_image, image_to_ground

    setup_logging(log_level)
    rig_calib = RigCalib.load(calib)
    typer.echo(f"{calib}: {len(rig_calib.cameras)} cameras, created {rig_calib.created_at}")
    x0, y0, x1, y1 = rig_calib.floor_extent()
    typer.echo(f"floor extent: x [{x0:.2f}, {x1:.2f}] m, y [{y0:.2f}, {y1:.2f}] m")

    probe = np.stack(
        np.meshgrid(np.linspace(x0, x1, 6), np.linspace(y0, y1, 6)), axis=-1
    ).reshape(-1, 2)

    worst = 0.0
    for cam in rig_calib.cameras:
        pixels, valid = ground_to_image(cam, probe)
        if not valid.any():
            typer.echo(f"  {cam.camera_id}: NO floor point projects into frame — bad calibration")
            worst = float("inf")
            continue
        back, ok = image_to_ground(cam, pixels[valid])
        error = float(np.nanmax(np.linalg.norm(back - probe[valid], axis=1)))
        worst = max(worst, error)
        typer.echo(
            f"  {cam.camera_id}: intrinsics RMS {cam.intrinsics.rms_reproj_px:.3f} px, "
            f"ground RMS {cam.ground.rms_error_m * 100:.1f} cm, "
            f"{int(valid.sum())}/{len(probe)} probe points in frame, "
            f"round-trip max {error * 1000:.3f} mm"
        )
    if worst > 1e-6:
        typer.echo(f"\nWARNING: round-trip error {worst * 1000:.3f} mm is above 1 um")
        raise typer.Exit(code=1)
    typer.echo("\nround-trip OK")


if __name__ == "__main__":  # pragma: no cover
    app()
