"""`mcreid-calibrate` — build a rig calib.json from checkerboard + AprilTag captures.

Expected capture layout (see capture_guide.md):

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
import typer
import yaml

from mcreid.calib.homography import TagPlacement, ground_plane_from_apriltags
from mcreid.calib.intrinsics import CheckerboardSpec, calibrate_intrinsics_from_dir
from mcreid.calib.schema import CameraCalib, RigCalib
from mcreid.utils.logging import get_logger, setup_logging

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

    camera_dirs = sorted(d for d in capture_dir.iterdir() if d.is_dir())
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

        intrinsics_dir = camera_dir / "intrinsics"
        if not intrinsics_dir.is_dir():
            failures.append(f"{camera_id}: missing {intrinsics_dir}")
            continue
        intrinsics = calibrate_intrinsics_from_dir(intrinsics_dir, spec, min_views=min_views)
        typer.echo(
            f"  intrinsics: {intrinsics.n_views} views, RMS {intrinsics.rms_reproj_px:.3f} px"
        )
        if intrinsics.rms_reproj_px > max_rms_px:
            failures.append(
                f"{camera_id}: intrinsics RMS {intrinsics.rms_reproj_px:.3f} px "
                f"> {max_rms_px} px — re-shoot the checkerboard"
            )
            continue

        ground_image = next(
            (p for p in (camera_dir / "ground.jpg", camera_dir / "ground.png") if p.is_file()),
            None,
        )
        if ground_image is None:
            failures.append(f"{camera_id}: no ground.jpg/ground.png with the floor tags")
            continue
        raw = cv2.imread(str(ground_image), cv2.IMREAD_COLOR)
        if raw is None:
            failures.append(f"{camera_id}: could not read {ground_image}")
            continue
        image = np.asarray(raw, dtype=np.uint8)
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
