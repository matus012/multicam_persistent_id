"""`mcreid-live` — live single-camera persistent-ID tracking from a webcam.

Runs the full per-view stack in real time: YOLO detection, OSNet appearance,
per-view tracking, occlusion coasting, and dormant-gallery re-identification for
someone who leaves the frame and comes back.

No calibration is required. With one camera there is no cross-view fusion to do,
so identity does not depend on knowing the floor plane — pass `--homography` only
if you want the metric BEV panel.

Hotkeys:  q  quit        s  save the last few seconds to reports/
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import typer

from mcreid.diagnostics.shadow import ShadowProbe, summarise
from mcreid.fusion.dormant import DormantConfig
from mcreid.fusion.global_id import FusionConfig
from mcreid.live import (
    LiveConfig,
    LiveSession,
    load_homography_calibration,
    pixel_plane_calibration,
)
from mcreid.track.gpu_view import GpuPerViewBackend, GpuViewConfig
from mcreid.track.reid_models import DEFAULT_EMBEDDER
from mcreid.utils.logging import get_logger, setup_logging
from mcreid.utils.seed import DEFAULT_SEED, seed_everything

logger = get_logger(__name__)
app = typer.Typer(add_completion=False, help="Live single-camera tracking.")

WINDOW = "mcreid live"

# Every mechanism --single-occupant switches on. Named here so the flag's full
# effect is one list, and `test_single_occupant_enables_every_listed_mechanism`
# can assert the resolver actually applies all of it.
SINGLE_OCCUPANT_NEAR_MISS_MARGIN = 0.10
SINGLE_OCCUPANT_RETRY_OFFSETS = (4, 9)


def resolve_fusion_config(
    dormant_gate: float | None, single_occupant: bool
) -> FusionConfig | None:
    """Turn the CLI flags into a `FusionConfig`, or None to keep the defaults.

    Extracted from the command body so it can be tested without a webcam. That
    is not a stylistic preference: shadow session s2 was run with
    `--single-occupant` and the retry never fired, because the inline version of
    this built a `DormantConfig` that simply never mentioned `retry_offsets`.
    The field existed, was documented and was covered by unit tests; nothing
    asserted that the FLAG REACHED IT. A config resolver that cannot be called
    from a test cannot be checked, so this one can.
    """
    if dormant_gate is None and not single_occupant:
        return None
    gate = dormant_gate if dormant_gate is not None else DormantConfig().appearance_distance
    return FusionConfig(
        dormant=DormantConfig(
            appearance_distance=gate,
            near_miss_margin=SINGLE_OCCUPANT_NEAR_MISS_MARGIN if single_occupant else 0.0,
            retry_offsets=SINGLE_OCCUPANT_RETRY_OFFSETS if single_occupant else (),
        )
    )


@app.command()
def run(
    device: int = typer.Option(0, help="Webcam device index."),
    width: int = typer.Option(1280, help="Requested capture width."),
    height: int = typer.Option(720, help="Requested capture height."),
    weights: Path = typer.Option(
        Path("weights/yolo11s.pt"),
        help=(
            "Detector weights. yolo11s is the real-time default at 720p; "
            "yolo11m/x are more accurate and slower."
        ),
    ),
    imgsz: int = typer.Option(960, help="Detector input size (multiple of 32)."),
    conf: float = typer.Option(0.35, help="Detection confidence floor."),
    embedder: str = typer.Option(DEFAULT_EMBEDDER, help="Appearance model."),
    homography: Path = typer.Option(
        None,
        help=(
            "Optional 4-point YAML (image_points / world_points) enabling the "
            "metric BEV panel. Without it distances are scaled pixels and the "
            "BEV is omitted."
        ),
    ),
    span_m: float = typer.Option(
        6.0, help="Assumed floor span of the frame height when uncalibrated."
    ),
    dormant_gate: float = typer.Option(
        None,
        help=(
            "Override the dormant (long-gap) appearance gate. Leave unset to use "
            "the shipped 0.42. Every probe's true distance is logged either way, "
            "so measure first and only then loosen. Must stay <= "
            "revive_appearance_distance (0.48)."
        ),
    ),
    single_occupant: bool = typer.Option(
        False,
        "--single-occupant",
        help=(
            "Enable duplicate suppression in the dormant gallery. Valid ONLY when "
            "you are the only person who will be in frame: it stops one missed "
            "re-identification from deadlocking every later return, but with "
            "strangers on file it was measured to cost more identity theft than "
            "the recall it buys. Use it for the sit/leave/return acceptance run."
        ),
    ),
    shadow_probe: Path = typer.Option(
        None,
        help=(
            "DIAGNOSTIC. Write a per-frame dormant-distance record to this path "
            "(.jsonl + .csv). Measures the appearance distance to every retired "
            "identity on every frame without acting on it, so one return yields a "
            "full d(t) curve instead of a single probe. Tracking behaviour is "
            "unchanged: resurrection still happens normally and the recorder only "
            "reads. Omit for normal runs."
        ),
    ),
    clip_seconds: float = typer.Option(8.0, help="Length of the save-clip buffer."),
    out_dir: Path = typer.Option(Path("reports"), help="Where 's' saves clips."),
    max_frames: int = typer.Option(0, help="Stop after N frames (0 = until 'q')."),
    seed: int = typer.Option(DEFAULT_SEED),
    log_level: str = typer.Option("INFO"),
) -> None:
    """Open the webcam and track continuously until 'q'."""
    setup_logging(log_level)
    seed_everything(seed)

    capture = cv2.VideoCapture(device)
    if not capture.isOpened():
        raise typer.BadParameter(
            f"could not open webcam device {device}. Try a different --device index, "
            "or check that no other application is holding the camera."
        )
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    ok, probe = capture.read()
    if not ok or probe is None:
        capture.release()
        raise typer.BadParameter(f"webcam {device} opened but returned no frames")
    actual_h, actual_w = probe.shape[:2]
    typer.echo(f"capture: {actual_w}x{actual_h} on device {device}")

    metric = homography is not None
    calibration = (
        load_homography_calibration(homography, actual_w, actual_h)
        if metric
        else pixel_plane_calibration(actual_w, actual_h, span_m)
    )
    if not metric:
        typer.echo(
            "no --homography: running uncalibrated. Identity, coasting and long-gap "
            "re-ID all work; distances are scaled pixels and the BEV is omitted."
        )

    backend = GpuPerViewBackend(
        "live",
        GpuViewConfig(
            weights=weights, imgsz=imgsz, conf_threshold=conf, embedder=embedder
        ),
    )
    fusion_config = resolve_fusion_config(dormant_gate, single_occupant)
    if fusion_config is not None:
        if dormant_gate is not None:
            typer.echo(f"dormant appearance gate overridden: {dormant_gate:.2f}")
        if single_occupant:
            typer.echo(
                "single-occupant mode: a near-missed return will not be stored as a "
                "rival record. Only valid if nobody else appears in frame."
            )

    dormant_cfg = (fusion_config or FusionConfig()).dormant
    shadow = (
        ShadowProbe(shadow_probe, gate=dormant_cfg.appearance_distance, top_k=dormant_cfg.top_k)
        if shadow_probe is not None
        else None
    )
    if shadow is not None:
        typer.echo(f"shadow probe ON (diagnostic): recording to {shadow_probe}.jsonl/.csv")

    session = LiveSession(
        backend=backend,
        calibration=calibration,
        metric=metric,
        config=LiveConfig(span_m=span_m, clip_seconds=clip_seconds),
        fusion_config=fusion_config,
        shadow=shadow,
    )

    typer.echo("running — 'q' to quit, 's' to save a clip")
    previous = time.perf_counter()
    # Provisional sizing; re-done from the measured loop rate below, because the
    # real rate is capture + detector + display and is not knowable up front.
    session.set_clip_capacity(30.0)
    processed = 0
    resized_clip = False

    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                logger.warning("capture returned no frame; stopping")
                break

            now = time.perf_counter()
            dt = max(now - previous, 1e-3)
            previous = now

            annotated, info = session.process(
                np.asarray(frame, dtype=np.uint8), now, dt
            )
            cv2.imshow(WINDOW, annotated)
            processed += 1

            if not resized_clip and processed == 60:
                # The loop rate has settled: size the buffer to hold the
                # requested number of *seconds*, not of 30-fps frames.
                session.set_clip_capacity(session.wall_fps)
                resized_clip = True
                logger.info(
                    "loop settled at %.1f FPS; clip buffer holds %d frames (%.1f s)",
                    session.wall_fps,
                    session.clip.maxlen or 0,
                    clip_seconds,
                )

            if processed % 60 == 0:
                logger.info(
                    "%.1f FPS (%.1f processing) | tracks %d | ids %d | coasting %d "
                    "| dormant %d | resurrected %d",
                    info["wall_fps"],
                    info["fps"],
                    info["tracks"],
                    info["reported_ids"],
                    info["coasting"],
                    info["dormant"],
                    info["resurrected"],
                )

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                path = session.save_clip(out_dir, session.wall_fps or 15.0)
                typer.echo(f"saved {path}" if path else "nothing buffered yet")
            if max_frames and processed >= max_frames:
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()

    # Report the identities that were actually shown, not the mint counter.
    # n_ids_issued includes tentative births that never survive n_init frames —
    # on a live webcam a single flicker of a false detection bumps it, so it
    # reads as ID churn when nothing churned.
    reported = session.reported_ids
    held = [
        (session.timeline.held_seconds(gid, session.last_now), gid) for gid in reported
    ]
    typer.echo(
        f"processed {processed} frames at {session.wall_fps:.1f} FPS end-to-end "
        f"({session.fps:.1f} FPS tracking throughput)"
    )
    typer.echo(
        f"{len(reported)} identities confirmed and shown "
        f"({session.manager.n_ids_issued} tracks minted incl. tentative), "
        f"{session.manager.dormant.n_resurrected} resurrected from the gallery"
    )
    if held:
        longest, gid = max(held)
        typer.echo(f"longest-held identity: ID {gid} for {longest:.1f} s")
    if session.timeline.reacquired_gap:
        gid, gap = max(session.timeline.reacquired_gap.items(), key=lambda kv: kv[1])
        typer.echo(f"longest gap survived: ID {gid} reacquired after {gap:.1f} s")

    # The long-gap path is the one that fails silently: a rejected probe looks
    # exactly like nobody having returned. Print what the gate actually saw.
    for line in session.manager.dormant.probe_report():
        typer.echo(line)
    if session.shadow is not None:
        jsonl, csv_path = session.shadow.write()
        for line in summarise(session.shadow.rows, session.shadow.gate):
            typer.echo(line)
        typer.echo(f"shadow record: {jsonl}  and  {csv_path}")

    if session.manager.dormant.n_suppressed_duplicates:
        typer.echo(
            f"{session.manager.dormant.n_suppressed_duplicates} identity/identities "
            f"not stored as duplicates of someone already in the gallery"
        )


if __name__ == "__main__":  # pragma: no cover
    app()
