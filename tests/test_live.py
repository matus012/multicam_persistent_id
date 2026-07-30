"""Tests for the live single-camera session.

No camera, no GPU, no torch: the detector backend and the clock are injected,
so the whole frame loop — identity, coasting, re-acquisition, clip buffering —
is exercised on CPU with synthetic frames.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from mcreid.fusion.types import ViewObservation
from mcreid.live import (
    LIVE_CAMERA_ID,
    IdentityTimeline,
    LiveConfig,
    LiveSession,
    load_homography_calibration,
    pixel_plane_calibration,
)

WIDTH, HEIGHT = 640, 480
DIM = 16


def _frame() -> npt.NDArray[np.uint8]:
    return np.full((HEIGHT, WIDTH, 3), 90, dtype=np.uint8)


def _unit(seed: int) -> npt.NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=DIM)
    return vec / np.linalg.norm(vec)


class _ScriptedBackend:
    """Emits a fixed person, optionally disappearing for a scripted window."""

    def __init__(
        self, blackout: tuple[int, int] | None = None, identity_seed: int = 3
    ) -> None:
        self.blackout = blackout
        self.embedding = _unit(identity_seed)
        self.calls = 0

    def step(self, image, frame: int) -> list[ViewObservation]:
        self.calls += 1
        if self.blackout and self.blackout[0] <= frame < self.blackout[1]:
            return []
        # Slow horizontal drift so the motion model has something to track.
        x = 260 + (frame % 40)
        return [
            ViewObservation(
                camera_id=LIVE_CAMERA_ID,
                frame=frame,
                local_track_id=1,
                bbox_xyxy=np.array([x, 150.0, x + 70.0, 400.0]),
                embedding=self.embedding,
                score=0.9,
            )
        ]


def _session(backend, metric: bool = False) -> LiveSession:
    return LiveSession(
        backend=backend,
        calibration=pixel_plane_calibration(WIDTH, HEIGHT),
        metric=metric,
        config=LiveConfig(clip_seconds=1.0),
    )


# --- pixel-plane stand-in calibration -------------------------------------------------------


def test_pixel_plane_calibration_is_usable_and_scaled() -> None:
    calib = pixel_plane_calibration(WIDTH, HEIGHT, span_m=6.0)
    assert calib.camera_id == LIVE_CAMERA_ID
    extent = calib.ground.floor_extent_m
    assert extent is not None
    assert extent[3] == pytest.approx(6.0), "frame height should span span_m"
    assert "not metres" in calib.notes, "the unit convention must be recorded"


@pytest.mark.parametrize(("w", "h", "span"), [(0, 480, 6.0), (640, 0, 6.0), (640, 480, 0.0)])
def test_pixel_plane_calibration_rejects_bad_input(w, h, span) -> None:
    with pytest.raises(ValueError):
        pixel_plane_calibration(w, h, span)


def test_homography_loader_rejects_a_bad_file(tmp_path) -> None:
    path = tmp_path / "h.yaml"
    path.write_text("image_points: [[0,0],[1,0]]\nworld_points: [[0,0],[1,0]]\n")
    with pytest.raises(ValueError, match=">= 4"):
        load_homography_calibration(path, WIDTH, HEIGHT)


def test_homography_loader_requires_the_keys(tmp_path) -> None:
    path = tmp_path / "h.yaml"
    path.write_text("image_points: [[0,0],[1,0],[1,1],[0,1]]\n")
    with pytest.raises(ValueError, match="world_points"):
        load_homography_calibration(path, WIDTH, HEIGHT)


# --- identity timeline ----------------------------------------------------------------------


def test_timeline_reports_held_duration() -> None:
    timeline = IdentityTimeline()
    timeline.observe(7, now=100.0)
    timeline.observe(7, now=100.5)
    assert timeline.held_seconds(7, 104.0) == pytest.approx(4.0)
    assert 7 not in timeline.reacquired_gap


def test_timeline_records_a_reacquisition_gap() -> None:
    """The banner's headline number: same ID returning after an absence."""
    timeline = IdentityTimeline()
    timeline.observe(3, now=10.0)
    timeline.observe(3, now=25.0, gap_threshold=1.0)
    assert timeline.reacquired_gap[3] == pytest.approx(15.0)
    assert timeline.held_seconds(3, 25.0) == pytest.approx(15.0), (
        "held time runs from first sighting, not from the re-acquisition"
    )


# --- the frame loop -------------------------------------------------------------------------


def test_process_returns_an_annotated_frame_and_state() -> None:
    session = _session(_ScriptedBackend())
    annotated, info = session.process(_frame(), now=0.0, dt=1 / 30)

    assert annotated.shape[0] > HEIGHT, "the banner strip should be appended"
    assert annotated.shape[1] == WIDTH, "uncalibrated runs must not attach a BEV"
    assert info["frame"] == 0
    assert info["observations"] == 1


def test_rejects_non_positive_dt() -> None:
    session = _session(_ScriptedBackend())
    with pytest.raises(ValueError, match="dt must be positive"):
        session.process(_frame(), now=0.0, dt=0.0)


def test_identity_is_stable_across_frames() -> None:
    session = _session(_ScriptedBackend())
    seen = set()
    for i in range(30):
        _, _info = session.process(_frame(), now=i / 30, dt=1 / 30)
        seen.update(s.global_id for s in session.manager.tracks if s.is_visible)
    assert len(seen) == 1, f"one person should hold one global id, got {seen}"


def test_track_coasts_through_a_short_disappearance() -> None:
    """Detection drops out; the track must persist rather than vanish."""
    session = _session(_ScriptedBackend(blackout=(20, 30)))
    coasting_seen = False
    for i in range(40):
        _, info = session.process(_frame(), now=i / 30, dt=1 / 30)
        if 20 <= i < 30 and info["coasting"] > 0:
            coasting_seen = True
    assert coasting_seen, "the track should coast while detection is absent"
    assert session.manager.n_ids_issued <= 2, (
        f"a 10-frame dropout must not churn identities; minted "
        f"{session.manager.n_ids_issued}"
    )


def test_wall_fps_is_the_callers_rate_not_the_processing_rate() -> None:
    """The two rates are different numbers and must not be conflated.

    A saved clip is written at one frame per loop iteration, so quoting the
    processing throughput as the session rate (and as the clip's frame rate)
    plays the video back time-compressed.
    """
    session = _session(_ScriptedBackend())
    for i in range(20):
        session.process(_frame(), now=i / 15, dt=1 / 15)

    assert session.wall_fps == pytest.approx(15.0, rel=1e-6)
    assert session.fps > session.wall_fps, (
        "synthetic frames process far faster than the 15 FPS clock they claim; "
        "if these two are equal the wall rate is not being measured"
    )


def test_reported_ids_excludes_tentative_births() -> None:
    """One flicker of a false detection must not read as a second identity."""

    class _FlickeringBackend(_ScriptedBackend):
        def step(self, image, frame: int) -> list[ViewObservation]:
            observations = super().step(image, frame)
            if frame == 10:  # a single-frame ghost, far from the person
                observations.append(
                    ViewObservation(
                        camera_id=LIVE_CAMERA_ID,
                        frame=frame,
                        local_track_id=2,
                        bbox_xyxy=np.array([20.0, 100.0, 80.0, 160.0]),
                        embedding=_unit(99),
                        score=0.6,
                    )
                )
            return observations

    session = _session(_FlickeringBackend())
    for i in range(30):
        session.process(_frame(), now=i / 30, dt=1 / 30)

    assert session.manager.n_ids_issued >= 2, "the ghost should mint a tentative track"
    assert session.reported_ids == [1], (
        f"only the person was ever confirmed, got {session.reported_ids}"
    )


def test_clip_buffer_is_bounded_and_saves(tmp_path) -> None:
    session = _session(_ScriptedBackend())
    session.set_clip_capacity(10.0)  # 1.0 s * 10 fps
    for i in range(40):
        session.process(_frame(), now=i / 10, dt=0.1)

    assert len(session.clip) == 10, "the rolling buffer must stay bounded"
    path = session.save_clip(tmp_path, fps=10.0)
    assert path is not None and path.is_file() and path.stat().st_size > 0


def test_save_clip_with_nothing_buffered_returns_none(tmp_path) -> None:
    session = _session(_ScriptedBackend())
    assert session.save_clip(tmp_path, fps=10.0) is None


def test_backend_is_called_once_per_frame() -> None:
    backend = _ScriptedBackend()
    session = _session(backend)
    for i in range(5):
        session.process(_frame(), now=i / 30, dt=1 / 30)
    assert backend.calls == 5


# --- CLI wiring: the flag must actually reach the config ---------------------
#
# Shadow session s2 was run with --single-occupant and the retry never fired.
# The config field existed, was documented, and was covered by unit tests; what
# nothing asserted was that the FLAG REACHED IT. The command body built a
# DormantConfig that simply never mentioned retry_offsets, so it stayed at its
# OFF default. These tests close that class of bug, not just the instance.


def test_single_occupant_enables_every_listed_mechanism() -> None:
    """THE s2 regression: --single-occupant must switch on the whole list."""
    from mcreid.cli.live import (
        SINGLE_OCCUPANT_NEAR_MISS_MARGIN,
        SINGLE_OCCUPANT_RETRY_OFFSETS,
        resolve_fusion_config,
    )

    cfg = resolve_fusion_config(dormant_gate=None, single_occupant=True)
    assert cfg is not None, "--single-occupant must produce a config, not fall through"
    assert cfg.dormant.retry_offsets == SINGLE_OCCUPANT_RETRY_OFFSETS == (4, 9), (
        "the retry never fired in s2 because this was left at its OFF default"
    )
    assert cfg.dormant.near_miss_margin == SINGLE_OCCUPANT_NEAR_MISS_MARGIN == 0.10


def test_default_run_leaves_both_single_occupant_mechanisms_off() -> None:
    """Neither mechanism may leak into a plain multi-person run.

    The retry is exploitable with more than one person in frame (adversarial
    review demonstrated a stranger taking a dormant identity), so "off unless
    the flag is given" is a safety property, not a preference.
    """
    from mcreid.cli.live import resolve_fusion_config
    from mcreid.fusion.global_id import FusionConfig

    assert resolve_fusion_config(dormant_gate=None, single_occupant=False) is None
    shipped = FusionConfig().dormant
    assert shipped.retry_offsets == ()
    assert shipped.near_miss_margin == 0.0


def test_a_gate_override_alone_does_not_enable_single_occupant_mechanisms() -> None:
    """--dormant-gate is a sweep knob and must not smuggle the retry in with it."""
    from mcreid.cli.live import resolve_fusion_config

    cfg = resolve_fusion_config(dormant_gate=0.40, single_occupant=False)
    assert cfg is not None
    assert cfg.dormant.appearance_distance == pytest.approx(0.40)
    assert cfg.dormant.retry_offsets == ()
    assert cfg.dormant.near_miss_margin == 0.0


def test_geometry_only_ablation_disables_every_appearance_path() -> None:
    """Same bug class, swept from the other CLI: --geometry-only built its config
    inline in the command body with no test, and it publishes README numbers.

    "Geometry only" has to mean ALL FOUR appearance paths are off. Missing one
    would leave appearance quietly voting in an ablation labelled as having none,
    which is the kind of error that shows up as an unexplained table row.
    """
    from mcreid.cli.wildtrack_run import resolve_fusion_config as resolve_wildtrack

    cfg = resolve_wildtrack(geometry_only=True)
    assert cfg.association.weight_appearance == 0.0, "association must not vote on appearance"
    assert cfg.association.weight_geometry == 1.0
    assert cfg.association.max_appearance_distance == 2.0, "gate open = no appearance veto"
    assert cfg.merge_appearance_distance == 2.0, "duplicate merge must not veto on appearance"
    assert cfg.revive_appearance_distance == 2.0, "revival must not veto on appearance"
    assert cfg.dormant.enabled is False, "the dormant path is appearance-only; it must be off"


def test_geometry_only_false_is_exactly_the_shipped_default() -> None:
    """The ablation must not perturb the baseline it is compared against."""
    from mcreid.cli.wildtrack_run import resolve_fusion_config as resolve_wildtrack
    from mcreid.fusion.global_id import FusionConfig

    assert resolve_wildtrack(geometry_only=False) == FusionConfig()
