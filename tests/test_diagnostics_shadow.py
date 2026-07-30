"""Tests for the shadow probe — the measurement instrument itself.

An instrument that quietly measures the wrong thing is worse than none, so
these check the two properties the conclusions will rest on: that it does not
perturb tracking, and that the d(t) curve it produces is not contaminated by
the resurrection it is supposed to be measuring across.
"""

from __future__ import annotations

import csv
import json

import numpy as np
import numpy.typing as npt
import pytest

from mcreid.diagnostics.shadow import QUERY_OBS, QUERY_TRACK_EMA, ShadowProbe, summarise
from mcreid.fusion.types import ViewObservation
from mcreid.live import LIVE_CAMERA_ID, LiveConfig, LiveSession, pixel_plane_calibration

WIDTH, HEIGHT = 640, 480
DIM = 16


def _unit(index: int) -> npt.NDArray[np.float64]:
    vec = np.zeros(DIM)
    vec[index] = 1.0
    return vec


class _ReturnBackend:
    """Present, absent, then back — clipped at the frame edge on re-entry.

    Reproduces the measured cycle-2 shape: the first frames of a return are
    half-body crops that read as a *different* appearance, which then resolve to
    the true one once the person is fully inside the frame.
    """

    def __init__(self, leave: int, back: int, clipped_frames: int) -> None:
        self.leave = leave
        self.back = back
        self.clipped_frames = clipped_frames
        self.whole = _unit(0)
        # A half-body crop reads as a different vector, not a noisy version of
        # the same one — that is the whole point of the truncation flag.
        self.half = _unit(1)

    def step(self, image, frame: int) -> list[ViewObservation]:
        if self.leave <= frame < self.back:
            return []
        clipped = self.back <= frame < self.back + self.clipped_frames
        box = (
            np.array([260.0, 150.0, 330.0, float(HEIGHT)])  # runs off the bottom
            if clipped
            else np.array([260.0, 150.0, 330.0, 400.0])
        )
        return [
            ViewObservation(
                camera_id=LIVE_CAMERA_ID,
                frame=frame,
                local_track_id=1,
                bbox_xyxy=box,
                embedding=self.half if clipped else self.whole,
                score=0.8,
            )
        ]


def _run(tmp_path, shadow: ShadowProbe | None, frames: int = 460) -> LiveSession:
    backend = _ReturnBackend(leave=40, back=440, clipped_frames=8)
    session = LiveSession(
        backend=backend,
        calibration=pixel_plane_calibration(WIDTH, HEIGHT),
        metric=False,
        config=LiveConfig(clip_seconds=0.2),
        shadow=shadow,
    )
    frame = np.full((HEIGHT, WIDTH, 3), 90, dtype=np.uint8)
    for i in range(frames):
        session.process(frame, now=i / 30, dt=1 / 30)
    return session


def _probe(tmp_path) -> ShadowProbe:
    return ShadowProbe(tmp_path / "shadow", gate=0.42, top_k=3)


# --- the property everything else depends on ---------------------------------------------


def test_the_shadow_probe_does_not_change_tracking(tmp_path) -> None:
    """Diagnostic-only means diagnostic-only: identical outcomes with it on."""
    without = _run(tmp_path, shadow=None)
    with_probe = _run(tmp_path, shadow=_probe(tmp_path))

    assert without.manager.n_ids_issued == with_probe.manager.n_ids_issued
    assert without.reported_ids == with_probe.reported_ids
    assert (
        without.manager.dormant.n_resurrected == with_probe.manager.dormant.n_resurrected
    )
    assert [t.global_id for t in without.manager.tracks] == [
        t.global_id for t in with_probe.manager.tracks
    ]


# --- does it actually capture the curve? --------------------------------------------------


def test_the_curve_survives_the_resurrection_it_is_measuring(tmp_path) -> None:
    """A frozen snapshot must keep producing rows after the entry is popped.

    Resurrection removes the entry from the live gallery. Probing the live
    gallery would therefore stop the curve at the exact frame the question is
    about — 'would waiting have helped?' cannot be answered from a curve that
    ends when the answer arrives.
    """
    probe = _probe(tmp_path)
    session = _run(tmp_path, shadow=probe)
    assert session.manager.dormant.n_resurrected >= 1, "test setup: a return must resurrect"

    return_rows = [r for r in probe.rows if r.frame >= 440]
    assert return_rows, "the return must be recorded at all"
    after = [r for r in return_rows if r.frame > min(r.frame for r in return_rows) + 8]
    assert after, "the curve must continue past the resurrection frame"


def test_truncated_frames_are_recorded_as_such_and_are_further_away(tmp_path) -> None:
    """The split the whole run exists to produce."""
    probe = _probe(tmp_path)
    _run(tmp_path, shadow=probe)

    obs_rows = [r for r in probe.rows if r.source == QUERY_OBS and r.frame >= 440]
    clipped = [r.distance for r in obs_rows if r.truncated]
    clean = [r.distance for r in obs_rows if not r.truncated]

    assert clipped and clean, f"need both kinds, got {len(clipped)} / {len(clean)}"
    assert np.mean(clipped) > np.mean(clean), (
        "a half-body crop must measure further from the stored identity"
    )
    assert not any(r.would_accept for r in obs_rows if r.truncated)
    assert any(r.would_accept for r in obs_rows if not r.truncated)


def test_frames_into_episode_restarts_on_the_return(tmp_path) -> None:
    """The x axis must be 'frames since they came back', not 'frames so far'."""
    probe = _probe(tmp_path)
    _run(tmp_path, shadow=probe)

    return_rows = sorted(
        (r for r in probe.rows if r.frame >= 440), key=lambda r: r.frame
    )
    assert return_rows[0].frames_into_episode == 0
    assert return_rows[0].episode >= 2, "the return is a new presence episode"


def test_seeding_a_gallery_must_not_move_the_ema() -> None:
    """Load-bearing invariant for this whole measurement.

    Adoption seeds the stored vectors into the resurrected track's own gallery.
    If those ever reached the EMA, the shadow probe's `track_ema` distance would
    fall toward zero for bookkeeping reasons and would look exactly like
    "waiting improved the query" — the result this run exists to test for. The
    shadow probe uses the raw EMA precisely because seeding does not touch it,
    so that assumption is pinned here rather than left implicit.
    """
    from mcreid.fusion.associate import AppearanceGallery

    gallery = AppearanceGallery()
    gallery.add(LIVE_CAMERA_ID, _unit(0))
    before = gallery.ema
    gallery.seed("_dormant", np.stack([_unit(1), _unit(2)]))

    assert before is not None and gallery.ema is not None
    np.testing.assert_allclose(gallery.ema, before, atol=1e-12)
    assert any(cam == "_dormant" for cam, _v in gallery.items()), (
        "test is vacuous unless the seeds were actually stored"
    )


def test_the_track_ema_query_is_recorded_at_all(tmp_path) -> None:
    probe = _probe(tmp_path)
    _run(tmp_path, shadow=probe)
    assert [r for r in probe.rows if r.source == QUERY_TRACK_EMA], (
        "both probe paths must appear in the record"
    )


# --- output shape -------------------------------------------------------------------------


def test_writes_jsonl_and_long_format_csv(tmp_path) -> None:
    probe = _probe(tmp_path)
    _run(tmp_path, shadow=probe)
    jsonl, csv_path = probe.write()

    records = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines()]
    assert len(records) == len(probe.rows)

    with csv_path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == len(probe.rows)
    # Everything a d(t) plot split by truncation needs, in one flat table.
    for column in ("frames_into_episode", "distance", "truncated", "hits", "sigma_m",
                   "entry_key", "source", "episode"):
        assert column in rows[0], f"the CSV must carry {column!r} for plotting"


def test_summary_reports_the_split_and_the_curve(tmp_path) -> None:
    probe = _probe(tmp_path)
    _run(tmp_path, shadow=probe)
    text = "\n".join(summarise(probe.rows, probe.gate))

    assert "truncated" in text and "clean" in text
    assert "d(t) per episode" in text
    assert "first frame under the gate" in text


def test_summary_is_honest_about_an_empty_run() -> None:
    assert "no rows" in summarise([], 0.42)[0]


@pytest.mark.parametrize("frames", [5, 30])
def test_no_rows_before_anything_goes_dormant(tmp_path, frames) -> None:
    """Nothing to probe against is not the same as a distance of zero."""
    probe = _probe(tmp_path)
    _run(tmp_path, shadow=probe, frames=frames)
    assert probe.rows == []
