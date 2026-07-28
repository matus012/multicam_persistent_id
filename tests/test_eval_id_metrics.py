"""Tests for mcreid.eval.id_metrics — the ID-consistency scoring used by the
cardboard (G-M1-1) gate.

All ground truth and snapshots are hand-built here; no toy scene or tracker is
involved, so every number is independently predictable.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from mcreid.eval.id_metrics import evaluate_id_consistency
from mcreid.fusion.types import GlobalTrackSnapshot, TrackState

FloatArray = npt.NDArray[np.float64]

N_FRAMES = 10


def _snapshot(global_id: int, xy: tuple[float, float], frame: int) -> GlobalTrackSnapshot:
    return GlobalTrackSnapshot(
        global_id=global_id,
        frame=frame,
        world_xy=np.array(xy, dtype=np.float64),
        velocity_mps=np.zeros(2),
        covariance=np.eye(2) * 0.01,
        state=TrackState.CONFIRMED,
        supporting_cameras=("cam0",),
        frames_since_measurement=0,
        hits=5,
    )


def _gt_always_present(
    n_frames: int = N_FRAMES, xy: tuple[float, float] = (0.0, 0.0)
) -> dict[int, FloatArray]:
    return {1: np.tile(np.array(xy, dtype=np.float64), (n_frames, 1))}


def _gt_visible_with_blackout(
    blackout: tuple[int, int], n_frames: int = N_FRAMES, n_cameras: int = 1
) -> dict[int, npt.NDArray[np.bool_]]:
    visible = np.ones((n_frames, n_cameras), dtype=bool)
    lo, hi = blackout
    visible[lo:hi] = False
    return {1: visible}


# --- switch counting: stable ids -----------------------------------------------------


def test_no_switch_when_id_is_stable() -> None:
    gt_world = _gt_always_present()
    gt_visible = _gt_visible_with_blackout((N_FRAMES, N_FRAMES))  # no blackout
    results = [[_snapshot(1, (0.0, 0.0), f)] for f in range(N_FRAMES)]

    report = evaluate_id_consistency(gt_world, gt_visible, results, n_ids_issued=1)
    assert report.id_switches == {1: 0}
    assert report.total_id_switches == 0
    assert report.coverage[1] == pytest.approx(1.0)
    assert report.coverage_visible[1] == pytest.approx(1.0)


def test_exactly_one_switch_when_id_changes_once() -> None:
    gt_world = _gt_always_present()
    gt_visible = _gt_visible_with_blackout((N_FRAMES, N_FRAMES))
    results = [[_snapshot(1 if f < 5 else 2, (0.0, 0.0), f)] for f in range(N_FRAMES)]

    report = evaluate_id_consistency(gt_world, gt_visible, results, n_ids_issued=2)
    assert report.id_switches == {1: 1}
    assert report.total_id_switches == 1
    assert report.ids_per_agent[1] == [1, 2]


# --- unmatched frames do not themselves count as switches ------------------------------


def test_unmatched_gap_then_same_id_resumes_causes_no_switch() -> None:
    gt_world = _gt_always_present()
    gt_visible = _gt_visible_with_blackout((3, 6))  # blackout frames 3,4,5
    results = []
    for f in range(N_FRAMES):
        results.append([] if f in (3, 4, 5) else [_snapshot(1, (0.0, 0.0), f)])

    report = evaluate_id_consistency(gt_world, gt_visible, results, n_ids_issued=1)
    assert report.id_switches == {1: 0}, "an unmatched gap must not itself be a switch"
    # 7 of 10 frames matched -> coverage; all 7 visible frames matched -> coverage_visible.
    assert report.coverage[1] == pytest.approx(0.7)
    assert report.coverage_visible[1] == pytest.approx(1.0)
    assert report.longest_coast_survived[1] == 3


def test_unmatched_gap_then_different_id_resumes_counts_exactly_one_switch() -> None:
    gt_world = _gt_always_present()
    gt_visible = _gt_visible_with_blackout((3, 6))
    results = []
    for f in range(N_FRAMES):
        if f in (3, 4, 5):
            results.append([])
        elif f < 3:
            results.append([_snapshot(1, (0.0, 0.0), f)])
        else:
            results.append([_snapshot(2, (0.0, 0.0), f)])

    report = evaluate_id_consistency(gt_world, gt_visible, results, n_ids_issued=2)
    assert report.id_switches == {
        1: 1
    }, "the switch must be counted exactly once, at the resumed frame, not once per gap frame"
    assert report.total_id_switches == 1


# --- coverage math -------------------------------------------------------------------------


def test_coverage_math_all_frames_vs_visible_only() -> None:
    gt_world = _gt_always_present()
    gt_visible = _gt_visible_with_blackout((3, 6))
    # Matched on every frame including the blackout (a coasted match).
    results = [[_snapshot(1, (0.0, 0.0), f)] for f in range(N_FRAMES)]

    report = evaluate_id_consistency(gt_world, gt_visible, results, n_ids_issued=1)
    assert report.coverage[1] == pytest.approx(1.0), "coasted matches during blackout count"
    assert report.coverage_visible[1] == pytest.approx(1.0), "7/7 visible frames matched"


def test_coverage_nan_when_agent_never_present() -> None:
    gt_world = {1: np.full((N_FRAMES, 2), np.nan)}
    gt_visible = {1: np.zeros((N_FRAMES, 1), dtype=bool)}
    results = [[] for _ in range(N_FRAMES)]

    report = evaluate_id_consistency(gt_world, gt_visible, results, n_ids_issued=0)
    assert np.isnan(report.coverage[1])
    assert np.isnan(report.coverage_visible[1])


# --- longest_coast_survived: a survived blackout is credited ------------------------------


def test_survived_blackout_credited_to_longest_coast_survived() -> None:
    gt_world = _gt_always_present()
    gt_visible = _gt_visible_with_blackout((3, 6))  # 3-frame blackout
    # Coasting keeps reporting the SAME id all the way through the blackout.
    results = [[_snapshot(1, (0.0, 0.0), f)] for f in range(N_FRAMES)]

    report = evaluate_id_consistency(gt_world, gt_visible, results, n_ids_issued=1)
    assert report.longest_coast_survived[1] == 3


def test_blackout_not_credited_if_id_changed_across_it() -> None:
    gt_world = _gt_always_present()
    gt_visible = _gt_visible_with_blackout((3, 6))
    results = []
    for f in range(N_FRAMES):
        gid = 1 if f < 3 else 2  # id changes across the blackout window
        results.append([_snapshot(gid, (0.0, 0.0), f)])

    report = evaluate_id_consistency(gt_world, gt_visible, results, n_ids_issued=2)
    assert (
        report.longest_coast_survived[1] == 0
    ), "a blackout is credited only if the SAME id is held before and after it"


# --- false positive tracks -----------------------------------------------------------------


def test_false_positive_track_never_matched_is_counted() -> None:
    gt_world = _gt_always_present()
    gt_visible = _gt_visible_with_blackout((N_FRAMES, N_FRAMES))
    results = [
        [_snapshot(1, (0.0, 0.0), f), _snapshot(99, (50.0, 50.0), f)] for f in range(N_FRAMES)
    ]

    report = evaluate_id_consistency(gt_world, gt_visible, results, n_ids_issued=2)
    assert report.false_positive_tracks == 1


def test_frame_count_mismatch_raises() -> None:
    gt_world = {1: np.zeros((N_FRAMES, 2))}
    gt_visible = {1: np.ones((N_FRAMES, 1), dtype=bool)}
    results = [[] for _ in range(N_FRAMES - 1)]  # wrong length
    with pytest.raises(ValueError, match="ground truth has"):
        evaluate_id_consistency(gt_world, gt_visible, results, n_ids_issued=0)
