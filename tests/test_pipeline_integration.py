"""End-to-end integration tests for mcreid.pipeline.

test_cardboard_scene_is_the_gm1_1_gate below IS the G-M1-1 ship gate: one
agent, occluded camera-by-camera and then totally, must keep exactly one
global ID for the whole clip, on every listed seed.
"""

from __future__ import annotations

import pytest

from mcreid.pipeline import run_toy_scene
from mcreid.sim.toy import cardboard_scene, crossing_scene, generate_scene

GATE_SEEDS = [1, 7, 42, 123, 2024]


@pytest.mark.parametrize("seed", GATE_SEEDS)
def test_cardboard_scene_is_the_gm1_1_gate(seed: int) -> None:
    """THE G-M1-1 gate.

    One agent walking a loop through escalating occlusions and a 2.5s total
    blackout must be tracked under a single global ID for the entire clip,
    across every one of the fixed seeds below. This is the headline
    ship-criterion for the zero-training, late-fusion multi-camera tracker.
    """
    scene = generate_scene(cardboard_scene(seed=seed))
    result = run_toy_scene(scene)
    report = result.report

    assert report.total_id_switches == 0, (
        f"seed={seed}: G-M1-1 requires zero ID switches, got {report.total_id_switches}\n"
        f"{report.summary()}"
    )
    assert report.longest_coast_survived[1] == 75, (
        f"seed={seed}: expected the tracker to survive the full 75-frame (2.5s @ 30fps) "
        f"total blackout under one id, got {report.longest_coast_survived[1]}"
    )
    assert (
        report.coverage_visible[1] > 0.95
    ), f"seed={seed}: coverage_visible must exceed 0.95, got {report.coverage_visible[1]}"
    assert (
        report.false_positive_tracks == 0
    ), f"seed={seed}: expected zero false-positive tracks, got {report.false_positive_tracks}"


def test_cardboard_scene_is_deterministic() -> None:
    """Running the identical scene twice must yield identical global-id sequences."""
    scene_a = generate_scene(cardboard_scene(seed=42))
    scene_b = generate_scene(cardboard_scene(seed=42))

    result_a = run_toy_scene(scene_a)
    result_b = run_toy_scene(scene_b)

    assert len(result_a.snapshots) == len(result_b.snapshots)
    paired = zip(result_a.snapshots, result_b.snapshots, strict=True)
    for frame, (snaps_a, snaps_b) in enumerate(paired):
        gids_a = sorted(s.global_id for s in snaps_a)
        gids_b = sorted(s.global_id for s in snaps_b)
        assert gids_a == gids_b, f"frame {frame}: determinism broke, {gids_a} vs {gids_b}"
    assert result_a.n_ids_issued == result_b.n_ids_issued


def test_crossing_scene_runs_end_to_end_and_is_a_known_limitation() -> None:
    """Secondary demo: two people crossing paths with mutual occlusions.

    This is NOT a v1 gate. The 2-person crossing case is a known limitation of
    the zero-training geometric baseline (no learned appearance model to break
    ties when two people are close together and mutually occlude each other in
    several views at once), so only a loose upper bound is asserted here —
    this test exists to catch a total regression, not to hold the baseline to
    the single-agent cardboard gate's zero-switch bar.
    """
    scene = generate_scene(crossing_scene(seed=42))
    result = run_toy_scene(scene)
    report = result.report

    assert report.total_id_switches <= 6, (
        f"crossing scene: expected <= 6 id switches (loose bound, not a gate), "
        f"got {report.total_id_switches}\n{report.summary()}"
    )
