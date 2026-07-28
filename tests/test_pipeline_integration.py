"""End-to-end integration tests for mcreid.pipeline.

test_cardboard_scene_is_the_gm1_1_gate below IS the G-M1-1 ship gate: one
agent, occluded camera-by-camera and then totally, must keep exactly one
global ID for the whole clip, on every listed seed.
"""

from __future__ import annotations

import numpy as np
import pytest

from mcreid.pipeline import run_toy_scene
from mcreid.sim.toy import cardboard_scene, crossing_scene, generate_scene

GATE_SEEDS = [1, 7, 42, 123, 2024]


@pytest.mark.parametrize("seed", GATE_SEEDS)
def test_cardboard_scene_is_the_gm1_1_gate(seed: int) -> None:
    """THE G-M1-1 gate.

    The hero agent walks a loop through escalating occlusions and a 2.5 s total
    blackout, and must keep one global ID for the entire clip on every seed.

    The scene deliberately also contains a second person and a persistent false
    positive. Without them this gate is not a gate: with a single agent, "zero ID
    switches" is achieved by any tracker that never mints a second confirmed ID,
    and a stateless stub with no ReID, no filter and no lifecycle passes all five
    seeds outright.
    """
    config = cardboard_scene(seed=seed)
    scene = generate_scene(config)
    result = run_toy_scene(scene)
    report = result.report
    blackout = max(e.n_frames for e in config.occlusions if e.camera_ids is None)

    assert report.total_id_switches == 0, (
        f"seed={seed}: G-M1-1 requires zero ID switches, got {report.total_id_switches}\n"
        f"{report.summary()}"
    )
    assert report.longest_blackout_id_held[1] == blackout, (
        f"seed={seed}: the hero must come out of the full {blackout}-frame blackout "
        f"under the same global ID, got {report.longest_blackout_id_held[1]}"
    )
    # The dot must never vanish; accuracy during the coast is reported, not gated
    # (constant-velocity prediction degrades, and ReID does the final re-lock).
    assert report.longest_blackout_alive[1] == blackout, (
        f"seed={seed}: the BEV dot must stay alive for the whole blackout, "
        f"got {report.longest_blackout_alive[1]}/{blackout}"
    )
    assert (
        report.coverage_visible[1] > 0.95
    ), f"seed={seed}: coverage_visible must exceed 0.95, got {report.coverage_visible[1]}"
    assert report.false_positive_tracks <= len(config.static_false_positives), (
        f"seed={seed}: expected at most {len(config.static_false_positives)} false-positive "
        f"tracks (the injected static ones), got {report.false_positive_tracks}"
    )


def test_gate_scene_actually_contains_distractors() -> None:
    """Guard the guard: the gate is only meaningful with more than one body.

    If someone simplifies `cardboard_scene` back to a single agent, the gate
    silently stops testing identity at all. This asserts the scene composition
    the gate depends on.
    """
    config = cardboard_scene(seed=42)
    assert len(config.agents) >= 2, "gate scene needs a distractor agent"
    assert len(config.static_false_positives) >= 1, "gate scene needs a persistent false positive"

    scene = generate_scene(config)
    hero = scene.gt_world[1]
    others = [scene.gt_world[a] for a in scene.gt_world if a != 1]
    closest = min(
        float(np.nanmin(np.linalg.norm(hero - other, axis=1))) for other in others
    )
    assert closest < 2.5, (
        f"the distractor never comes within 2.5 m of the hero (min {closest:.2f} m), "
        "so it cannot exercise identity discrimination"
    )


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
