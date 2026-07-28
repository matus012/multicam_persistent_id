"""Integration tests for long-gap re-identification (the G-M1-2 workstream-1 gate).

The hero leaves every camera for well over a minute and returns. The live,
motion-gated revival path cannot serve this — by then the track is long dead —
so the identity must be recovered from the dormant gallery by appearance alone.
"""

from __future__ import annotations

import numpy as np
import pytest

from mcreid.fusion.dormant import DormantConfig
from mcreid.fusion.global_id import FusionConfig
from mcreid.pipeline import run_toy_scene
from mcreid.sim.toy import generate_scene, long_gap_scene

GATE_SEEDS = [1, 42, 2024]


def test_long_gap_scene_actually_contains_distractors() -> None:
    """Guard the guard.

    An earlier version of this scene had one agent and nothing else, and a
    stateless 25-line stub passed every assertion below — the same failure
    already fixed once for the cardboard gate. If someone simplifies the scene
    again, fail here loudly rather than silently disarming the gate.
    """
    config = long_gap_scene(gap_s=75.0, seed=42)
    assert len(config.agents) >= 2, "the long-gap gate needs a second person in the room"
    assert len(config.static_false_positives) >= 1, "the gate needs a persistent false positive"

    scene = generate_scene(config)
    hero_hidden = ~scene.gt_visible[1].any(axis=1)
    others_present = [
        a for a in scene.gt_world if a != 1 and scene.gt_visible[a][hero_hidden].any()
    ]
    assert others_present, "someone must remain visible while the hero is away"


@pytest.mark.parametrize("seed", GATE_SEEDS)
def test_hero_keeps_id_across_a_75_second_absence(seed: int) -> None:
    """THE long-gap gate: leave for 75 s, come back, same global ID.

    Asserted against a scene that also contains a second person and a persistent
    false positive, so the result cannot be earned by simply never minting a
    second confirmed identity.
    """
    config = long_gap_scene(gap_s=75.0, seed=seed)
    scene = generate_scene(config)
    result = run_toy_scene(scene)
    report = result.report

    assert report.total_id_switches == 0, (
        f"seed={seed}: the hero must keep one global ID across the absence, got "
        f"{report.total_id_switches} switch(es); id sequence "
        f"{report.ids_per_agent.get(1)}\n{report.summary()}"
    )
    assert report.ids_per_agent[1] == [report.ids_per_agent[1][0]], (
        f"seed={seed}: expected a single global id, got {report.ids_per_agent[1]}"
    )
    # The stub beat the real pipeline on both of these, so they are gated now.
    assert report.coverage_visible[1] > 0.95, (
        f"seed={seed}: coverage_visible {report.coverage_visible[1]:.3f} <= 0.95"
    )
    expected_ids = len(config.agents) + len(config.static_false_positives)
    assert result.n_ids_issued <= expected_ids + 1, (
        f"seed={seed}: {result.n_ids_issued} global IDs minted for "
        f"{len(config.agents)} people plus {len(config.static_false_positives)} "
        f"false positive(s) — the tracker is churning identities"
    )
    assert report.false_positive_tracks <= len(config.static_false_positives), (
        f"seed={seed}: {report.false_positive_tracks} false-positive tracks, expected at "
        f"most the {len(config.static_false_positives)} injected"
    )


@pytest.mark.parametrize("seed", [1, 42])
def test_disabling_the_dormant_gallery_loses_the_identity(seed: int) -> None:
    """Control: without the gallery the same clip mints a new ID.

    This is what makes the test above meaningful — it proves the result is
    produced by the dormant path and not by some incidental property of the
    scene.
    """
    scene = generate_scene(long_gap_scene(gap_s=75.0, seed=seed))
    config = FusionConfig(dormant=DormantConfig(enabled=False))
    report = run_toy_scene(scene, fusion_config=config).report

    assert report.total_id_switches >= 1, (
        f"seed={seed}: with the dormant gallery disabled the hero should NOT keep "
        f"their id; got {report.total_id_switches} switches, which means this test "
        "no longer isolates the mechanism"
    )
    assert len(report.ids_per_agent[1]) >= 2


@pytest.mark.parametrize("seed", GATE_SEEDS)
def test_intruder_during_the_gap_does_not_inherit_the_dormant_id(seed: int) -> None:
    """Adversarial: a DIFFERENT person present only during the gap must get their
    own new global ID, never the dormant one.

    A dormant gallery that cannot refuse is worse than no gallery at all: it
    launders an identity swap into a confident-looking track.
    """
    config = long_gap_scene(gap_s=75.0, seed=seed, intruder=True)
    scene = generate_scene(config)
    report = run_toy_scene(scene).report

    hero_ids = set(report.ids_per_agent.get(1, []))
    intruder_ids = set(report.ids_per_agent.get(2, []))
    assert hero_ids, "the hero was never tracked"
    assert intruder_ids, "the intruder was never tracked"

    shared = hero_ids & intruder_ids
    assert not shared, (
        f"seed={seed}: global id(s) {sorted(shared)} were used for BOTH the hero and the "
        f"intruder — the dormant identity was inherited by a different person. "
        f"hero={sorted(hero_ids)} intruder={sorted(intruder_ids)}"
    )


@pytest.mark.parametrize("seed", GATE_SEEDS)
def test_hero_returns_under_its_original_id_even_with_an_intruder(seed: int) -> None:
    """The hero's first and last observed identity must be the same one.

    A short-lived duplicate track can appear at the moment of reappearance (see
    the known-transient note in status.txt), so this asserts the identity is
    recovered, not that the path there is perfectly clean.
    """
    scene = generate_scene(long_gap_scene(gap_s=75.0, seed=seed, intruder=True))
    report = run_toy_scene(scene).report
    sequence = report.ids_per_agent.get(1, [])

    assert sequence, "the hero was never tracked"
    assert sequence[0] == sequence[-1], (
        f"seed={seed}: the hero ended the clip under a different global id than it "
        f"started with: {sequence}"
    )


def test_a_stateless_stub_cannot_pass_the_long_gap_gate() -> None:
    """Stub attack, as applied to G-M1-1.

    A tracker with no memory across the gap cannot produce a single stable ID,
    because the hero walks while unobserved and returns somewhere else entirely.
    Emitting a fixed ID for whatever is currently visible fails the moment the
    intruder is the only person in the room.
    """
    from mcreid.calib.geometry import feet_point, image_to_ground
    from mcreid.eval.id_metrics import evaluate_id_consistency
    from mcreid.fusion.types import GlobalTrackSnapshot, TrackState

    scene = generate_scene(long_gap_scene(gap_s=75.0, seed=42, intruder=True))
    snapshots: list[list[GlobalTrackSnapshot]] = []
    for frame in range(scene.n_frames):
        points = []
        for camera_id, detections in scene.frame_detections(frame).items():
            camera = scene.rig.get(camera_id)
            for detection in detections:
                world, ok = image_to_ground(camera, feet_point(detection.bbox_xyxy))
                if ok[0] and np.isfinite(world[0]).all():
                    points.append(world[0])
        if not points:
            snapshots.append([])
            continue
        snapshots.append(
            [
                GlobalTrackSnapshot(
                    global_id=1,
                    frame=frame,
                    world_xy=np.median(np.stack(points), axis=0),
                    velocity_mps=np.zeros(2),
                    covariance=np.eye(2) * 0.05,
                    state=TrackState.CONFIRMED,
                    supporting_cameras=("cam0",),
                    frames_since_measurement=0,
                    hits=frame + 1,
                )
            ]
        )

    config = long_gap_scene(gap_s=75.0, seed=42, intruder=True)
    report = evaluate_id_consistency(
        scene.gt_world, scene.gt_visible, snapshots, n_ids_issued=1
    )
    hero_ids = report.ids_per_agent.get(1, [])
    passes_gate = (
        report.total_id_switches == 0
        and len(hero_ids) == 1
        and report.coverage_visible.get(1, 0.0) > 0.95
        and report.false_positive_tracks <= len(config.static_false_positives)
    )
    assert not passes_gate, (
        "a stateless stub passed the long-gap gate — the scene has stopped testing "
        f"identity. hero ids={hero_ids} switches={report.total_id_switches} "
        f"coverage_visible={report.coverage_visible.get(1)}"
    )
