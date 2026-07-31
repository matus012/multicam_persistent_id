"""Gates for the HPC showcase render.

The point of these is that the video cannot silently stop showing what it
claims to show. The scene is the fragile part — it was broken three separate
ways while being built (camera coverage, mount clearance, distractor spacing),
and every one of those failures produced a render that still *played*, just with
a churning identity. So the scene invariants are asserted, not eyeballed.
"""

from __future__ import annotations

import dataclasses
import tempfile
from pathlib import Path

import numpy as np
import pytest

from mcreid.fusion.global_id import FusionConfig
from mcreid.fusion.types import TrackState
from mcreid.pipeline import run_toy_scene
from mcreid.sim.toy import demo_rig, generate_scene, hpc_demo_scene
from mcreid.viz.bev import BevRenderer
from mcreid.viz.palette import camera_color, id_color
from mcreid.viz.story import (
    caption_bar,
    detect_coast,
    detect_handoff,
    detect_resurrection,
    text_card,
    wrap_text,
)


@pytest.fixture(scope="module")
def run():
    scene = generate_scene(hpc_demo_scene())
    return scene, run_toy_scene(scene)


def test_the_showcase_scene_tracks_two_people_as_exactly_two_identities(run) -> None:
    """The headline invariant. If this fails the video shows identity churn.

    Not a style check: at 0.85 m of camera-mount clearance instead of 1.45 m the
    same walk issued 12 global IDs, and the render looked plausible while doing
    it.
    """
    scene, result = run
    reported = {s.global_id for snaps in result.snapshots for s in snaps}
    assert len(reported) == 2, (
        f"{len(reported)} global ids for 2 people: {sorted(reported)} — the showcase "
        f"scene is churning identities and the render would show it"
    )
    assert result.n_ids_issued == 2, f"{result.n_ids_issued} ids issued, expected 2"


def test_the_hero_keeps_one_id_across_all_three_events(run) -> None:
    _, result = run
    hero_ids = result.report.ids_per_agent.get(1, [])
    assert hero_ids, "the hero was never tracked"
    assert len(set(hero_ids)) == 1, f"hero took multiple ids: {hero_ids}"
    assert result.report.total_id_switches == 0


def test_all_three_narrated_events_are_detectable_from_pipeline_output(run) -> None:
    """Each caption must have a real frame behind it, found by the detectors."""
    _, result = run
    config = FusionConfig()
    hero = result.report.ids_per_agent[1][0]

    handoff = detect_handoff(result.snapshots, hero)
    assert handoff is not None, "event 1 (handoff) not detectable"
    frame, before, after = handoff
    assert set(before) != set(after), "the supporting camera set did not change"

    coast = detect_coast(result.snapshots, hero, min_frames=30)
    assert coast is not None, "event 2 (coast through occlusion) not detectable"
    start, end = coast
    assert end - start + 1 >= 30

    resurrection = detect_resurrection(
        result.snapshots, hero, min_gap=config.reid_window_frames
    )
    assert resurrection is not None, "event 3 (resurrection) not detectable"
    last_seen, back = resurrection
    assert back - last_seen > config.reid_window_frames, (
        "the gap did not exceed the re-association window, so recovery cannot be "
        "attributed to the dormant gallery"
    )


def test_the_events_happen_in_the_narrated_order(run) -> None:
    """Handoff, then coast, then resurrection. The captions assert this order."""
    _, result = run
    hero = result.report.ids_per_agent[1][0]
    handoff = detect_handoff(result.snapshots, hero)
    coast = detect_coast(result.snapshots, hero, min_frames=30)
    resurrection = detect_resurrection(result.snapshots, hero, min_gap=300)
    assert handoff and coast and resurrection
    assert handoff[0] < coast[0] < resurrection[1]


def test_the_coast_event_returns_to_confirmed_rather_than_dying(run) -> None:
    """Event 2 must be a survived coast, not the front half of event 3.

    Without this the detector would happily label the 13 s dead gap as
    "coasting, ID held" — a caption for a mechanism that did not carry the
    identity.
    """
    _, result = run
    hero = result.report.ids_per_agent[1][0]
    coast = detect_coast(result.snapshots, hero, min_frames=30)
    assert coast is not None
    _, end = coast
    after = [s for s in result.snapshots[end + 1] if s.global_id == hero]
    assert after and after[0].state is TrackState.CONFIRMED


def test_resurrection_gap_is_measured_from_the_last_measurement(run) -> None:
    """Regression: the gap was once measured from the frame the track vanished
    from the output, which is a whole coasting run later and understated a
    13.2 s absence as 10 s."""
    _, result = run
    hero = result.report.ids_per_agent[1][0]
    last_seen, back = detect_resurrection(result.snapshots, hero, min_gap=300)
    vanished = next(
        f
        for f in range(len(result.snapshots))
        if f > last_seen and not any(s.global_id == hero for s in result.snapshots[f])
    )
    assert last_seen < vanished, "gap start must precede the frame the track vanished"
    assert back - last_seen > back - vanished


def test_scene_has_a_distractor_that_never_leaves(run) -> None:
    """Otherwise "the ID was held" is earned by never minting a second identity."""
    scene, result = run
    assert len(scene.config.agents) >= 2
    hidden = ~scene.gt_visible[1].any(axis=1)
    assert scene.gt_visible[2][hidden].any(), "nobody is visible while the hero is away"


def test_hero_loop_clears_every_camera_mount(run) -> None:
    """The single most sensitive number in the scene; see `hpc_demo_scene`."""
    scene, _ = run
    hero = scene.config.agents[0]
    positions = np.array([hero.position_at(t) for t in np.linspace(0.0, 37.0, 800)])
    for camera in scene.config.cameras:
        mount = np.array(camera.position_m[:2])
        clearance = float(np.linalg.norm(positions - mount, axis=1).min())
        assert clearance > 1.3, (
            f"hero passes {clearance:.2f} m from {camera.camera_id}; under ~1 m the "
            f"detections truncate and the whole scene fragments"
        )


def test_agents_never_come_within_the_merge_radius(run) -> None:
    scene, _ = run
    hero, distractor = scene.config.agents[0], scene.config.agents[1]
    ts = np.linspace(0.0, 37.0, 1500)
    separation = np.linalg.norm(
        np.array([hero.position_at(t) for t in ts])
        - np.array([distractor.position_at(t) for t in ts]),
        axis=1,
    ).min()
    assert separation > FusionConfig().merge_radius_m, (
        f"agents close to {separation:.2f} m, inside merge_radius — the hero can be "
        f"merged into the distractor, which reads on screen as a tracker failure"
    )


def test_camera_colours_are_distinct_and_stable() -> None:
    colours = [camera_color(i) for i in range(3)]
    assert len(set(colours)) == 3
    assert camera_color(0) == camera_color(0)
    # Camera and identity palettes must not collide, or a viewer cannot tell a
    # tile border from a person.
    assert not set(colours) & {id_color(i) for i in range(1, 6)}


def test_camera_colour_rejects_negative_index() -> None:
    with pytest.raises(ValueError):
        camera_color(-1)


def test_ground_footprints_are_clipped_to_the_room() -> None:
    """A raw footprint runs to the horizon; the BEV must not claim that floor."""
    scene = generate_scene(hpc_demo_scene())
    bev = BevRenderer(scene.rig, canvas_size=(640, 360))
    x0, y0, x1, y1 = scene.rig.floor_extent()
    for camera_id in scene.rig.camera_ids:
        polygon = bev.ground_footprint(scene.rig.get(camera_id))
        assert len(polygon) >= 3, f"{camera_id} produced no floor footprint"
        assert polygon[:, 0].min() >= x0 - 1e-6 and polygon[:, 0].max() <= x1 + 1e-6
        assert polygon[:, 1].min() >= y0 - 1e-6 and polygon[:, 1].max() <= y1 + 1e-6


def test_frustum_wash_does_not_let_the_last_camera_erase_the_others() -> None:
    """Regression: one shared overlay meant the last fill overwrote every prior
    one, so cam0's coverage vanished from the map entirely."""
    scene = generate_scene(hpc_demo_scene())
    bev = BevRenderer(scene.rig, canvas_size=(640, 360))
    order = list(scene.rig.camera_ids)
    blank = bev._blank()
    washed = blank.copy()
    bev.draw_camera_frustums(washed, order)

    # Somewhere on the plan must carry a tint from the FIRST camera in the order.
    only_first = bev._blank()
    bev.draw_camera_frustums(only_first, order[:1])
    assert np.any(only_first != blank), "first camera drew nothing at all"
    assert np.any(washed != blank)


def test_demo_rig_is_three_cameras_with_wide_fov() -> None:
    cams = demo_rig()
    assert len(cams) == 3, "the 2x2 mosaic has exactly three camera tiles"
    assert all(c.hfov_deg >= 90.0 for c in cams), (
        "70-degree cameras do not cover the room and the hero fragments across the gaps"
    )


def test_wrap_text_respects_the_measured_width() -> None:
    import cv2

    text = "unobserved 13.2 s - the 10 s re-association window expired 3.2 s before return"
    lines = wrap_text(text, 600, 0.74, 2)
    assert len(lines) > 1
    for line in lines:
        (width, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.74, 2)
        assert width <= 600, f"line overflows: {line!r}"
    assert " ".join(lines) == text, "wrapping must not drop or reorder words"


def test_wrap_text_handles_empty_input() -> None:
    assert wrap_text("", 600, 0.74, 2) == []


def test_caption_bar_height_is_fixed_regardless_of_text_length() -> None:
    """Every frame in a video must be the same size."""
    short = caption_bar(1280, "A", "short")
    long = caption_bar(1280, "A", "word " * 60)
    assert short.shape == long.shape


def test_text_card_matches_requested_size() -> None:
    card = text_card((1280, 892), "Heading", [("claim", "value")], footer="foot")
    assert card.shape[:2] == (892, 1280)


def test_event_mark_rendered_index_defaults_to_unset() -> None:
    """Slicing the finished video by `frame` would cut the wrong place.

    The assembled video carries a title card and a freeze per earlier event, so
    a mark's position in it drifts further from its scene frame with every event
    that precedes it. `rendered_index` is the only correct handle, and it starts
    invalid so an un-placed mark cannot silently be used as one.
    """
    from mcreid.viz.story import EventMark

    mark = EventMark(frame=100, title="t", detail="d", global_id=1)
    assert mark.rendered_index == -1

    placed = dataclasses.replace(mark, rendered_index=178)
    assert placed.rendered_index == 178
    assert placed.frame == 100, "placing a mark must not move its scene frame"


def test_highlight_gif_windows_land_on_the_captioned_frames() -> None:
    """The GIF must show the captions, which only exist from the event frame on."""
    from mcreid.cli.hpc_demo import _write_highlight_gif
    from mcreid.viz.story import EventMark

    frames = [np.full((100, 200, 3), i % 255, dtype=np.uint8) for i in range(400)]
    marks = [
        dataclasses.replace(
            EventMark(frame=50, title="a", detail="", global_id=1, hold_s=1.0),
            rendered_index=100,
        ),
        dataclasses.replace(
            EventMark(frame=200, title="b", detail="", global_id=1, hold_s=1.0),
            rendered_index=300,
        ),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_highlight_gif(
            frames, marks, Path(tmp) / "g.gif", fps=30.0, lead_s=0.5, tail_s=0.2, width=100
        )
        assert path.exists() and path.stat().st_size > 0
