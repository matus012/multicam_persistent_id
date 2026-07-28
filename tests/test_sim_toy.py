"""Tests for mcreid.sim.toy — scripted multi-view toy sequences."""

from __future__ import annotations

import numpy as np
import pytest

from mcreid.sim.toy import (
    AgentSpec,
    NoiseSpec,
    OcclusionEvent,
    ToySceneConfig,
    bedroom_rig,
    cardboard_scene,
    crossing_scene,
    generate_scene,
)


def _agent(**overrides: object) -> AgentSpec:
    kwargs: dict[str, object] = dict(
        agent_id=1,
        waypoints_m=((0.0, 0.0), (2.0, 0.0)),
        speed_mps=1.0,
    )
    kwargs.update(overrides)
    return AgentSpec(**kwargs)  # type: ignore[arg-type]


# --- AgentSpec ------------------------------------------------------------------


def test_agent_spec_rejects_too_few_waypoints() -> None:
    with pytest.raises(ValueError, match="waypoints"):
        _agent(waypoints_m=((0.0, 0.0),))


def test_agent_spec_rejects_non_positive_speed() -> None:
    with pytest.raises(ValueError, match="speed"):
        _agent(speed_mps=0.0)


def test_agent_spec_rejects_non_positive_body_dims() -> None:
    with pytest.raises(ValueError, match="dimensions"):
        _agent(height_m=0.0)
    with pytest.raises(ValueError, match="dimensions"):
        _agent(width_m=-0.1)


def test_agent_spec_position_at_start_and_ping_pong() -> None:
    agent = _agent(waypoints_m=((0.0, 0.0), (2.0, 0.0)), speed_mps=1.0)
    assert np.allclose(agent.position_at(0.0), [0.0, 0.0])
    assert np.allclose(agent.position_at(1.0), [1.0, 0.0])
    assert np.allclose(agent.position_at(2.0), [2.0, 0.0]), "reaches the far waypoint at t=2s"
    # Total round trip is 4 m (there and back) at 1 m/s -> period 4 s.
    assert np.allclose(agent.position_at(3.0), [1.0, 0.0]), "ping-pongs back at t=3s"
    assert np.allclose(agent.position_at(4.0), [0.0, 0.0]), "back at the start after one period"


# --- OcclusionEvent ---------------------------------------------------------------


def test_occlusion_event_rejects_empty_window() -> None:
    with pytest.raises(ValueError, match="empty occlusion window"):
        OcclusionEvent(agent_id=1, start_frame=10, end_frame=10)


def test_occlusion_event_rejects_negative_start() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        OcclusionEvent(agent_id=1, start_frame=-1, end_frame=5)


def test_occlusion_event_hides_specific_cameras_only() -> None:
    occ = OcclusionEvent(agent_id=1, start_frame=10, end_frame=20, camera_ids=("cam0", "cam1"))
    assert occ.hides(1, "cam0", 15)
    assert not occ.hides(1, "cam2", 15), "camera not in the list is unaffected"
    assert not occ.hides(1, "cam0", 25), "outside the frame window is unaffected"
    assert not occ.hides(2, "cam0", 15), "different agent is unaffected"
    assert occ.n_frames == 10


def test_occlusion_event_camera_ids_none_hides_every_camera() -> None:
    occ = OcclusionEvent(agent_id=1, start_frame=0, end_frame=5, camera_ids=None)
    assert occ.hides(1, "cam0", 2)
    assert occ.hides(1, "cam_anything", 2)


# --- NoiseSpec ---------------------------------------------------------------------


def test_noise_spec_rejects_negative_magnitudes() -> None:
    with pytest.raises(ValueError, match="bbox_jitter_px"):
        NoiseSpec(bbox_jitter_px=-1.0)


def test_noise_spec_rejects_probabilities_out_of_range() -> None:
    with pytest.raises(ValueError, match="dropout_prob"):
        NoiseSpec(dropout_prob=1.5)
    with pytest.raises(ValueError, match="false_positive_rate"):
        NoiseSpec(false_positive_rate=-0.1)


# --- ToySceneConfig ------------------------------------------------------------------


def _minimal_config(**overrides: object) -> ToySceneConfig:
    kwargs: dict[str, object] = dict(
        cameras=bedroom_rig(),
        agents=(_agent(),),
        n_frames=10,
        embed_dim=16,
    )
    kwargs.update(overrides)
    return ToySceneConfig(**kwargs)  # type: ignore[arg-type]


def test_toy_scene_config_rejects_no_cameras() -> None:
    with pytest.raises(ValueError, match=">= 1 camera"):
        _minimal_config(cameras=())


def test_toy_scene_config_rejects_no_agents() -> None:
    with pytest.raises(ValueError, match=">= 1 agent"):
        _minimal_config(agents=())


def test_toy_scene_config_rejects_low_embed_dim() -> None:
    with pytest.raises(ValueError, match="embed_dim"):
        _minimal_config(embed_dim=4)


def test_toy_scene_config_rejects_duplicate_camera_ids() -> None:
    cams = bedroom_rig()
    with pytest.raises(ValueError, match="duplicate camera_id"):
        _minimal_config(cameras=(cams[0], cams[0]))


def test_toy_scene_config_rejects_duplicate_agent_ids() -> None:
    with pytest.raises(ValueError, match="duplicate agent_id"):
        _minimal_config(agents=(_agent(agent_id=1), _agent(agent_id=1)))


def test_toy_scene_config_rejects_occlusion_with_unknown_agent() -> None:
    bad_occ = (OcclusionEvent(agent_id=99, start_frame=0, end_frame=2),)
    with pytest.raises(ValueError, match="unknown agent"):
        _minimal_config(occlusions=bad_occ)


def test_toy_scene_config_rejects_occlusion_with_unknown_camera() -> None:
    bad_occ = (OcclusionEvent(agent_id=1, start_frame=0, end_frame=2, camera_ids=("nope",)),)
    with pytest.raises(ValueError, match="unknown camera"):
        _minimal_config(occlusions=bad_occ)


# --- bedroom_rig ---------------------------------------------------------------------


def test_bedroom_rig_has_four_unique_cameras() -> None:
    cams = bedroom_rig()
    assert len(cams) == 4
    assert len({c.camera_id for c in cams}) == 4


# --- generate_scene: determinism + shapes ---------------------------------------------


def test_generate_scene_is_deterministic_given_seed() -> None:
    config = cardboard_scene(seed=7)
    scene_a = generate_scene(config)
    scene_b = generate_scene(config)

    for frame in range(scene_a.n_frames):
        for cam_id in scene_a.config.camera_ids:
            dets_a = scene_a.frame_detections(frame)[cam_id]
            dets_b = scene_b.frame_detections(frame)[cam_id]
            assert len(dets_a) == len(dets_b)
            for da, db in zip(dets_a, dets_b, strict=True):
                assert np.allclose(da.bbox_xyxy, db.bbox_xyxy)
                assert np.allclose(da.embedding, db.embedding)
                assert da.score == db.score
                assert da.gt_agent_id == db.gt_agent_id


def test_generate_scene_shapes() -> None:
    config = _minimal_config(n_frames=20)
    scene = generate_scene(config)
    assert scene.n_frames == 20
    assert len(scene.detections) == 20
    assert scene.gt_world[1].shape == (20, 2)
    assert scene.gt_visible[1].shape == (20, len(config.cameras))


def test_generate_scene_produces_false_positives_when_configured() -> None:
    config = _minimal_config(
        n_frames=60, noise=NoiseSpec(false_positive_rate=1.0, dropout_prob=0.0)
    )
    scene = generate_scene(config)
    any_fp = any(
        det.gt_agent_id is None
        for frame_dets in scene.detections
        for cam_dets in frame_dets.values()
        for det in cam_dets
    )
    assert any_fp, "false_positive_rate=1.0 must produce at least one ghost detection"


def test_generate_scene_embeddings_are_unit_norm() -> None:
    config = _minimal_config(n_frames=10)
    scene = generate_scene(config)
    for frame_dets in scene.detections:
        for cam_dets in frame_dets.values():
            for det in cam_dets:
                assert np.linalg.norm(det.embedding) == pytest.approx(1.0, abs=1e-6)


# --- cardboard_scene / crossing_scene --------------------------------------------------


def test_cardboard_scene_raises_when_too_short_for_schedule() -> None:
    with pytest.raises(ValueError, match="leaves no room"):
        cardboard_scene(n_frames=50)


def test_cardboard_scene_blackout_matches_the_gm1_1_gate_schedule() -> None:
    scene = generate_scene(cardboard_scene(n_frames=420, seed=42, blackout_s=2.5))
    assert scene.longest_blackout(1) == 75, "2.5s @ 30fps total blackout must be exactly 75 frames"


def test_crossing_scene_runs_and_has_two_agents() -> None:
    config = crossing_scene(n_frames=60, seed=1)
    scene = generate_scene(config)
    assert set(scene.gt_world) == {1, 2}
    assert scene.n_frames == 60
