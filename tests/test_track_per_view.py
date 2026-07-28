"""Tests for mcreid.track.per_view — the lightweight per-camera tracker."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from mcreid.track.per_view import Detection, PerViewConfig, PerViewTracker, iou_matrix

FloatArray = npt.NDArray[np.float64]


def _det(
    box: tuple[float, float, float, float], score: float = 0.9, dim: int = 8, embed_index: int = 0
) -> Detection:
    emb = np.zeros(dim)
    emb[embed_index] = 1.0
    return Detection(bbox_xyxy=np.array(box, dtype=np.float64), score=score, embedding=emb)


# --- Detection validation ------------------------------------------------------------


def test_detection_rejects_malformed_bbox() -> None:
    with pytest.raises(ValueError, match="malformed bbox"):
        Detection(bbox_xyxy=np.array([10.0, 10.0, 5.0, 20.0]), score=0.9, embedding=np.ones(4))


# --- PerViewConfig validation -----------------------------------------------------------


def test_per_view_config_rejects_low_score_above_high_score() -> None:
    with pytest.raises(ValueError, match="low_score must be below high_score"):
        PerViewConfig(low_score=0.6, high_score=0.5)


def test_per_view_config_rejects_bad_velocity_alpha() -> None:
    with pytest.raises(ValueError, match="velocity_alpha"):
        PerViewConfig(velocity_alpha=0.0)


def test_per_view_config_rejects_bad_embedding_alpha() -> None:
    with pytest.raises(ValueError, match="embedding_alpha"):
        PerViewConfig(embedding_alpha=1.0)


# --- iou_matrix ---------------------------------------------------------------------------


def test_iou_matrix_full_overlap_is_one() -> None:
    boxes = np.array([[0.0, 0.0, 10.0, 10.0]])
    iou = iou_matrix(boxes, boxes)
    assert iou[0, 0] == pytest.approx(1.0)


def test_iou_matrix_disjoint_boxes_is_zero() -> None:
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array([[100.0, 100.0, 110.0, 110.0]])
    iou = iou_matrix(a, b)
    assert iou[0, 0] == pytest.approx(0.0)


def test_iou_matrix_partial_overlap() -> None:
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array([[5.0, 0.0, 15.0, 10.0]])
    iou = iou_matrix(a, b)
    # intersection = 5x10=50, union = 100+100-50=150
    assert iou[0, 0] == pytest.approx(50.0 / 150.0)


def test_iou_matrix_empty_inputs() -> None:
    a = np.zeros((0, 4))
    b = np.array([[0.0, 0.0, 1.0, 1.0]])
    iou = iou_matrix(a, b)
    assert iou.shape == (0, 1)


# --- PerViewTracker: frame ordering -------------------------------------------------------


def test_tracker_rejects_non_increasing_frames() -> None:
    tracker = PerViewTracker("cam0")
    tracker.update([_det((0, 0, 10, 10))], 0)
    with pytest.raises(ValueError, match="frames must increase"):
        tracker.update([], 0)


# --- stable ids over a smooth trajectory --------------------------------------------------


def test_stable_local_ids_over_smooth_trajectory() -> None:
    tracker = PerViewTracker("cam0")
    ids_per_frame = []
    for f in range(10):
        box = (100.0 + f * 2, 100.0, 150.0 + f * 2, 200.0)
        obs = tracker.update([_det(box)], f)
        ids_per_frame.append([o.local_track_id for o in obs])

    confirmed_ids = [ids[0] for ids in ids_per_frame if ids]
    assert confirmed_ids, "the trajectory should produce confirmed observations"
    assert len(set(confirmed_ids)) == 1, f"expected one stable id, saw {set(confirmed_ids)}"


# --- detection gap shorter than max_age does not create a new id --------------------------


def test_short_detection_gap_does_not_create_new_id() -> None:
    cfg = PerViewConfig(max_age=30)
    tracker = PerViewTracker("cam0", cfg)

    for f in range(5):
        box = (100.0 + f * 2, 100.0, 150.0 + f * 2, 200.0)
        tracker.update([_det(box)], f)

    gap_len = 10
    assert gap_len < cfg.max_age
    for f in range(5, 5 + gap_len):
        obs = tracker.update([], f)
        assert obs == [], "no detections during the gap means no confirmed observation this frame"

    resume_frame = 5 + gap_len
    box = (100.0 + (resume_frame - 1) * 2, 100.0, 150.0 + (resume_frame - 1) * 2, 200.0)
    obs = tracker.update([_det(box)], resume_frame)
    assert len(obs) == 1
    assert obs[0].local_track_id == 1, "the same local id must persist across the short gap"
    assert len(tracker.tracks) == 1


# --- unconfirmed tracks die immediately on a miss ------------------------------------------


def test_unconfirmed_track_dies_immediately_on_a_miss() -> None:
    tracker = PerViewTracker("cam0")  # default n_init=2
    tracker.update([_det((100.0, 100.0, 150.0, 200.0))], 0)
    assert len(tracker.tracks) == 1
    assert not tracker.tracks[0].confirmed

    tracker.update([], 1)  # single miss, no grace period for unconfirmed tracks
    assert tracker.tracks == [], "an unconfirmed track must be dropped on its very first miss"


# --- low-score detections rescue a track in stage 2 ----------------------------------------


def test_low_score_detection_rescues_track_in_stage_two() -> None:
    cfg = PerViewConfig()
    tracker = PerViewTracker("cam0", cfg)
    box = (100.0, 100.0, 150.0, 200.0)
    tracker.update([_det(box, score=0.9)], 0)
    tracker.update([_det(box, score=0.9)], 1)  # hits=2 -> confirmed (n_init=2)
    assert tracker.tracks[0].confirmed

    low_score_det = _det((102.0, 100.0, 152.0, 200.0), score=0.3)
    assert cfg.low_score <= low_score_det.score < cfg.high_score
    obs = tracker.update([low_score_det], 2)

    assert len(obs) == 1, "a confirmed track must be rescued by a stage-2 low-score detection"
    assert obs[0].local_track_id == 1
    assert tracker.tracks[0].time_since_update == 0
