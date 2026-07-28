"""Per-camera tracker: turns detections into short-lived local tracklets.

Deliberately lightweight and torch-free. The heavy lifting — occlusion coasting,
identity persistence, cross-camera handoff — belongs to the fusion stage; this
layer only has to bridge frame-to-frame gaps within one view and hand up a
temporally smoothed foot point plus a stable appearance vector.

Two-stage association follows ByteTrack: match high-confidence detections first,
then let low-confidence ones rescue tracks that would otherwise be dropped.
Appearance acts as a veto, never as the primary cue — inside a single view over
one frame, IoU is the stronger signal.

The GPU path (Ultralytics BoT-SORT) lives in `mcreid.track.ultralytics_view` and
emits the same `ViewObservation` contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from mcreid.fusion.associate import INFEASIBLE, linear_assignment
from mcreid.fusion.types import ViewObservation
from mcreid.utils.logging import get_logger

logger = get_logger(__name__)

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Detection:
    """One detector output in one view."""

    bbox_xyxy: FloatArray
    score: float
    embedding: FloatArray

    def __post_init__(self) -> None:
        box = np.asarray(self.bbox_xyxy, dtype=np.float64)
        if box.shape != (4,):
            raise ValueError(f"bbox_xyxy must be (4,), got {box.shape}")
        if box[2] < box[0] or box[3] < box[1]:
            raise ValueError(f"malformed bbox {box.tolist()}")


@dataclass(frozen=True)
class PerViewConfig:
    high_score: float = 0.55
    low_score: float = 0.15
    iou_match_thresh: float = 0.20
    """Minimum IoU for a stage-1 match (cost = 1 - IoU, so max cost = 0.80)."""
    iou_match_thresh_low: float = 0.10
    """Looser threshold for the low-confidence rescue stage."""
    max_appearance_distance: float = 0.55
    """Appearance veto. Loose on purpose: within one view the same person can
    change appearance fast (turning, partial occlusion) and IoU already
    constrains the match."""
    max_age: int = 30
    """Frames a local track survives without a detection before deletion."""
    n_init: int = 2
    velocity_alpha: float = 0.5
    embedding_alpha: float = 0.9

    def __post_init__(self) -> None:
        if self.low_score >= self.high_score:
            raise ValueError("low_score must be below high_score")
        if not 0.0 < self.velocity_alpha <= 1.0:
            raise ValueError("velocity_alpha must be in (0, 1]")
        if not 0.0 < self.embedding_alpha < 1.0:
            raise ValueError("embedding_alpha must be in (0, 1)")


def iou_matrix(boxes_a: FloatArray, boxes_b: FloatArray) -> FloatArray:
    """Pairwise IoU between (N, 4) and (M, 4) xyxy boxes."""
    if boxes_a.size == 0 or boxes_b.size == 0:
        return np.zeros((boxes_a.shape[0], boxes_b.shape[0]), dtype=np.float64)

    area_a = np.clip(boxes_a[:, 2] - boxes_a[:, 0], 0, None) * np.clip(
        boxes_a[:, 3] - boxes_a[:, 1], 0, None
    )
    area_b = np.clip(boxes_b[:, 2] - boxes_b[:, 0], 0, None) * np.clip(
        boxes_b[:, 3] - boxes_b[:, 1], 0, None
    )
    lt = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    rb = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0.0, inter / np.maximum(union, 1e-12), 0.0)


class _LocalTrack:
    """One tracklet inside one camera."""

    __slots__ = (
        "box",
        "config",
        "confirmed",
        "embedding",
        "hits",
        "score",
        "time_since_update",
        "track_id",
        "velocity",
    )

    def __init__(self, track_id: int, detection: Detection, config: PerViewConfig) -> None:
        self.track_id = track_id
        self.box = np.asarray(detection.bbox_xyxy, dtype=np.float64).copy()
        self.velocity = np.zeros(4, dtype=np.float64)
        self.embedding = _unit(detection.embedding)
        self.score = detection.score
        self.hits = 1
        self.time_since_update = 0
        self.config = config
        self.confirmed = config.n_init <= 1

    def predict(self) -> None:
        self.box = self.box + self.velocity
        self.time_since_update += 1

    def update(self, detection: Detection) -> None:
        new_box = np.asarray(detection.bbox_xyxy, dtype=np.float64)
        alpha = self.config.velocity_alpha
        self.velocity = alpha * (new_box - self.box) + (1.0 - alpha) * self.velocity
        self.box = new_box.copy()

        beta = self.config.embedding_alpha
        blended = beta * self.embedding + (1.0 - beta) * _unit(detection.embedding)
        self.embedding = _unit(blended)

        self.score = detection.score
        self.hits += 1
        self.time_since_update = 0
        if self.hits >= self.config.n_init:
            self.confirmed = True


def _unit(vec: npt.ArrayLike) -> FloatArray:
    arr = np.asarray(vec, dtype=np.float64).ravel()
    norm = float(np.linalg.norm(arr))
    if norm < 1e-12:
        raise ValueError("cannot normalise a zero embedding")
    return arr / norm


class PerViewTracker:
    """Stateful per-camera tracker emitting `ViewObservation`s."""

    def __init__(self, camera_id: str, config: PerViewConfig | None = None) -> None:
        self.camera_id = camera_id
        self.config = config or PerViewConfig()
        self.tracks: list[_LocalTrack] = []
        self._next_id = 0
        self._frame = -1

    def update(self, detections: list[Detection], frame: int) -> list[ViewObservation]:
        """Advance one frame and return this view's confirmed tracklets."""
        if frame <= self._frame:
            raise ValueError(f"frames must increase: got {frame} after {self._frame}")
        self._frame = frame

        for track in self.tracks:
            track.predict()

        cfg = self.config
        high = [d for d in detections if d.score >= cfg.high_score]
        low = [d for d in detections if cfg.low_score <= d.score < cfg.high_score]

        remaining = list(self.tracks)
        matched_high, unmatched_high, remaining = self._associate(
            high, remaining, 1.0 - cfg.iou_match_thresh
        )
        # Stage 2: only tracks still unmatched compete for low-score detections.
        matched_low, _unmatched_low, remaining = self._associate(
            low, remaining, 1.0 - cfg.iou_match_thresh_low
        )

        for track, detection in [*matched_high, *matched_low]:
            track.update(detection)

        self.tracks = [
            t for t in self.tracks if t.time_since_update <= cfg.max_age and _alive(t, cfg)
        ]
        for detection in unmatched_high:
            self._next_id += 1
            self.tracks.append(_LocalTrack(self._next_id, detection, cfg))

        return [
            ViewObservation(
                camera_id=self.camera_id,
                frame=frame,
                local_track_id=t.track_id,
                bbox_xyxy=t.box.copy(),
                embedding=t.embedding.copy(),
                score=float(np.clip(t.score, 0.0, 1.0)),
            )
            for t in self.tracks
            if t.confirmed and t.time_since_update == 0
        ]

    def _associate(
        self, detections: list[Detection], tracks: list[_LocalTrack], max_cost: float
    ) -> tuple[list[tuple[_LocalTrack, Detection]], list[Detection], list[_LocalTrack]]:
        """IoU assignment with an appearance veto.

        Returns (matched pairs, unmatched detections, still-unmatched tracks).
        """
        if not detections or not tracks:
            return [], list(detections), list(tracks)

        det_boxes = np.stack([np.asarray(d.bbox_xyxy, dtype=np.float64) for d in detections])
        trk_boxes = np.stack([t.box for t in tracks])
        cost = 1.0 - iou_matrix(det_boxes, trk_boxes)

        det_emb = np.stack([_unit(d.embedding) for d in detections])
        trk_emb = np.stack([t.embedding for t in tracks])
        appearance = 1.0 - det_emb @ trk_emb.T
        cost = np.where(appearance > self.config.max_appearance_distance, INFEASIBLE, cost)

        pairs, unmatched_dets, unmatched_trks = linear_assignment(cost, max_cost)
        matched = [(tracks[t_idx], detections[d_idx]) for d_idx, t_idx in pairs]
        return (
            matched,
            [detections[i] for i in unmatched_dets],
            [tracks[i] for i in unmatched_trks],
        )


def _alive(track: _LocalTrack, config: PerViewConfig) -> bool:
    """Unconfirmed tracks get no grace period — that is what kills false positives."""
    return track.confirmed or track.time_since_update == 0
