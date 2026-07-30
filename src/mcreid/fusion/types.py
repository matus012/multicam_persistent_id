"""Frozen interface contract between the per-view stage and the fusion stage.

Anything that produces per-camera tracks — the real YOLO+BoT-SORT pipeline, the
toy simulator, or a dataset loader — emits `ViewObservation`s. The fusion stage
consumes only these, so the two halves can be developed and tested apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


class TrackState(str, Enum):
    """Lifecycle of a global track."""

    TENTATIVE = "tentative"  # born, not yet confirmed — may be a false positive
    CONFIRMED = "confirmed"  # seen by enough frames; drawn on the BEV
    COASTING = "coasting"  # no measurement this frame; constant-velocity predicted
    LOST = "lost"  # coasted too long; held in the ReID gallery for re-association
    DEAD = "dead"  # past the re-association window; ID retired


@dataclass(frozen=True)
class ViewObservation:
    """One per-camera track's state at one frame.

    ``embedding`` must be L2-normalised — cosine similarity is computed as a
    plain dot product throughout the fusion stage.
    """

    camera_id: str
    frame: int
    local_track_id: int
    bbox_xyxy: FloatArray
    embedding: FloatArray
    score: float

    def __post_init__(self) -> None:
        box = np.asarray(self.bbox_xyxy, dtype=np.float64)
        if box.shape != (4,):
            raise ValueError(f"bbox_xyxy must be (4,), got {box.shape}")
        if box[2] < box[0] or box[3] < box[1]:
            raise ValueError(f"malformed bbox {box.tolist()} (need x2>=x1, y2>=y1)")
        emb = np.asarray(self.embedding, dtype=np.float64)
        if emb.ndim != 1:
            raise ValueError(f"embedding must be 1-D, got shape {emb.shape}")
        norm = float(np.linalg.norm(emb))
        if not np.isclose(norm, 1.0, atol=1e-3):
            raise ValueError(f"embedding must be L2-normalised, got ||e|| = {norm:.4f}")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score must be in [0, 1], got {self.score}")


@dataclass(frozen=True)
class GroundObservation:
    """A `ViewObservation` projected onto the floor plane."""

    camera_id: str
    frame: int
    local_track_id: int
    world_xy: FloatArray  # (2,) metres
    world_cov: FloatArray  # (2, 2) m^2 — pixel noise propagated through the homography
    embedding: FloatArray
    score: float
    truncated: bool = False
    """The source box was clipped by the frame edge, so this is a half-body crop.

    Already reflected in ``world_cov`` (inflated, because the feet are missing and
    the foot point is a guess). Carried separately because it also says something
    about the *appearance* vector, which no covariance can express: a crop of half
    a person is a bad ReID query, and a long-gap probe built from one fails at any
    threshold. Without this, a rejected probe cannot be attributed to a tight gate
    versus a bad query, and those want opposite fixes."""

    @property
    def position_sigma_m(self) -> float:
        """Scalar summary of positional uncertainty (mean std over both axes)."""
        return float(np.sqrt(np.trace(self.world_cov) / 2.0))


@dataclass(frozen=True)
class GlobalTrackSnapshot:
    """Immutable per-frame output of the fusion stage — what the BEV draws."""

    global_id: int
    frame: int
    world_xy: FloatArray
    velocity_mps: FloatArray
    covariance: FloatArray
    state: TrackState
    supporting_cameras: tuple[str, ...]
    frames_since_measurement: int
    hits: int

    @property
    def is_coasting(self) -> bool:
        return self.frames_since_measurement > 0
