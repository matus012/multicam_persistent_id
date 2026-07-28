"""Late fusion: ground projection, cross-view association, global identities."""

from mcreid.fusion.associate import (
    AppearanceGallery,
    AssociationConfig,
    build_cost_matrix,
    linear_assignment,
)
from mcreid.fusion.global_id import FusionConfig, GlobalIDManager, GlobalTrack
from mcreid.fusion.motion import CHI2_2DOF_95, CHI2_2DOF_99, GroundKalman
from mcreid.fusion.types import (
    GlobalTrackSnapshot,
    GroundObservation,
    TrackState,
    ViewObservation,
)

__all__ = [
    "CHI2_2DOF_95",
    "CHI2_2DOF_99",
    "AppearanceGallery",
    "AssociationConfig",
    "FusionConfig",
    "GlobalIDManager",
    "GlobalTrack",
    "GlobalTrackSnapshot",
    "GroundKalman",
    "GroundObservation",
    "TrackState",
    "ViewObservation",
    "build_cost_matrix",
    "linear_assignment",
]
