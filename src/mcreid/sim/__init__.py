"""Synthetic multi-view scene generation (CI stand-in for real footage)."""

from mcreid.sim.toy import (
    AgentSpec,
    NoiseSpec,
    OcclusionEvent,
    ToyDetection,
    ToyScene,
    ToySceneConfig,
    bedroom_rig,
    cardboard_scene,
    crossing_scene,
    generate_scene,
)
from mcreid.sim.virtual_camera import VirtualCamera

__all__ = [
    "AgentSpec",
    "NoiseSpec",
    "OcclusionEvent",
    "ToyDetection",
    "ToyScene",
    "ToySceneConfig",
    "VirtualCamera",
    "bedroom_rig",
    "cardboard_scene",
    "crossing_scene",
    "generate_scene",
]
