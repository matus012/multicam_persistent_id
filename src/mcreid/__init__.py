"""mcreid — multi-camera persistent-ID tracking with a live bird's-eye-view map.

Late-fusion architecture (LOCKED):
    per-view detection+tracking+ReID  ->  ground-plane projection  ->  cross-view
    Hungarian association  ->  global ID manager (birth/death, re-association
    window, constant-velocity coasting)  ->  BEV visualisation.
"""

__version__ = "0.1.0"
