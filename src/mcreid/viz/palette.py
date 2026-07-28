"""Deterministic per-ID colours.

The same global ID must get the same colour in every view and on the BEV, in
every run — the whole point of the demo GIF is that the viewer can follow one
colour/number through the occlusions.
"""

from __future__ import annotations

import colorsys

BGR = tuple[int, int, int]

# Golden-ratio hue stepping: consecutive IDs land far apart on the colour wheel.
_GOLDEN_RATIO_CONJUGATE = 0.618033988749895

COAST_COLOR: BGR = (120, 120, 120)
TEXT_COLOR: BGR = (255, 255, 255)
GRID_COLOR: BGR = (60, 60, 60)
FLOOR_COLOR: BGR = (28, 28, 32)
CAMERA_COLOR: BGR = (0, 200, 255)


def id_color(global_id: int, saturation: float = 0.72, value: float = 1.0) -> BGR:
    """Stable BGR colour for a global ID."""
    if global_id < 0:
        raise ValueError(f"global_id must be non-negative, got {global_id}")
    hue = (global_id * _GOLDEN_RATIO_CONJUGATE) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return (int(b * 255), int(g * 255), int(r * 255))


def dim(color: BGR, factor: float = 0.45) -> BGR:
    """Darken a colour — used for coasting tracks."""
    if not 0.0 <= factor <= 1.0:
        raise ValueError(f"factor must be in [0, 1], got {factor}")
    return (int(color[0] * factor), int(color[1] * factor), int(color[2] * factor))
