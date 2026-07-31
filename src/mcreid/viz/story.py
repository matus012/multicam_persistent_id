"""Turn a pipeline run into a narrated showcase: events, captions, cards.

The events this module marks are **detected from the pipeline's own output**,
never from the schedule that generated the scene. That distinction is the whole
value of the render: a caption that says "resurrected under the original ID" is
only worth showing if the frame it points at is a frame where the manager
actually did that. Reading the marks off the scripted occlusion timeline instead
would produce a video that says the same words whether or not the system works.

Consequence worth knowing: if the pipeline stops producing an event, the
corresponding caption disappears from the render rather than going stale, and
`detect_events` returns fewer marks than the scene set up. The CLI treats that
as a failure, which is the intended behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

from mcreid.fusion.types import GlobalTrackSnapshot, TrackState
from mcreid.viz.palette import TEXT_COLOR

Image = npt.NDArray[np.uint8]

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_ACCENT = (150, 220, 255)
_MUTED = (170, 170, 175)


@dataclass(frozen=True)
class EventMark:
    """One narrated moment, anchored to a frame the pipeline produced."""

    frame: int
    title: str
    detail: str
    global_id: int
    hold_s: float = 1.6
    """Freeze-frame duration. The reviewer reads the caption here, so this is
    the difference between a demo that can be followed and the dense one that
    was rejected."""


def _present(snapshots: list[list[GlobalTrackSnapshot]], gid: int) -> list[bool]:
    return [any(s.global_id == gid for s in frame) for frame in snapshots]


def _state_at(
    snapshots: list[list[GlobalTrackSnapshot]], gid: int, frame: int
) -> TrackState | None:
    for snap in snapshots[frame]:
        if snap.global_id == gid:
            return snap.state
    return None


def detect_handoff(
    snapshots: list[list[GlobalTrackSnapshot]], gid: int
) -> tuple[int, tuple[str, ...], tuple[str, ...]] | None:
    """First frame where the track's supporting set gains a camera it never had.

    Returns ``(frame, before, after)``. This is a handoff and not a
    re-acquisition precisely because the ID is continuous across it — the caller
    asserts that by only ever calling this with an ID that survives the clip.
    """
    seen: set[str] = set()
    previous: tuple[str, ...] = ()
    for frame, snaps in enumerate(snapshots):
        for snap in snaps:
            if snap.global_id != gid or snap.state is not TrackState.CONFIRMED:
                continue
            current = tuple(sorted(snap.supporting_cameras))
            if not current:
                continue
            fresh = set(current) - seen
            if fresh and seen:
                return frame, previous, current
            seen.update(current)
            previous = current
    return None


def detect_coast(
    snapshots: list[list[GlobalTrackSnapshot]], gid: int, min_frames: int
) -> tuple[int, int] | None:
    """Longest COASTING run that ends by returning to CONFIRMED under the same ID.

    The "ends by returning" clause is what separates event 2 from event 3: a
    coast that runs out and dies is the start of a resurrection story, not a
    coast-through-occlusion story, and captioning it as the latter would be a
    lie about which mechanism carried the identity.
    """
    runs: list[tuple[int, int]] = []
    current: list[int] = []
    for frame in range(len(snapshots)):
        if _state_at(snapshots, gid, frame) is TrackState.COASTING:
            current.append(frame)
            continue
        if current:
            runs.append((current[0], current[-1]))
            current = []
    if current:
        runs.append((current[0], current[-1]))

    survived = [
        (a, b)
        for a, b in runs
        if b - a + 1 >= min_frames
        and b + 1 < len(snapshots)
        and _state_at(snapshots, gid, b + 1) is TrackState.CONFIRMED
    ]
    return max(survived, key=lambda r: r[1] - r[0]) if survived else None


def detect_resurrection(
    snapshots: list[list[GlobalTrackSnapshot]], gid: int, min_gap: int
) -> tuple[int, int] | None:
    """Return ``(last_measured_frame, return_frame)`` for a return after a dead gap.

    The gap is measured from the last frame the track was actually **measured**,
    not from the frame it vanished from the output. Those differ by the whole
    coasting run — here about 3 s — and the first version of this reported the
    smaller number, which understated a 13.1 s absence as 10 s. The unobserved
    duration is the claim a viewer cares about, so it is the one computed.

    ``min_gap`` should be ``reid_window_frames``: past it the track is gone from
    the re-association gallery entirely, so recovering the ID can only have come
    from the dormant gallery. Below it the claim would be ambiguous.
    """
    present = _present(snapshots, gid)
    measured = [
        frame
        for frame in range(len(snapshots))
        if _state_at(snapshots, gid, frame) is TrackState.CONFIRMED
    ]
    gap_start: int | None = None
    for frame, here in enumerate(present):
        if not here:
            if gap_start is None:
                gap_start = frame
            continue
        if gap_start is not None:
            if frame - gap_start >= min_gap:
                before = [m for m in measured if m < gap_start]
                return (before[-1] if before else gap_start), frame
            gap_start = None
    return None


def wrap_text(text: str, max_width: int, scale: float, thickness: int) -> list[str]:
    """Greedy word wrap measured with the real font metrics, not a char count.

    A character budget guesses; `getTextSize` knows. The first version of the
    caption bar guessed, and the resurrection caption — the longest and the most
    important of the three — ran off the right edge of the frame.
    """
    words = text.split()
    if not words:
        return []
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        (tw, _), _ = cv2.getTextSize(candidate, _FONT, scale, thickness)
        if tw <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def caption_bar(width: int, title: str, detail: str, height: int = 172) -> Image:
    """Wide, high-contrast caption strip. Deliberately large: legibility is the bar.

    Height is fixed regardless of how many lines the detail wraps to — every
    frame in a video must be the same size, and a bar that grew with its text
    would change the frame height mid-clip.
    """
    bar = np.full((height, width, 3), 16, dtype=np.uint8)
    cv2.line(bar, (0, 0), (width, 0), (60, 60, 66), 2)
    cv2.putText(bar, title, (28, 56), _FONT, 1.15, TEXT_COLOR, 3, cv2.LINE_AA)
    for index, line in enumerate(wrap_text(detail, width - 56, 0.74, 2)[:2]):
        cv2.putText(bar, line, (28, 104 + index * 40), _FONT, 0.74, _ACCENT, 2, cv2.LINE_AA)
    return bar


def text_card(
    size: tuple[int, int],
    heading: str,
    lines: list[tuple[str, str]],
    footer: str = "",
) -> Image:
    """Full-frame card: heading, then ``(claim, value)`` rows, then a footer."""
    width, height = size
    card = np.full((height, width, 3), 14, dtype=np.uint8)
    cv2.putText(card, heading, (60, 96), _FONT, 1.5, TEXT_COLOR, 3, cv2.LINE_AA)
    cv2.line(card, (60, 126), (width - 60, 126), (70, 70, 76), 2)

    y = 196
    for claim, value in lines:
        cv2.putText(card, claim, (60, y), _FONT, 0.82, TEXT_COLOR, 2, cv2.LINE_AA)
        if value:
            (tw, _), _ = cv2.getTextSize(value, _FONT, 0.82, 2)
            cv2.putText(card, value, (width - 60 - tw, y), _FONT, 0.82, _ACCENT, 2, cv2.LINE_AA)
        y += 62

    if footer:
        cv2.putText(card, footer, (60, height - 52), _FONT, 0.66, _MUTED, 2, cv2.LINE_AA)
    return card


def stamp_watermark(frame: Image, text: str) -> None:
    """Persistent corner label, drawn in place. Never omitted from a render."""
    (tw, th), _ = cv2.getTextSize(text, _FONT, 0.62, 2)
    x, y = frame.shape[1] - tw - 24, 34
    cv2.rectangle(frame, (x - 12, y - th - 10), (x + tw + 12, y + 12), (12, 12, 14), -1)
    cv2.putText(frame, text, (x, y), _FONT, 0.62, (120, 200, 255), 2, cv2.LINE_AA)
