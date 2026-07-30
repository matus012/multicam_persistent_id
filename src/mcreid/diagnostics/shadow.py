"""Shadow probe — measure the long-gap appearance distance every frame.

The live system probes the dormant gallery **once**, on the first frame a
returning person is clustered, and then commits. Two live cycles therefore
produced three data points, which is not enough to set a threshold or a
re-probe schedule. This module asks the same question the real probe asks, on
*every* frame, and writes the answer down without acting on it.

What it is for, concretely: the d(t) curve of a return, split by whether the
source box was clipped by the frame edge. That curve is what says whether a
rejected probe would have been accepted a few frames later once the person was
fully in view — and therefore whether the defect is the threshold or the timing.

**It never influences tracking.** It reads the manager after `step()` has already
run and computes distances against its own frozen copies of the gallery.

Two deliberate design choices:

1. **Frozen snapshots, real behaviour.** Resurrection is *not* suppressed, so a
   shadow session is simultaneously a valid behavioural observation — which
   matters when the webcam session is run once. The cost is that a resurrected
   entry leaves the live gallery mid-return, so the curve would stop exactly
   where it gets interesting. Snapshots solve that: every entry is copied on
   first sight and probed for the rest of the session, resurrected or not.

2. **Two query kinds, recorded separately.** `obs` is the raw per-frame
   observation embedding — what `_resurrect` probes with. `track_ema` is the
   track's running EMA, what `_adopt_dormant_identity` probes with. Both are the
   real queries, not reconstructions.

   There is a contamination risk here worth naming, because it would fake the
   exact result this run is looking for: adoption seeds the stored vectors into
   the resurrected track's own gallery, and if those reached the EMA then the
   measured distance would fall toward zero for bookkeeping reasons and look
   like "waiting improved the query". It does not happen —
   `AppearanceGallery.seed` adds vectors for matching breadth *without* moving
   the EMA, by design. That invariant is load-bearing for this measurement, so
   `test_seeding_a_gallery_must_not_move_the_ema` pins it.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from mcreid.fusion.dormant import DormantEntry
from mcreid.fusion.global_id import GlobalIDManager, GlobalTrack
from mcreid.utils.logging import get_logger

logger = get_logger(__name__)

QUERY_OBS = "obs"
QUERY_TRACK_EMA = "track_ema"
DORMANT_SEED_KEY = "_dormant"
"""The pseudo-camera under which a resurrected track stores inherited vectors."""


@dataclass(frozen=True)
class ShadowRow:
    """One (frame, query, dormant entry) distance measurement."""

    frame: int
    t_s: float
    episode: int
    frames_into_episode: int
    """Frames since this presence episode began — the x axis of the d(t) curve."""
    source: str
    track_id: int
    hits: int
    n_obs: int
    n_truncated: int
    truncated: int
    """1 if any box behind this query was clipped by the frame edge."""
    sigma_m: float
    score: float
    entry_key: str
    """``<global_id>@<retired_frame>`` — distinguishes re-admissions of one ID."""
    entry_id: int
    distance: float
    gate: float
    would_accept: int
    event: str


class ShadowProbe:
    """Per-frame dormant-distance recorder. Read-only with respect to tracking."""

    def __init__(
        self,
        out_path: Path,
        gate: float,
        top_k: int,
        episode_gap_frames: int = 15,
    ) -> None:
        self.out_path = Path(out_path)
        self.gate = gate
        self.top_k = top_k
        self.episode_gap_frames = episode_gap_frames
        """Frames of emptiness that end a presence episode. 15 @ ~20 FPS is
        0.75 s — longer than a detector dropout, far shorter than leaving."""
        self.rows: list[ShadowRow] = []
        self._snapshots: dict[str, DormantEntry] = {}
        self._episode = 0
        self._episode_start: int | None = None
        self._last_seen_frame: int | None = None
        self._seen_ids: set[int] = set()

    # --- capture ----------------------------------------------------------

    def _refresh_snapshots(self, manager: GlobalIDManager) -> None:
        """Copy any gallery entry not already frozen. Existing copies are kept."""
        for gid in manager.dormant.ids:
            entry = manager.dormant.entry(gid)
            key = f"{gid}@{entry.retired_frame}"
            if key not in self._snapshots:
                self._snapshots[key] = entry
                logger.info("shadow: froze dormant entry %s for continued probing", key)

    def _update_episode(self, frame: int, present: bool) -> None:
        if not present:
            return
        gap_ended = (
            self._last_seen_frame is None
            or frame - self._last_seen_frame > self.episode_gap_frames
        )
        if gap_ended:
            self._episode += 1
            self._episode_start = frame
        self._last_seen_frame = frame

    def observe(self, manager: GlobalIDManager, frame: int, t_s: float) -> None:
        """Record every dormant distance visible this frame. Call after step()."""
        self._refresh_snapshots(manager)
        ground = manager.last_ground
        self._update_episode(frame, present=bool(ground))
        if not self._snapshots or not ground:
            return

        event = self._detect_event(manager)
        into = frame - self._episode_start if self._episode_start is not None else 0

        # --- one row per raw observation: exactly what _resurrect probes with.
        for obs in ground:
            track_id = manager.last_assignment.get((obs.camera_id, obs.local_track_id), -1)
            self._emit(
                query=obs.embedding,
                frame=frame,
                t_s=t_s,
                into=into,
                source=QUERY_OBS,
                track_id=track_id,
                hits=self._hits_of(manager, track_id),
                n_obs=1,
                n_truncated=int(obs.truncated),
                sigma_m=obs.position_sigma_m,
                score=obs.score,
                event=event,
            )

        # --- one row per live track: what _adopt_dormant_identity probes with.
        for track in manager.tracks:
            ema = self._ema_query(track)
            if ema is None:
                continue
            self._emit(
                query=ema,
                frame=frame,
                t_s=t_s,
                into=into,
                source=QUERY_TRACK_EMA,
                track_id=track.global_id,
                hits=track.hits,
                n_obs=track.last_observations,
                n_truncated=track.last_truncated,
                sigma_m=float(np.sqrt(np.trace(track.cov[:2, :2]) / 2.0)),
                score=float("nan"),
                event=event,
            )

    def _emit(
        self,
        query: npt.ArrayLike,
        frame: int,
        t_s: float,
        into: int,
        source: str,
        track_id: int,
        hits: int,
        n_obs: int,
        n_truncated: int,
        sigma_m: float,
        score: float,
        event: str,
    ) -> None:
        vector = np.asarray(query, dtype=np.float64).reshape(1, -1)
        for key in sorted(self._snapshots):
            entry = self._snapshots[key]
            if entry.embeddings.size == 0 or entry.embeddings.shape[1] != vector.shape[1]:
                continue
            distance = float(entry.distance(vector, self.top_k)[0])
            self.rows.append(
                ShadowRow(
                    frame=frame,
                    t_s=t_s,
                    episode=self._episode,
                    frames_into_episode=into,
                    source=source,
                    track_id=track_id,
                    hits=hits,
                    n_obs=n_obs,
                    n_truncated=n_truncated,
                    truncated=int(n_truncated > 0),
                    sigma_m=sigma_m,
                    score=score,
                    entry_key=key,
                    entry_id=entry.global_id,
                    distance=distance,
                    gate=self.gate,
                    would_accept=int(distance <= self.gate),
                    event=event,
                )
            )

    def _detect_event(self, manager: GlobalIDManager) -> str:
        """Note real resurrections/births so the curve can be annotated."""
        live = {t.global_id for t in manager.tracks}
        fresh = live - self._seen_ids
        self._seen_ids |= live
        return ";".join(f"appeared:{gid}" for gid in sorted(fresh))

    @staticmethod
    def _hits_of(manager: GlobalIDManager, track_id: int) -> int:
        for track in manager.tracks:
            if track.global_id == track_id:
                return track.hits
        return 0

    @staticmethod
    def _ema_query(track: GlobalTrack) -> npt.NDArray[np.float64] | None:
        """Exactly the query `_adopt_dormant_identity` uses — the track's EMA.

        Safe to use verbatim because `AppearanceGallery.seed` adds inherited
        vectors without moving the EMA, so a resurrected track's EMA is not
        polluted by the entry it was resurrected from. Reconstructing a
        "cleaned" query here instead would measure a statistic the real system
        never computes.
        """
        return track.gallery.ema

    # --- output -----------------------------------------------------------

    def write(self) -> tuple[Path, Path]:
        """Write the JSONL record and the long-format CSV. Returns both paths."""
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl = self.out_path.with_suffix(".jsonl")
        csv_path = self.out_path.with_suffix(".csv")

        with jsonl.open("w", encoding="utf-8") as fh:
            for row in self.rows:
                fh.write(json.dumps(asdict(row)) + "\n")

        fields = list(ShadowRow.__dataclass_fields__)
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            for row in self.rows:
                writer.writerow(asdict(row))

        logger.info("shadow: wrote %d rows -> %s and %s", len(self.rows), jsonl, csv_path)
        return jsonl, csv_path


def _stats(values: list[float]) -> str:
    arr = np.asarray(values, dtype=np.float64)
    sd = float(arr.std(ddof=1)) if arr.size > 1 else float("nan")
    return (
        f"n={arr.size:5d}  mean {arr.mean():.3f}  sd {sd:.3f}  "
        f"min {arr.min():.3f}  p50 {np.median(arr):.3f}  "
        f"p95 {np.percentile(arr, 95):.3f}  max {arr.max():.3f}"
    )


def summarise(rows: list[ShadowRow], gate: float) -> list[str]:
    """The numbers the run was for: d split by truncation, and d(t) per episode."""
    if not rows:
        return ["shadow probe: no rows (the gallery was never populated)"]

    out = [f"shadow probe: {len(rows)} measurements, gate {gate:.2f}"]

    by_group: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        by_group[(row.source, row.truncated)].append(row.distance)
    out.append("  distance by query kind and truncation:")
    for (source, truncated) in sorted(by_group):
        label = "truncated" if truncated else "clean    "
        out.append(f"    {source:9s} {label}  {_stats(by_group[(source, truncated)])}")

    clean = [r.distance for r in rows if not r.truncated and r.source == QUERY_OBS]
    dirty = [r.distance for r in rows if r.truncated and r.source == QUERY_OBS]
    if clean and dirty:
        shift = float(np.mean(dirty) - np.mean(clean))
        out.append(
            f"  truncation costs {shift:+.3f} of appearance distance "
            f"({np.mean(clean):.3f} clean vs {np.mean(dirty):.3f} truncated)"
        )

    # d(t): the curve that sets n_init and any re-probe schedule.
    out.append("  d(t) per episode, obs query, per dormant entry:")
    per_ep: dict[tuple[int, str], list[ShadowRow]] = defaultdict(list)
    for row in rows:
        if row.source == QUERY_OBS:
            per_ep[(row.episode, row.entry_key)].append(row)
    for (episode, key) in sorted(per_ep):
        series = sorted(per_ep[(episode, key)], key=lambda r: r.frames_into_episode)
        first_ok = next((r for r in series if r.would_accept), None)
        first_clean = next((r for r in series if not r.truncated), None)
        marks = []
        for target in (0, 2, 4, 9, 19, 29):
            hit = next((r for r in series if r.frames_into_episode >= target), None)
            if hit is not None:
                flag = "T" if hit.truncated else "-"
                marks.append(f"f+{target:<2d} {hit.distance:.3f}{flag}")
        out.append(f"    episode {episode} vs {key}: " + "  ".join(marks))
        out.append(
            "      first frame under the gate: "
            + (
                f"f+{first_ok.frames_into_episode} at {first_ok.distance:.3f}"
                if first_ok
                else "never"
            )
            + " | first untruncated: "
            + (f"f+{first_clean.frames_into_episode}" if first_clean else "never")
        )
    return out
