"""Dormant gallery — long-gap re-identification.

The live re-association path (`GlobalIDManager._revive`) is motion-gated: it asks
"could this person have walked here in the time they were missing?". That gate is
what makes it safe, and it is exactly what stops working once someone leaves the
room for a minute. Past the coasting/revive window a track is therefore
**demoted, not deleted**: its identity and a small set of representative
appearance vectors are kept, with *no position claim at all*, and any newly born
track in any camera is tested against that gallery before a fresh global ID is
minted.

Because no motion gate applies here, appearance has to carry the whole decision,
so this path is deliberately stricter than live re-association on three axes:

1. a tighter cosine threshold;
2. a top-k mean rather than max-similarity (a max over a large gallery has a
   false-accept rate that climbs with gallery size — the wrong property when the
   consequence is giving one person another person's identity);
3. a ratio test — if the two best dormant candidates are comparably close, the
   evidence does not distinguish them and *nothing* is resurrected.

Storing full appearance history would make (2) worse over time and grow without
bound, so each identity keeps a fixed-size representative subset instead.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from mcreid.fusion.associate import INFEASIBLE, linear_assignment
from mcreid.utils.logging import get_logger

logger = get_logger(__name__)

FloatArray = npt.NDArray[np.float64]

ACCEPTED = "accepted"
REJECTED_GATE = "rejected_gate"
REJECTED_AMBIGUOUS = "rejected_ambiguous"
REJECTED_ASSIGNMENT = "rejected_assignment"


@dataclass(frozen=True)
class MatchAttempt:
    """One dormant probe, recorded whether it succeeded or not.

    Exists because the failure it documents was *invisible*: a probe rejected by
    the appearance gate logged nothing at all, so a live run that minted a fresh
    ID for a person who had obviously returned left no evidence of how close it
    had come to recognising them. Without the number there is no way to tell a
    threshold that is slightly too tight from an embedding that is nowhere near
    — and those want opposite fixes.
    """

    query_index: int
    ranked: tuple[tuple[int, float], ...]
    """(global_id, distance) over the whole gallery, best first — ungated."""
    outcome: str
    context: str = ""
    """Caller-supplied provenance for the query (which track, how mature)."""

    @property
    def best(self) -> tuple[int, float] | None:
        return self.ranked[0] if self.ranked else None

    @property
    def runner_up(self) -> tuple[int, float] | None:
        return self.ranked[1] if len(self.ranked) > 1 else None

    def describe(self) -> str:
        ranked = ", ".join(f"id {gid} {dist:.3f}" for gid, dist in self.ranked[:4])
        suffix = f" | {self.context}" if self.context else ""
        return f"query {self.query_index}: {self.outcome} [{ranked}]{suffix}"


def select_representative(vectors: npt.ArrayLike, k: int) -> FloatArray:
    """Pick ``k`` vectors that summarise a set: the medoid, then farthest-point.

    A person's gallery spans viewpoints. Keeping the ``k`` vectors nearest the
    centroid would collapse to one viewpoint and fail to match them from any
    other; keeping ``k`` at random is unstable. Seeding at the medoid and then
    greedily taking the vector farthest from everything kept so far preserves
    both the typical appearance and the spread around it.
    """
    data = np.atleast_2d(np.asarray(vectors, dtype=np.float64))
    if data.size == 0:
        return np.zeros((0, 0), dtype=np.float64)
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if data.shape[0] <= k:
        return data.copy()

    centroid = data.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 1e-12:
        centroid = centroid / norm

    chosen = [int(np.argmax(data @ centroid))]
    while len(chosen) < k:
        # Similarity of every vector to its nearest already-chosen vector;
        # take whichever is least well covered.
        covered = (data @ data[chosen].T).max(axis=1)
        covered[chosen] = np.inf
        chosen.append(int(np.argmin(covered)))
    return data[chosen].copy()


@dataclass
class DormantEntry:
    """A retired identity kept alive for later appearance-only re-matching."""

    global_id: int
    embeddings: FloatArray  # (k, D), L2-normalised
    retired_frame: int
    hits: int
    cameras_seen: tuple[str, ...]
    # Purely informational — the dormant path makes no position claim, and this
    # is never used for gating. It exists so logs are debuggable.
    last_world_xy: FloatArray = field(default_factory=lambda: np.full(2, np.nan))

    def distance(self, queries: npt.ArrayLike, top_k: int) -> FloatArray:
        """Top-k mean cosine distance from each query row to this identity."""
        query = np.atleast_2d(np.asarray(queries, dtype=np.float64))
        if self.embeddings.size == 0:
            return np.ones(query.shape[0], dtype=np.float64)
        if query.shape[1] != self.embeddings.shape[1]:
            raise ValueError(
                f"embedding dim mismatch: query {query.shape[1]} "
                f"vs dormant {self.embeddings.shape[1]}"
            )
        similarity = query @ self.embeddings.T
        k = min(top_k, similarity.shape[1])
        return 1.0 - np.sort(similarity, axis=1)[:, -k:].mean(axis=1)


@dataclass(frozen=True)
class DormantConfig:
    """Long-gap re-identification parameters."""

    enabled: bool = True
    ttl_s: float = 600.0
    """How long a retired identity stays resurrectable. 10 minutes by default —
    long enough to leave the room and come back, short enough that a day-long
    session does not accumulate stale identities."""
    max_entries: int = 256
    """Bound on gallery size. When full, the oldest identity is evicted."""
    embeddings_per_id: int = 8
    """Representative vectors kept per identity."""
    appearance_distance: float = 0.42
    """Strict cosine gate. Tighter than live re-association, because no motion
    gate constrains this decision.

    Measured on real WILDTRACK crops with the shipped OSNet embedder, this
    accepts ~20% of true cross-camera pairs at ~0.9% false accepts. Recall is
    deliberately sacrificed: failing to resurrect costs a new ID, whereas a
    wrong resurrection hands one person another person's identity."""
    top_k: int = 3
    """Top-k mean used for the query-to-identity distance."""
    ratio_test: float = 0.85
    """The best candidate must be at least this much better than the runner-up
    (best <= ratio * second_best). Applied only when two or more candidates pass
    the gate. Ambiguous evidence resurrects nothing."""
    duplicate_distance: float = 0.48
    """Entry-to-entry distance below which two *dormant identities* are judged to
    be the same person, and collapsed into one. ``0.0`` disables the check.

    This is the escape hatch for a self-deadlock the ratio test creates on its
    own. The ratio test asks "do two candidates fit comparably well?" and
    assumes that means *two different people, indistinguishable*. But one failed
    resurrection puts a **second copy of the same person** in the gallery, and
    from then on the two best candidates are always comparably close — because
    they are the same person — so every future return is rejected as ambiguous.
    One miss would otherwise disable long-gap re-ID permanently.

    The premise of the ratio test fails exactly when the two contenders are
    indistinguishable *from each other*, which is measurable directly. Defaults
    to the merge gate (0.48): the same decision — "are these two records one
    person?" — under the same irreversibility, so it gets the same threshold
    rather than a new tunable pulled from nowhere."""
    min_hits: int = 10
    """Only identities with this much accumulated evidence are worth storing;
    below it, a demotion would be preserving a false positive."""

    def __post_init__(self) -> None:
        if self.ttl_s <= 0.0:
            raise ValueError(f"ttl_s must be positive, got {self.ttl_s}")
        if self.max_entries < 1:
            raise ValueError(f"max_entries must be >= 1, got {self.max_entries}")
        if self.embeddings_per_id < 1:
            raise ValueError(f"embeddings_per_id must be >= 1, got {self.embeddings_per_id}")
        if not 0.0 < self.appearance_distance <= 2.0:
            raise ValueError(
                f"appearance_distance must be in (0, 2], got {self.appearance_distance}"
            )
        if self.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k}")
        if not 0.0 < self.ratio_test <= 1.0:
            raise ValueError(
                f"ratio_test must be in (0, 1] — a value above 1 would accept a match "
                f"that is worse than its runner-up; got {self.ratio_test}"
            )
        if not 0.0 <= self.duplicate_distance <= 2.0:
            raise ValueError(
                f"duplicate_distance must be in [0, 2] (0 disables the check), "
                f"got {self.duplicate_distance}"
            )
        if self.min_hits < 1:
            raise ValueError(f"min_hits must be >= 1, got {self.min_hits}")


class DormantGallery:
    """Retired identities, matchable by appearance alone."""

    def __init__(self, config: DormantConfig | None = None) -> None:
        self.config = config or DormantConfig()
        self._entries: dict[int, DormantEntry] = {}
        self.n_admitted = 0
        self.n_resurrected = 0
        self.n_expired = 0
        self.n_rejected_ambiguous = 0
        self.n_collapsed = 0
        self.attempts: deque[MatchAttempt] = deque(maxlen=512)
        """Every probe this gallery has seen, accepted or not. Bounded, so a long
        session keeps the recent history rather than growing without limit."""

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, global_id: int) -> bool:
        return global_id in self._entries

    @property
    def ids(self) -> list[int]:
        return sorted(self._entries)

    def entry(self, global_id: int) -> DormantEntry:
        return self._entries[global_id]

    def admit(
        self,
        global_id: int,
        vectors: npt.ArrayLike,
        frame: int,
        hits: int,
        cameras_seen: tuple[str, ...] = (),
        last_world_xy: npt.ArrayLike | None = None,
    ) -> bool:
        """Demote a retiring identity into the gallery. Returns True if stored."""
        if not self.config.enabled:
            return False
        if hits < self.config.min_hits:
            logger.debug(
                "not demoting global id %d: %d hits < min_hits %d (likely a false positive)",
                global_id,
                hits,
                self.config.min_hits,
            )
            return False
        data = np.atleast_2d(np.asarray(vectors, dtype=np.float64))
        if data.size == 0:
            return False

        if len(self._entries) >= self.config.max_entries and global_id not in self._entries:
            oldest = min(self._entries.values(), key=lambda e: e.retired_frame)
            del self._entries[oldest.global_id]
            logger.debug("dormant gallery full — evicted global id %d", oldest.global_id)

        self._entries[global_id] = DormantEntry(
            global_id=global_id,
            embeddings=select_representative(data, self.config.embeddings_per_id),
            retired_frame=frame,
            hits=hits,
            cameras_seen=cameras_seen,
            last_world_xy=(
                np.full(2, np.nan)
                if last_world_xy is None
                else np.asarray(last_world_xy, dtype=np.float64).reshape(2)
            ),
        )
        self.n_admitted += 1
        logger.info(
            "frame %d: global id %d went dormant (%d hits, %d representative vectors)",
            frame,
            global_id,
            hits,
            self._entries[global_id].embeddings.shape[0],
        )
        return True

    def expire(self, frame: int, dt: float) -> list[int]:
        """Drop identities past their TTL. Returns the ids removed."""
        if not self._entries:
            return []
        ttl_frames = self.config.ttl_s / max(dt, 1e-9)
        stale = [
            gid
            for gid, entry in self._entries.items()
            if (frame - entry.retired_frame) > ttl_frames
        ]
        for gid in stale:
            del self._entries[gid]
            self.n_expired += 1
            logger.info(
                "frame %d: dormant global id %d expired (TTL %.0f s)",
                frame,
                gid,
                self.config.ttl_s,
            )
        return stale

    def entry_distance(self, a: int, b: int) -> float:
        """Symmetric top-k mean distance between two stored identities.

        Symmetric because the two entries hold different numbers of vectors
        spanning different viewpoints, so ``a -> b`` and ``b -> a`` are not the
        same number and neither alone is the honest one.
        """
        first, second = self._entries[a], self._entries[b]
        if first.embeddings.size == 0 or second.embeddings.size == 0:
            return 1.0
        forward = float(first.distance(second.embeddings, self.config.top_k).mean())
        backward = float(second.distance(first.embeddings, self.config.top_k).mean())
        return 0.5 * (forward + backward)

    def _collapse(self, keep: int, absorb: int, reason: str) -> None:
        """Fold one dormant identity into another. Irreversible."""
        kept, gone = self._entries[keep], self._entries[absorb]
        merged = np.vstack([kept.embeddings, gone.embeddings])
        self._entries[keep] = DormantEntry(
            global_id=keep,
            embeddings=select_representative(merged, self.config.embeddings_per_id),
            # The TTL should run from the most recent sighting of the person, not
            # from whichever copy happens to survive.
            retired_frame=max(kept.retired_frame, gone.retired_frame),
            # Both records are the same person's real evidence, so the support
            # adds. This also keeps the merged identity senior to anything that
            # might later contest it.
            hits=kept.hits + gone.hits,
            cameras_seen=tuple(sorted(set(kept.cameras_seen) | set(gone.cameras_seen))),
            last_world_xy=(
                gone.last_world_xy
                if gone.retired_frame >= kept.retired_frame
                else kept.last_world_xy
            ),
        )
        del self._entries[absorb]
        self.n_collapsed += 1
        logger.info(
            "dormant ids %d and %d collapsed into %d (%s): they are the same "
            "person stored twice, %d + %d hits",
            keep,
            absorb,
            keep,
            reason,
            kept.hits,
            gone.hits,
        )

    def _collapse_contested_duplicates(self, cost: FloatArray, ids: list[int]) -> bool:
        """Collapse contenders that are indistinguishable *from each other*.

        The ratio test's premise is "two different identities fit comparably
        well, so the evidence cannot pick one". When the two contenders are
        themselves within the duplicate distance, that premise is false: there is
        no identity at risk, because there is only one identity. Resolving it
        here is what stops a single missed resurrection from deadlocking the
        gallery forever.

        Returns True if the gallery changed and the costs must be recomputed.
        """
        if self.config.duplicate_distance <= 0.0 or cost.shape[1] < 2:
            return False

        changed = False
        for row in range(cost.shape[0]):
            order = np.argsort(cost[row])
            best_col, second_col = int(order[0]), int(order[1])
            best, second = float(cost[row, best_col]), float(cost[row, second_col])
            if not np.isfinite(second) or best <= self.config.ratio_test * second:
                continue  # not contested — the ratio test is satisfied
            # Only collapse when the query is evidence about *these* identities.
            # Two entries may well be one person, but a query that matches
            # neither has no standing to say so.
            #
            # The bar for *standing* is deliberately the duplicate distance, not
            # the resurrection gate. Collapsing two records is gallery hygiene,
            # not an identity claim about the query: it answers "are these two
            # entries one person?", which is the same question the duplicate
            # distance already governs. Requiring the stricter resurrection gate
            # here would have made this fix a no-op on the run that motivated it
            # — that query sat at 0.45, just *outside* the 0.42 gate, which is
            # precisely why the person was not recognised in the first place.
            # The gate still decides whether anything is resurrected; it just no
            # longer decides whether the gallery may be de-duplicated.
            if best > max(self.config.appearance_distance, self.config.duplicate_distance):
                continue
            first_id, second_id = ids[best_col], ids[second_col]
            if first_id not in self._entries or second_id not in self._entries:
                continue  # already collapsed by an earlier row this pass
            separation = self.entry_distance(first_id, second_id)
            if separation > self.config.duplicate_distance:
                continue  # genuinely two people who look alike — leave it ambiguous
            keep, absorb = self._seniority(first_id, second_id)
            self._collapse(
                keep,
                absorb,
                f"contested by one query at {best:.3f}/{second:.3f}, "
                f"mutual distance {separation:.3f} <= {self.config.duplicate_distance}",
            )
            changed = True
        return changed

    def _seniority(self, a: int, b: int) -> tuple[int, int]:
        """Order two identities as (keep, absorb).

        More accumulated evidence wins, then the lower — that is, older — global
        ID. Keeping the senior one is what makes a recovered identity the
        *original* one rather than whichever duplicate happened to be minted
        last; a fix that resurrects the newer copy would recover the ID switch
        without undoing it.
        """
        first, second = self._entries[a], self._entries[b]
        if (first.hits, -a) >= (second.hits, -b):
            return a, b
        return b, a

    def match(
        self, queries: npt.ArrayLike, contexts: list[str] | None = None
    ) -> list[tuple[int, int, float]]:
        """Match query embeddings to dormant identities, one-to-one.

        Every probe is recorded in :attr:`attempts` and logged with its actual
        distances, including the ones that fail — a silent rejection here is
        indistinguishable downstream from "nobody returned".

        Args:
            queries: (N, D) L2-normalised embeddings, one per candidate new track.
            contexts: optional per-query provenance strings, for the log only.

        Returns:
            [(query_index, global_id, distance)] for accepted resurrections.
        """
        if not self.config.enabled or not self._entries:
            return []
        query = np.atleast_2d(np.asarray(queries, dtype=np.float64))
        if query.size == 0:
            return []

        ids = self.ids
        cost = self._cost(query, ids)

        # Break any same-person deadlock before judging ambiguity, then re-cost
        # against the collapsed gallery.
        if self._collapse_contested_duplicates(cost, ids):
            ids = self.ids
            cost = self._cost(query, ids)

        # Ratio test, computed on the UNGATED row and applied before gating.
        # Ranking post-gate candidates is worse than useless: the true owner
        # sitting marginally outside the threshold becomes +inf and can never be
        # the runner-up, so a stranger who happens to fall inside is handed the
        # identity unopposed — exactly the confusable case the test exists for.
        if cost.shape[1] >= 2:
            order = np.sort(cost, axis=1)
            best, second = order[:, 0], order[:, 1]
            ambiguous = np.isfinite(second) & (best > self.config.ratio_test * second)
        else:
            ambiguous = np.zeros(cost.shape[0], dtype=bool)

        gated = (cost > self.config.appearance_distance) | ambiguous[:, None]
        matrix = np.where(gated, INFEASIBLE, cost)
        pairs, _unmatched_rows, _unmatched_cols = linear_assignment(
            matrix, self.config.appearance_distance
        )

        # A global one-to-one solve can hand a query its runner-up because its
        # best match was cheaper for someone else. Two people returning together
        # would swap identities. Reject any pair materially worse than that
        # query's own best option.
        outcomes: dict[int, str] = {}
        accepted: list[tuple[int, int, float]] = []
        for row, col in pairs:
            row_best = float(cost[row].min())
            if float(cost[row, col]) > row_best / max(self.config.ratio_test, 1e-9):
                outcomes[row] = REJECTED_ASSIGNMENT
                self.n_rejected_ambiguous += 1
                continue
            outcomes[row] = ACCEPTED
            accepted.append((row, ids[col], float(cost[row, col])))

        for row in range(cost.shape[0]):
            if row in outcomes:
                continue
            if ambiguous[row]:
                outcomes[row] = REJECTED_AMBIGUOUS
                self.n_rejected_ambiguous += 1
            else:
                outcomes[row] = REJECTED_GATE

        self._record(cost, ids, outcomes, contexts)
        return accepted

    def _cost(self, query: FloatArray, ids: list[int]) -> FloatArray:
        """(n_queries, n_dormant) top-k mean distances."""
        return np.stack(
            [self._entries[gid].distance(query, self.config.top_k) for gid in ids], axis=1
        )

    def _record(
        self,
        cost: FloatArray,
        ids: list[int],
        outcomes: dict[int, str],
        contexts: list[str] | None,
    ) -> None:
        """Log and retain every probe, with the distances that decided it."""
        for row in range(cost.shape[0]):
            order = np.argsort(cost[row])
            attempt = MatchAttempt(
                query_index=row,
                ranked=tuple((ids[int(c)], float(cost[row, int(c)])) for c in order),
                outcome=outcomes[row],
                context=(
                    contexts[row] if contexts is not None and row < len(contexts) else ""
                ),
            )
            self.attempts.append(attempt)
            logger.info(
                "dormant probe | gate %.2f ratio %.2f | %s",
                self.config.appearance_distance,
                self.config.ratio_test,
                attempt.describe(),
            )

    def probe_report(self) -> list[str]:
        """Digest of every recorded probe — what the gate actually saw.

        Meant to be printed at the end of a session: a rejection is only
        actionable next to the distance that caused it.
        """
        if not self.attempts:
            return ["dormant probes: none (no candidate track ever queried the gallery)"]

        by_outcome: dict[str, list[float]] = {}
        for attempt in self.attempts:
            best = attempt.best
            if best is not None:
                by_outcome.setdefault(attempt.outcome, []).append(best[1])

        lines = [
            f"dormant probes: {len(self.attempts)} "
            f"(gate {self.config.appearance_distance:.2f}, "
            f"ratio {self.config.ratio_test:.2f}, "
            f"duplicate {self.config.duplicate_distance:.2f})"
        ]
        for outcome in sorted(by_outcome):
            values = np.asarray(by_outcome[outcome], dtype=np.float64)
            lines.append(
                f"  {outcome:22s} n={values.size:3d}  best-distance "
                f"min {values.min():.3f}  median {np.median(values):.3f}  "
                f"max {values.max():.3f}"
            )
        near = [
            distance
            for distance in by_outcome.get(REJECTED_GATE, [])
            if distance <= self.config.appearance_distance * 1.5
        ]
        if near:
            lines.append(
                f"  {len(near)} gate rejection(s) within 1.5x the gate — "
                f"closest {min(near):.3f} vs gate {self.config.appearance_distance:.2f}"
            )
        return lines

    def pop(self, global_id: int) -> DormantEntry:
        """Remove and return an identity being resurrected."""
        entry = self._entries.pop(global_id)
        self.n_resurrected += 1
        return entry
