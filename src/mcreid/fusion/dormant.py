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

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from mcreid.fusion.associate import INFEASIBLE, linear_assignment
from mcreid.utils.logging import get_logger

logger = get_logger(__name__)

FloatArray = npt.NDArray[np.float64]


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
    appearance_distance: float = 0.26
    """Strict cosine gate. Tighter than live re-association, because no motion
    gate constrains this decision."""
    top_k: int = 3
    """Top-k mean used for the query-to-identity distance."""
    ratio_test: float = 0.85
    """The best candidate must be at least this much better than the runner-up
    (best <= ratio * second_best). Applied only when two or more candidates pass
    the gate. Ambiguous evidence resurrects nothing."""
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

    def match(self, queries: npt.ArrayLike) -> list[tuple[int, int, float]]:
        """Match query embeddings to dormant identities, one-to-one.

        Args:
            queries: (N, D) L2-normalised embeddings, one per candidate new track.

        Returns:
            [(query_index, global_id, distance)] for accepted resurrections.
        """
        if not self.config.enabled or not self._entries:
            return []
        query = np.atleast_2d(np.asarray(queries, dtype=np.float64))
        if query.size == 0:
            return []

        ids = self.ids
        cost = np.stack(
            [self._entries[gid].distance(query, self.config.top_k) for gid in ids], axis=1
        )  # (n_queries, n_dormant)

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

        if ambiguous.any():
            self.n_rejected_ambiguous += int(ambiguous.sum())
            logger.info(
                "dormant match rejected as ambiguous for %d query/queries "
                "(best and runner-up within the %.2f ratio)",
                int(ambiguous.sum()),
                self.config.ratio_test,
            )

        gated = (cost > self.config.appearance_distance) | ambiguous[:, None]
        matrix = np.where(gated, INFEASIBLE, cost)
        pairs, _unmatched_rows, _unmatched_cols = linear_assignment(
            matrix, self.config.appearance_distance
        )

        # A global one-to-one solve can hand a query its runner-up because its
        # best match was cheaper for someone else. Two people returning together
        # would swap identities. Reject any pair materially worse than that
        # query's own best option.
        accepted: list[tuple[int, int, float]] = []
        for row, col in pairs:
            row_best = float(cost[row].min())
            if float(cost[row, col]) > row_best / max(self.config.ratio_test, 1e-9):
                self.n_rejected_ambiguous += 1
                logger.info(
                    "dormant assignment rejected: query %d was given id %d at %.3f "
                    "but its own best option was %.3f",
                    row,
                    ids[col],
                    float(cost[row, col]),
                    row_best,
                )
                continue
            accepted.append((row, ids[col], float(cost[row, col])))
        return accepted

    def pop(self, global_id: int) -> DormantEntry:
        """Remove and return an identity being resurrected."""
        entry = self._entries.pop(global_id)
        self.n_resurrected += 1
        return entry
