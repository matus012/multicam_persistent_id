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
    retry_offsets: tuple[int, ...] = ()
    """Frames after a near-miss gate rejection at which the SAME track may
    re-probe the SAME dormant entry. ``()`` = OFF, which is the default.
    ``(4, 9)`` is the value to use with ``--single-occupant``.

    The idea: a returning person is measured against their own record at the
    worst possible moment — first frame seen, half in frame, mid-stride, against
    settled whole-body crops. The distance is not wrong, it is *early*.

    **OFF by default on measured evidence, not caution.** Adversarial review
    (session 3R) drove this against real WILDTRACK crops and the shipped OSNet
    weights, 650 owner-vs-stranger pairs per gallery shape. Extra probe slots
    bought **+2.0 to +3.4 points of identity theft for +0.0 points of owner
    recall**, because on those crops the owner is already inside the gate at f+0.
    That is the same trade that got PROVENANCE-SUPPRESS shipped OFF in 3g, and it
    is the wrong direction for a module whose whole premise is that a wrong
    resurrection costs more than a missed one.

    **The supporting evidence is also thinner than it first looked.** The "8/8
    recovered" figure came from the shadow log's raw distances, which ignores the
    ratio test — and the ratio test runs BEFORE the gate. Replayed through the
    real gallery (one live record per identity, not the shadow's frozen
    snapshots), four of the six f+0 probes are ``rejected_ambiguous``, which this
    policy deliberately does not arm on. Reachable evidence is ONE return
    episode, on the one gallery shape that has no ratio test at all.

    Safe to enable where the harm provably cannot occur: with a single occupant
    there is no stranger to steal an identity, so the failure mode is pure recall
    loss. Pool a second live session before considering it as a default.

    Scoping, which is what makes it safe at all: a schedule belongs to one
    (entry, track) pair, arms only on a NEAR miss (see ``retry_arm_margin``),
    fires once per pair per presence, and lets only that track adopt only that
    entry. An entry-keyed schedule — the first implementation — let any track in
    the scene spend the exemption, and review demonstrated a stranger taking a
    500-frame-old confirmed identity through it.

    NOT conditioned on truncation: s1 measured truncated crops as marginally
    *closer* to their own record than clean ones (pooled -0.005), so truncation
    does not predict a bad query."""
    retry_arm_margin: float = 0.15
    """How far past the gate a rejection may sit and still arm a retry, as a
    fraction of :attr:`appearance_distance`. 0.15 -> arm only within 0.483.

    A retry is for a query that was *early*, so arming has to require a near
    miss. Without a ceiling the first implementation armed on a probe at
    d=1.284 — nearly antipodal, unambiguously a different person — and every
    such arrival opened the gallery to every candidate track for two frames."""
    near_miss_margin: float = 0.0
    """Duplicate suppression for a **single-occupant** gallery. ``0.0`` = OFF,
    which is the default. ``0.10`` is the validated value when enabling it.

    What it addresses: the ratio test deadlocks itself. It asks "do two
    candidates fit comparably well?" and treats yes as "two different people,
    indistinguishable". But one failed resurrection stores a second copy of the
    same person under a new ID, and from then on the two best candidates are
    always comparably close — because they are the same person — so every later
    return is rejected as ambiguous. One miss disables long-gap re-ID for the
    rest of the session.

    When set, a track whose probe missed the gate by less than this margin is not
    stored as a rival record of the identity it nearly matched, so the pair that
    deadlocks the test never forms.

    **OFF by default because it is net-harmful with strangers present.** Measured
    over 2000 real leave/return cascades on WILDTRACK crops with the shipped
    OSNet, gallery = owner + 1 stranger: own identity recovered 41.1% -> 43.0%,
    but identity theft 6.2% -> 8.8%. It buys 1.9 points of recall for 2.6 points
    of theft, which is the wrong direction for this module — see
    :attr:`appearance_distance`. The cause is instructive: the suppressed
    duplicate was acting as a *protective competitor*, and removing it lets a
    stranger win a probe that the ratio test had been correctly refusing.

    **Safe, and worth enabling, when the gallery holds one person's records
    only** — the cardboard/acceptance scenario, one occupant in a room. There the
    measured harm cannot occur: suppression never *assigns* an identity, so with
    no stranger on file the only thing a probe can recover is the occupant, and
    the deadlock is pure recall loss. Two mechanisms that instead judged identity
    directly were tried first and were worse in every regime: merging two entries
    by their mutual distance fuses two genuinely different people 40.5% of the
    times it fires, and trusting a near-miss to say *which* identity a later
    ambiguous probe belongs to points at the wrong person 45% of the time even in
    a two-entry gallery.

    Provenance of those two rates: measured during development on real WILDTRACK
    crops with the shipped OSNet weights, over ~2000 simulated leave/return
    cascades. The harness was exploratory and is NOT in this repository, so the
    percentages are not reproducible from a clone — treat them as the recorded
    reason these mechanisms are off, not as a citable result. What IS reproducible
    is the behaviour they justify: see ``tests/test_fusion_dormant.py``, which
    pins the refusal and the deadlock it causes."""
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
        if not 0.0 <= self.near_miss_margin <= 1.0:
            raise ValueError(
                f"near_miss_margin must be in [0, 1] (0 disables provenance "
                f"linking), got {self.near_miss_margin}"
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
        self.n_rejected_assignment = 0
        """Counted separately from ambiguity: the one-to-one solve vetoing a pair
        and the ratio test refusing to choose are different failures, and a test
        asserting on a shared counter cannot tell which one fired."""
        self.n_suppressed_duplicates = 0
        self.attempts: deque[MatchAttempt] = deque(maxlen=512)
        """Every probe this gallery has seen, accepted or not. Bounded, so a long
        session keeps the recent history rather than growing without limit."""
        self.last_attempts: list[MatchAttempt] = []
        """Just the probes from the most recent :meth:`match` call, in query
        order, so the caller can act on rejections as well as acceptances."""
        self._retry_due: dict[tuple[int, int], list[int]] = {}
        """(dormant global_id, owning track id) -> pending re-probe frames."""
        self._retry_armed: set[tuple[int, int]] = set()
        """Every (entry, track) pair that has ever been armed, so a pair gets ONE
        schedule per presence and no more. A sliding time window is not a bound:
        it re-arms forever at a fixed period, which is exactly "keep asking until
        the gate lets something through". Cleared when the entry leaves the
        gallery or the track dies."""
        self.n_retries_scheduled = 0
        self.n_retries_fired = 0

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
        same_as: int | None = None,
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

        # Do not store a rival record of someone already in the gallery. Two
        # entries for one person is what deadlocks the ratio test permanently;
        # forgetting this one costs a fresh ID next visit and nothing else.
        defer_to = self.would_duplicate(global_id, same_as)
        if defer_to is not None:
            self.n_suppressed_duplicates += 1
            logger.info(
                "frame %d: NOT storing global id %d — it probed and missed id %d by "
                "a hair, so it is most likely the same person, and a second record "
                "would deadlock the ratio test for every future return",
                frame,
                global_id,
                defer_to,
            )
            return False

        if len(self._entries) >= self.config.max_entries and global_id not in self._entries:
            oldest = min(self._entries.values(), key=lambda e: e.retired_frame)
            del self._entries[oldest.global_id]
            # Eviction is a removal like any other. Skipping this strands retry
            # state that both blocks re-arming and outlives the entry it belongs to.
            self.cancel_retries(oldest.global_id)
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
            self.cancel_retries(gid)
            self.n_expired += 1
            logger.info(
                "frame %d: dormant global id %d expired (TTL %.0f s)",
                frame,
                gid,
                self.config.ttl_s,
            )
        return stale

    def schedule_retries(self, frame: int, owners: dict[int, int] | None = None) -> int:
        """Queue re-probes for entries the last :meth:`match` rejected on the gate.

        Call immediately after :meth:`match`. Reads :attr:`last_attempts`, so it
        schedules against the entry each query *nearly* matched — the best-ranked
        one on the ungated row — rather than against the gallery at large.

        Returns the number of schedules created. Idempotent within a schedule's
        own lifetime: an entry with retries still pending, or whose schedule
        ended less than ``max(retry_offsets)`` frames ago, is not rescheduled.
        Without that guard a rejected retry would schedule its own successors and
        the "two retries, then stop" bound would not hold.
        """
        offsets = self.config.retry_offsets
        if not offsets:
            return 0
        ceiling = self.config.appearance_distance * (1.0 + self.config.retry_arm_margin)
        created = 0
        for attempt in self.last_attempts:
            if attempt.outcome != REJECTED_GATE:
                continue
            best = attempt.best
            if best is None:
                continue
            gid, distance = best
            owner = owners.get(attempt.query_index) if owners else None
            if owner is None:
                # No identifiable owner means no way to scope the retry to the
                # track that earned it, and an unscoped retry is an open door:
                # it lets ANY track adopt on that frame. Refuse to arm instead.
                continue
            if gid not in self._entries:
                continue
            # A near miss is a query that was early. A miss by a mile is a
            # different person, and arming on one opens the gallery to every
            # candidate for two frames on the strength of no evidence at all.
            if distance > ceiling:
                continue
            key = (gid, owner)
            if key in self._retry_due or key in self._retry_armed:
                continue
            self._retry_due[key] = [frame + off for off in sorted(offsets)]
            self._retry_armed.add(key)
            self.n_retries_scheduled += 1
            created += 1
            logger.info(
                "frame %d: dormant id %d rejected on the gate at %.3f by track %d — "
                "re-probing at %s (that track only)",
                frame,
                gid,
                distance,
                owner,
                ", ".join(f"f+{off}" for off in sorted(offsets)),
            )
        return created

    def schedule_retries_owned(self, frame: int, owner: int = 0) -> int:
        """Convenience for a single-query probe: arm every query to ``owner``."""
        return self.schedule_retries(
            frame, owners={a.query_index: owner for a in self.last_attempts}
        )

    def retries_due(self, frame: int) -> set[tuple[int, int]]:
        """``(dormant_id, track_id)`` pairs owed a re-probe at or before ``frame``.

        Consumes what it returns. The pairing is the safety property: a retry is
        an exemption earned by ONE track against ONE identity, and returning bare
        ids would let any track in the scene spend it.
        """
        due: set[tuple[int, int]] = set()
        for key, pending in list(self._retry_due.items()):
            if not any(f <= frame for f in pending):
                continue
            remaining = [f for f in pending if f > frame]
            if remaining:
                self._retry_due[key] = remaining
            else:
                del self._retry_due[key]
            if key[0] in self._entries:
                due.add(key)
                self.n_retries_fired += 1
        return due

    def cancel_retries(self, global_id: int) -> None:
        """Forget every pending and spent re-probe for one identity.

        Called on every path that removes an entry — resurrection, expiry and
        capacity eviction. Missing one strands state that both blocks future
        arming and keeps a stale exemption alive.
        """
        for key in [k for k in self._retry_due if k[0] == global_id]:
            del self._retry_due[key]
        self._retry_armed = {k for k in self._retry_armed if k[0] != global_id}

    def forget_track(self, track_id: int) -> None:
        """Drop retry state for a track that has died, so its id can be reused."""
        for key in [k for k in self._retry_due if k[1] == track_id]:
            del self._retry_due[key]
        self._retry_armed = {k for k in self._retry_armed if k[1] != track_id}

    def would_duplicate(self, global_id: int, same_as: int | None) -> int | None:
        """The stored identity ``global_id`` would become a rival record of.

        Returns the identity to defer to, or None if this one should be stored
        normally. Deferring requires the older record to still be *present*: if
        it expired, was evicted, or was already resurrected, suppressing this
        identity would forget the person for no benefit at all.
        """
        if same_as is None or self.config.near_miss_margin <= 0.0:
            return None
        if same_as == global_id or same_as not in self._entries:
            return None
        return same_as

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
                self.n_rejected_assignment += 1
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
        self.last_attempts = []
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
            self.last_attempts.append(attempt)
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
            f"near-miss margin {self.config.near_miss_margin:.2f})"
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
        self.cancel_retries(global_id)
        self.n_resurrected += 1
        return entry
