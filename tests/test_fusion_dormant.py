"""Unit tests for the dormant (long-gap re-identification) gallery."""

from __future__ import annotations

import numpy as np
import pytest

from mcreid.fusion.dormant import (
    DormantConfig,
    DormantGallery,
    select_representative,
)

DIM = 32


def _unit(vec) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float64)
    return arr / np.linalg.norm(arr)


def _identity(seed: int, n: int = 6, spread: float = 0.15) -> np.ndarray:
    """n noisy observations of one identity."""
    rng = np.random.default_rng(seed)
    proto = _unit(rng.normal(size=DIM))
    out = proto + spread * rng.normal(size=(n, DIM)) / np.sqrt(DIM)
    return out / np.linalg.norm(out, axis=1, keepdims=True)


# --- select_representative ---------------------------------------------------------------


def test_select_representative_returns_everything_when_under_k():
    data = _identity(0, n=3)
    picked = select_representative(data, 8)
    assert picked.shape == (3, DIM)


def test_select_representative_caps_at_k():
    picked = select_representative(_identity(1, n=40), 8)
    assert picked.shape == (8, DIM)


def test_select_representative_covers_the_spread():
    """Farthest-point selection must beat 'nearest to the centroid' on coverage.

    The stored subset exists to match the person from any past viewpoint, so the
    worst-covered original vector is what matters, not the average.
    """
    data = _identity(2, n=40, spread=0.6)
    chosen = select_representative(data, 6)
    worst_farthest = (data @ chosen.T).max(axis=1).min()

    centroid = _unit(data.mean(axis=0))
    nearest = data[np.argsort(-(data @ centroid))[:6]]
    worst_nearest = (data @ nearest.T).max(axis=1).min()

    assert worst_farthest > worst_nearest, (
        f"farthest-point coverage {worst_farthest:.4f} should beat "
        f"nearest-to-centroid {worst_nearest:.4f}"
    )


def test_select_representative_rejects_bad_k():
    with pytest.raises(ValueError, match="k must be"):
        select_representative(_identity(3), 0)


# --- config validation -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"ttl_s": 0.0}, "ttl_s"),
        ({"max_entries": 0}, "max_entries"),
        ({"embeddings_per_id": 0}, "embeddings_per_id"),
        ({"appearance_distance": 0.0}, "appearance_distance"),
        ({"appearance_distance": 2.5}, "appearance_distance"),
        ({"top_k": 0}, "top_k"),
        ({"ratio_test": 1.5}, "ratio_test"),
        ({"ratio_test": 0.0}, "ratio_test"),
        ({"min_hits": 0}, "min_hits"),
        ({"near_miss_margin": -0.1}, "near_miss_margin"),
        ({"near_miss_margin": 1.5}, "near_miss_margin"),
    ],
)
def test_dormant_config_rejects_garbage(kwargs, match):
    with pytest.raises(ValueError, match=match):
        DormantConfig(**kwargs)


# --- admission ---------------------------------------------------------------------------


def test_admit_stores_and_matches():
    gallery = DormantGallery()
    person = _identity(10)
    assert gallery.admit(7, person, frame=100, hits=50)
    assert len(gallery) == 1
    assert 7 in gallery

    matches = gallery.match(_identity(10)[:1])
    assert matches and matches[0][1] == 7


def test_admit_refuses_low_evidence():
    """A track with barely any history is probably a false positive; storing it
    would let a hallucination reclaim an identity later."""
    gallery = DormantGallery(DormantConfig(min_hits=10))
    assert not gallery.admit(1, _identity(11), frame=0, hits=3)
    assert len(gallery) == 0


def test_admit_respects_disabled():
    gallery = DormantGallery(DormantConfig(enabled=False))
    assert not gallery.admit(1, _identity(12), frame=0, hits=100)
    assert gallery.match(_identity(12)) == []


def test_gallery_evicts_oldest_when_full():
    gallery = DormantGallery(DormantConfig(max_entries=2))
    for i in range(3):
        gallery.admit(i, _identity(20 + i), frame=i * 10, hits=50)
    assert len(gallery) == 2
    assert 0 not in gallery, "the oldest entry should have been evicted"


# --- expiry ------------------------------------------------------------------------------


def test_entries_expire_after_ttl():
    gallery = DormantGallery(DormantConfig(ttl_s=10.0))
    gallery.admit(1, _identity(30), frame=0, hits=50)
    dt = 1 / 30

    # The TTL is a DURATION, so drive the clock the way a session does: one
    # expire() per step, each advancing by dt. Asserting it from a single call
    # with a big frame number is what let the frame-based bug through.
    for frame in range(1, 201):  # 200 frames @ 1/30 s = 6.7 s
        assert gallery.expire(frame=frame, dt=dt) == []
    assert len(gallery) == 1, "6.7 s is still inside the 10 s TTL"

    removed: list[int] = []
    for frame in range(201, 401):  # to 13.3 s
        removed += gallery.expire(frame=frame, dt=dt)
    assert removed == [1], "past 10 s the identity must go"
    assert len(gallery) == 0


# --- matching ----------------------------------------------------------------------------


def test_different_identity_is_rejected():
    """The whole point: a stranger must not inherit a stored identity."""
    gallery = DormantGallery()
    gallery.admit(1, _identity(40), frame=0, hits=50)
    assert gallery.match(_identity(41)[:1]) == [], "a different identity must not match"


def _pair_at(separation: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two identities exactly ``separation`` apart, plus a query equidistant.

    Built analytically rather than sampled, because the ratio test turns on
    where these distances sit relative to a threshold, and a noisy draw that
    lands on the wrong side makes the test assert something other than what it
    says. Note the query ends up ``1 - sqrt((2 - separation) / 2)`` from both,
    which is much closer than ``separation`` — tests that need a specific query
    distance must build it explicitly rather than assume this one.
    """
    first, second_axis = np.zeros(DIM), np.zeros(DIM)
    first[0], second_axis[1] = 1.0, 1.0
    cosine = 1.0 - separation
    second = cosine * first + np.sqrt(1.0 - cosine**2) * second_axis
    query = _unit(first + second)  # equidistant from both by symmetry
    return first, second, query


def test_ratio_test_rejects_ambiguous_candidates():
    """Two stored identities that both fit equally well identify nobody."""
    first, second, query = _pair_at(separation=0.70)
    gallery = DormantGallery(DormantConfig(appearance_distance=1.5, ratio_test=0.85))
    gallery.admit(1, first[None, :], frame=0, hits=50)
    gallery.admit(2, second[None, :], frame=0, hits=50)

    assert gallery.match(query[None, :]) == [], "ambiguous evidence must resurrect nothing"
    assert gallery.n_rejected_ambiguous >= 1
    assert gallery.last_attempts[0].outcome == "rejected_ambiguous"
    assert len(gallery) == 2


# --- provenance: not storing a duplicate in the first place ------------------------------


def test_a_near_miss_duplicate_is_not_stored():
    """The cure for the deadlock: never hold two rival records of one person."""
    person = _identity(90)
    gallery = DormantGallery(DormantConfig(near_miss_margin=0.10))
    assert gallery.admit(1, person[:3], frame=0, hits=609)
    assert not gallery.admit(2, person[3:], frame=100, hits=40, same_as=1), (
        "an identity born from a near miss must not become a rival record"
    )
    assert gallery.ids == [1], "the original record is the one that survives"
    assert gallery.n_suppressed_duplicates == 1


def test_suppression_never_forgets_someone_for_nothing():
    """Defer only to a record that is actually still there to serve the return.

    If the older entry expired, was evicted, or was already resurrected, then
    withholding this one buys no de-duplication and loses the identity outright.
    """
    person = _identity(91)
    gallery = DormantGallery(DormantConfig(ttl_s=10.0, near_miss_margin=0.10))
    gallery.admit(1, person[:3], frame=0, hits=609)
    for frame in range(1, 401):  # 13.3 s of accumulated time: id 1 is gone
        gallery.expire(frame=frame, dt=1 / 30)
    assert gallery.ids == []

    assert gallery.admit(2, person[3:], frame=401, hits=40, same_as=1), (
        "with nothing to defer to, the identity must be stored normally"
    )
    assert gallery.ids == [2]
    assert gallery.n_suppressed_duplicates == 0


def test_suppression_ignores_a_self_or_unknown_reference():
    person = _identity(92)
    gallery = DormantGallery(DormantConfig(near_miss_margin=0.10))
    assert gallery.admit(1, person, frame=0, hits=50, same_as=99)  # never existed
    assert gallery.admit(2, person, frame=0, hits=50, same_as=2)  # itself
    assert gallery.ids == [1, 2]
    assert gallery.n_suppressed_duplicates == 0


def test_suppression_is_off_by_default():
    """Net-harmful with strangers present, so it must not ship enabled."""
    assert DormantConfig().near_miss_margin == 0.0
    person = _identity(93)
    gallery = DormantGallery()
    gallery.admit(1, person[:3], frame=0, hits=609)
    assert gallery.admit(2, person[3:], frame=100, hits=40, same_as=1), (
        "margin 0 must restore the old behaviour exactly"
    )
    assert gallery.ids == [1, 2]


def test_unlinked_lookalikes_are_left_alone():
    """The safety property, and the one an appearance-based fix broke.

    Two entries can be mutually close for the ordinary reason that two people
    look alike — measured on this project's real WILDTRACK crops, the
    ratio-contested top-2 pair is two *different* people 40.5% of the time. With
    no near-miss provenance saying otherwise, both records must be kept and an
    ambiguous probe must still resurrect nothing.
    """
    first, second, query = _pair_at(separation=0.30)  # very close, but unrelated
    gallery = DormantGallery(DormantConfig(appearance_distance=0.42))
    gallery.admit(1, first[None, :], frame=0, hits=609)
    gallery.admit(2, second[None, :], frame=100, hits=40)

    assert gallery.match(query[None, :]) == [], "similarity alone must decide nothing"
    assert gallery.n_suppressed_duplicates == 0
    assert len(gallery) == 2, "and nothing may be merged or deleted"


def test_suppression_makes_no_identity_claim_about_the_query():
    """Withholding storage must never *assign* an identity.

    The measured reason for this shape: a near-miss points at the wrong person
    45% of the time in a two-entry gallery. Trusted to assign, that hands people
    each other's IDs; trusted only to withhold, it costs a fresh ID next visit.
    """
    first, second, _q = _pair_at(separation=0.45)
    gallery = DormantGallery(
        DormantConfig(appearance_distance=0.42, near_miss_margin=0.10)
    )
    gallery.admit(1, first[None, :], frame=0, hits=609)
    gallery.admit(2, second[None, :], frame=100, hits=40, same_as=1)

    # id 2 was suppressed, so only id 1 is on record — and a query still has to
    # clear the gate on its own merits to get it.
    far = np.zeros(DIM)
    far[7] = 1.0
    assert gallery.ids == [1]
    assert gallery.match(far[None, :]) == [], "a suppression must not smuggle a match"


def test_pop_removes_and_counts():
    gallery = DormantGallery()
    gallery.admit(5, _identity(70), frame=0, hits=50)
    entry = gallery.pop(5)
    assert entry.global_id == 5
    assert len(gallery) == 0
    assert gallery.n_resurrected == 1


def test_empty_gallery_matches_nothing():
    assert DormantGallery().match(_identity(80)) == []


# --- retry policy: bounded re-probes after a gate rejection -------------------
#
# Shadow session s1 (reports/shadow_s1.csv, 39590 probe rows, 6 return episodes)
# recorded the dormant appearance distance every frame against frozen copies of
# every retired identity. Every one of the 8 f+0 gate rejections in that session
# came under the 0.42 gate by f+7. These tests pin the schedule that exploits it.


def _entry_vector(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return _unit(rng.normal(size=DIM))


def _query_at_distance(target: np.ndarray, distance: float, seed: int = 11) -> np.ndarray:
    """A unit query whose cosine distance to ``target`` is exactly ``distance``.

    Lets a recorded d(t) trace be replayed through the real gate instead of
    mocking the decision that is under test.
    """
    rng = np.random.default_rng(seed)
    perp = rng.normal(size=DIM)
    perp -= perp @ target * target
    perp = _unit(perp)
    sim = 1.0 - distance
    return _unit(sim * target + np.sqrt(max(1.0 - sim**2, 0.0)) * perp)


OWNER = 77  # the track that owns every retry in these tests
OWNERS = {0: OWNER}


def _gallery_with_one_entry(vec: np.ndarray, **cfg: object) -> DormantGallery:
    cfg.setdefault("retry_offsets", (4, 9))  # OFF by default; these tests exercise it
    gallery = DormantGallery(DormantConfig(**cfg))
    gallery.admit(1, vec[None, :], frame=0, hits=50)
    return gallery


def test_query_at_distance_helper_is_exact() -> None:
    """The trace replay is worthless if the harness cannot hit a distance."""
    vec = _entry_vector()
    for want in (0.386, 0.4276, 0.4381):
        got = 1.0 - float(_query_at_distance(vec, want) @ vec)
        assert got == pytest.approx(want, abs=1e-9)


def test_a_gate_rejection_schedules_exactly_two_reprobes() -> None:
    vec = _entry_vector()
    gallery = _gallery_with_one_entry(vec)
    assert gallery.match(_query_at_distance(vec, 0.4381)[None, :]) == []
    assert gallery.schedule_retries_owned(100, OWNER) == 1
    assert gallery.n_retries_scheduled == 1
    assert gallery.retries_due(103) == set(), "nothing is owed before f+4"
    assert gallery.retries_due(104) == {(1, OWNER)}, "first re-probe at f+4"
    assert gallery.retries_due(108) == set(), "and nothing again until f+9"
    assert gallery.retries_due(109) == {(1, OWNER)}, "second re-probe at f+9"
    assert gallery.retries_due(200) == set(), "two retries, then stop"


def test_an_accepted_probe_schedules_nothing() -> None:
    vec = _entry_vector()
    gallery = _gallery_with_one_entry(vec)
    assert gallery.match(_query_at_distance(vec, 0.20)[None, :]) != []
    assert gallery.schedule_retries_owned(100, OWNER) == 0
    assert gallery.retries_due(109) == set()


def test_a_rejected_retry_does_not_schedule_further_retries() -> None:
    """The bound has to hold, or 'two retries' becomes 'retry until it passes'.

    A retry that misses is still a gate rejection, so without the guard it would
    queue its own successors and the policy would drift into manufacturing a
    false accept by repetition.
    """
    vec = _entry_vector()
    gallery = _gallery_with_one_entry(vec)
    gallery.match(_query_at_distance(vec, 0.4381)[None, :])
    gallery.schedule_retries_owned(100, OWNER)
    gallery.retries_due(104)
    gallery.match(_query_at_distance(vec, 0.4276)[None, :])
    assert gallery.schedule_retries_owned(104, OWNER) == 0, "a retry rejection must not re-arm"
    gallery.retries_due(109)
    gallery.match(_query_at_distance(vec, 0.4300)[None, :])
    assert gallery.schedule_retries_owned(109, OWNER) == 0
    assert gallery.n_retries_scheduled == 1, "one schedule for this return, ever"


def test_retry_would_have_prevented_the_s1_rival_record() -> None:
    """THE regression, from the leak in shadow session s1.

    Track 2 appeared at frame 2401 and probed dormant record 1@1209 at d=0.4381
    — outside the 0.42 gate by 0.018. It was refused, went on to confirm under a
    new identity, and retired at frame 2863 as record 2@2863: a RIVAL RECORD of
    the person it had just failed to match. Two records of one person is what
    deadlocks the ratio test for every subsequent return.

    The recorded distances for that track against 1@1209 were:
        f+0  0.4381  rejected
        f+4  0.4276  still outside the gate
        f+9  0.3860  INSIDE — recoverable all along, just measured too early

    LIMITATION, stated plainly: this is a TRACE REPLAY, not a pipeline replay. It
    drives the real gate, the real ratio test and the real retry scheduler with
    the distances s1 actually recorded, and asserts the decision flips. It does
    NOT re-run detection, tracking or the track lifecycle — no replay harness
    exists for that, and the session was a live webcam with no clip retained. So
    this pins the decision, not the whole causal chain to the rival record.
    """
    vec = _entry_vector()
    gallery = _gallery_with_one_entry(vec)
    trace = {0: 0.4381, 4: 0.4276, 9: 0.3860}
    birth = 2401

    assert gallery.match(_query_at_distance(vec, trace[0])[None, :]) == [], (
        "f+0 must still be refused — the gate is unchanged at 0.42"
    )
    assert gallery.schedule_retries_owned(birth, OWNER) == 1

    assert gallery.retries_due(birth + 4) == {(1, OWNER)}
    assert gallery.match(_query_at_distance(vec, trace[4])[None, :]) == [], (
        "f+4 is genuinely outside the gate for this track; f+9 is what saves it"
    )

    assert gallery.retries_due(birth + 9) == {(1, OWNER)}
    recovered = gallery.match(_query_at_distance(vec, trace[9])[None, :])
    assert recovered == [(0, 1, pytest.approx(trace[9], abs=1e-6))], (
        "f+9 must recover identity 1, so no rival record is ever minted"
    )
    assert gallery.n_retries_fired == 2


def test_dropping_f9_loses_the_s1_recovery() -> None:
    """f+9 is load-bearing, not padding — the control for the test above."""
    vec = _entry_vector()
    gallery = _gallery_with_one_entry(vec, retry_offsets=(4,))
    gallery.match(_query_at_distance(vec, 0.4381)[None, :])
    gallery.schedule_retries_owned(2401, OWNER)
    assert gallery.retries_due(2405) == {(1, OWNER)}
    assert gallery.match(_query_at_distance(vec, 0.4276)[None, :]) == []
    assert gallery.retries_due(2410) == set(), "no second chance without f+9"
    assert len(gallery) == 1, "identity 1 is still stranded in the gallery"


def test_retry_is_not_conditioned_on_truncation() -> None:
    """s1 measured truncated crops as marginally CLOSER to their own record than
    clean ones (pooled gap -0.005), so truncation does not predict a bad query.
    The scheduler must therefore carry no truncation state at all."""
    import dataclasses

    fields = {f.name for f in dataclasses.fields(DormantConfig)}
    assert not any("trunc" in name for name in fields), (
        f"retry policy must not gate on truncation, found: {sorted(fields)}"
    )


def test_resurrection_cancels_a_pending_retry() -> None:
    vec = _entry_vector()
    gallery = _gallery_with_one_entry(vec)
    gallery.match(_query_at_distance(vec, 0.4381)[None, :])
    gallery.schedule_retries_owned(100, OWNER)
    gallery.pop(1)
    assert gallery.retries_due(104) == set(), "a recovered identity owes nothing"


def test_expiry_cancels_a_pending_retry() -> None:
    vec = _entry_vector()
    gallery = _gallery_with_one_entry(vec, ttl_s=1.0)
    gallery.match(_query_at_distance(vec, 0.4381)[None, :])
    gallery.schedule_retries_owned(100, OWNER)
    expired: list[int] = []
    for frame in range(1, 200):  # ttl_s=1.0, so ~30 frames @ 1/30 s suffices
        expired += gallery.expire(frame=frame, dt=1.0 / 30.0)
    assert expired == [1]
    assert gallery.retries_due(104) == set()


def test_no_behaviour_change_outside_the_probe_path() -> None:
    """Scopes what the retry actually changed.

    The retry IS a deliberate behaviour change on the probe path — that is the
    point of it — so a blanket "nothing changed" assertion would be false. What
    must still hold is that it changed *only* that path: with the schedule empty
    the gallery decides exactly as it did before, and every other mechanism keeps
    its shipped setting.
    """
    vec = _entry_vector()
    disabled = _gallery_with_one_entry(vec, retry_offsets=())

    assert disabled.match(_query_at_distance(vec, 0.4381)[None, :]) == []
    assert disabled.schedule_retries_owned(100, OWNER) == 0, "no schedule when the policy is empty"
    assert disabled.retries_due(104) == set()
    assert disabled.retries_due(109) == set()
    assert disabled.n_retries_scheduled == 0
    assert disabled.n_retries_fired == 0
    assert len(disabled) == 1, "and the identity stays stranded, exactly as before"

    # Untouched by this change, and each one was a measured decision.
    shipped = DormantConfig()
    assert shipped.appearance_distance == pytest.approx(0.42), "gate unchanged"
    assert shipped.ratio_test == pytest.approx(0.85), "ratio test unchanged"
    assert shipped.near_miss_margin == 0.0, "duplicate suppression still OFF by default"
    assert shipped.top_k == 3
    assert shipped.min_hits == 10


# --- TTL is a duration, not a frame budget -----------------------------------


def test_one_slow_frame_does_not_evict_a_young_entry() -> None:
    """THE s2 regression, with that session's measured numbers.

    The TTL used to be evaluated as ``frames_dormant > ttl_s / dt`` using the
    CURRENT frame's dt. That is not a duration: it re-derives a frame budget every
    step from whatever the last frame happened to cost, so one slow frame shrinks
    the budget for every entry at once.

    In s2 the loop ran at a 0.0475 s median (21 FPS) and frame 4220 took 0.6079 s
    — a 12.8x hiccup. On that one frame the budget collapsed from ~12,600 frames
    to 987, and an entry 1,631 frames old was evicted 58 s into a 600 s TTL. The
    person returned 30 frames later, matched the evicted entry at d=0.374-0.404
    (well inside the 0.42 gate) on all 7 probed frames, and was minted as a new
    identity because there was nothing left to match.
    """
    median_dt, hiccup_dt = 0.0475, 0.6079
    gallery = DormantGallery(DormantConfig(ttl_s=600.0))
    gallery.admit(1, _identity(70), frame=0, hits=50)

    for frame in range(1, 1632):  # 1631 frames at 21 FPS ~ 77 s
        assert gallery.expire(frame=frame, dt=median_dt) == []

    assert gallery.expire(frame=1632, dt=hiccup_dt) == [], (
        "a single 0.61 s frame must not evict an entry 77 s into a 600 s TTL"
    )
    assert 1 in gallery, "the identity the returning person needed must still be there"


def test_ttl_is_measured_in_accumulated_seconds_not_frames() -> None:
    """Same wall-clock elapsed must expire the same, at any frame rate.

    A frame-budget TTL makes a 60 FPS session hold identities for half as long in
    wall-clock terms as a 30 FPS one, for no stated reason.
    """
    elapsed: dict[float, int] = {}
    for dt in (1 / 15, 1 / 30, 1 / 60):
        gallery = DormantGallery(DormantConfig(ttl_s=10.0))
        gallery.admit(1, _identity(71), frame=0, hits=50)
        frame = 0
        while len(gallery):
            frame += 1
            gallery.expire(frame=frame, dt=dt)
            assert frame < 10_000, "expiry must terminate"
        elapsed[dt] = frame

    seconds = {dt: n * dt for dt, n in elapsed.items()}
    for dt, secs in seconds.items():
        assert secs == pytest.approx(10.0, abs=2 * dt), (
            f"at dt={dt:.4f} the identity lived {secs:.2f} s, expected ~10 s "
            f"(frames={elapsed[dt]})"
        )


def test_the_clock_advances_even_when_the_gallery_is_empty() -> None:
    """Otherwise time stops while nobody is dormant and every later TTL is short."""
    gallery = DormantGallery(DormantConfig(ttl_s=10.0))
    for frame in range(1, 301):  # 10 s of empty-gallery steps
        gallery.expire(frame=frame, dt=1 / 30)

    gallery.admit(1, _identity(72), frame=300, hits=50)
    for frame in range(301, 451):  # 5 s dormant
        assert gallery.expire(frame=frame, dt=1 / 30) == [], (
            "the entry is 5 s old; the 10 s already elapsed before it existed "
            "must not count against it"
        )
    assert 1 in gallery
