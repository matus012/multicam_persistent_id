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

    assert gallery.expire(frame=200, dt=dt) == [], "200 frames = 6.7 s, still inside the TTL"
    assert len(gallery) == 1
    assert gallery.expire(frame=400, dt=dt) == [1], "400 frames = 13.3 s, past the TTL"
    assert len(gallery) == 0


# --- matching ----------------------------------------------------------------------------


def test_different_identity_is_rejected():
    """The whole point: a stranger must not inherit a stored identity."""
    gallery = DormantGallery()
    gallery.admit(1, _identity(40), frame=0, hits=50)
    assert gallery.match(_identity(41)[:1]) == [], "a different identity must not match"


def _pair_at(separation: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two identities exactly ``separation`` apart, plus a query equidistant.

    Built analytically rather than sampled, because both the ratio test and the
    duplicate check turn on where these distances sit relative to two different
    thresholds, and a noisy draw that lands on the wrong side of either makes
    the test assert something other than what it says.
    """
    first, second_axis = np.zeros(DIM), np.zeros(DIM)
    first[0], second_axis[1] = 1.0, 1.0
    cosine = 1.0 - separation
    second = cosine * first + np.sqrt(1.0 - cosine**2) * second_axis
    query = _unit(first + second)  # equidistant from both by symmetry
    return first, second, query


def test_ratio_test_rejects_ambiguous_candidates():
    """Two *different* stored identities that both fit equally well identify nobody.

    They must be far enough apart to be genuinely two people — otherwise this
    asserts the duplicate-collapse path instead, which is the opposite outcome.
    """
    first, second, query = _pair_at(separation=0.70)
    gallery = DormantGallery(
        DormantConfig(appearance_distance=1.5, ratio_test=0.85, duplicate_distance=0.48)
    )
    gallery.admit(1, first[None, :], frame=0, hits=50)
    gallery.admit(2, second[None, :], frame=0, hits=50)

    assert gallery.match(query[None, :]) == [], "ambiguous evidence must resurrect nothing"
    assert gallery.n_rejected_ambiguous >= 1
    assert gallery.n_collapsed == 0, "0.70 apart is two people, not one stored twice"
    assert len(gallery) == 2


def test_contested_duplicates_collapse_into_the_senior_identity():
    """The ratio test's premise fails when the contenders are the same person."""
    first, second, query = _pair_at(separation=0.45)
    gallery = DormantGallery(
        DormantConfig(appearance_distance=0.42, ratio_test=0.85, duplicate_distance=0.48)
    )
    gallery.admit(1, first[None, :], frame=0, hits=609)
    gallery.admit(2, second[None, :], frame=100, hits=40)

    matches = gallery.match(query[None, :])

    assert [gid for _row, gid, _d in matches] == [1], (
        f"the original identity must be the one resurrected, got {matches}"
    )
    assert gallery.n_collapsed == 1
    # `match` reports; the caller pops. Two entries became one.
    assert len(gallery) == 1 and gallery.ids == [1]
    assert gallery.entry(1).hits == 649, "the absorbed copy's evidence carries over"


def test_collapse_keeps_the_higher_evidence_identity_regardless_of_id_order():
    """Seniority is accumulated evidence first — not whichever ID is smaller."""
    first, second, query = _pair_at(separation=0.45)
    gallery = DormantGallery(DormantConfig(appearance_distance=0.42))
    gallery.admit(7, first[None, :], frame=0, hits=12)
    gallery.admit(9, second[None, :], frame=50, hits=800)

    matches = gallery.match(query[None, :])
    assert [gid for _row, gid, _d in matches] == [9]


def test_collapse_can_be_disabled():
    """The escape hatch is configurable, and off means the deadlock stands."""
    first, second, query = _pair_at(separation=0.45)
    gallery = DormantGallery(
        DormantConfig(appearance_distance=0.42, duplicate_distance=0.0)
    )
    gallery.admit(1, first[None, :], frame=0, hits=609)
    gallery.admit(2, second[None, :], frame=100, hits=40)

    assert gallery.match(query[None, :]) == []
    assert gallery.n_collapsed == 0


def test_collapse_needs_the_query_to_match_something():
    """Two stored copies of a stranger must not be collapsed by an unrelated query.

    Entry-to-entry similarity alone is not standing to rewrite identities: the
    check only fires when a live candidate is actually contesting *these* two.
    """
    first, second, _query = _pair_at(separation=0.45)
    stranger = np.zeros(DIM)
    stranger[5] = 1.0  # orthogonal to both — distance 1.0 to each
    gallery = DormantGallery(DormantConfig(appearance_distance=0.42))
    gallery.admit(1, first[None, :], frame=0, hits=50)
    gallery.admit(2, second[None, :], frame=0, hits=50)

    assert gallery.match(stranger[None, :]) == []
    assert gallery.n_collapsed == 0, "no query evidence, no irreversible collapse"
    assert len(gallery) == 2


def test_gallery_is_deduplicated_even_when_the_query_misses_the_gate():
    """De-duplication is gallery hygiene, not an identity claim about the query.

    The run that motivated this fix probed at ~0.45 against a 0.42 gate. If the
    collapse required the query to clear the resurrection gate, it could never
    fire on exactly the distances that create the duplicate — the fix would be a
    no-op on its own bug report. So this probe resurrects nothing (correctly:
    0.45 is outside the gate) while still leaving a gallery that is no longer
    deadlocked.
    """
    first, second, midpoint = _pair_at(separation=0.45)
    third_axis = np.zeros(DIM)
    third_axis[2] = 1.0
    # Tilt out of the plane until the query sits 0.45 from both entries.
    cosine_to_midpoint = 0.55 / float(midpoint @ first)
    query = _unit(
        cosine_to_midpoint * midpoint + np.sqrt(1.0 - cosine_to_midpoint**2) * third_axis
    )

    gallery = DormantGallery(
        DormantConfig(appearance_distance=0.42, duplicate_distance=0.48)
    )
    gallery.admit(1, first[None, :], frame=0, hits=609)
    gallery.admit(2, second[None, :], frame=100, hits=40)

    assert gallery.match(query[None, :]) == [], "0.45 is outside the 0.42 gate"
    assert gallery.n_collapsed == 1, "but the duplicate must still be cleared"
    assert gallery.ids == [1]

    # The gallery is now clean, so a closer look recovers the original identity.
    assert [gid for _r, gid, _d in gallery.match(first[None, :])] == [1]


def test_one_failed_resurrection_does_not_deadlock_the_gallery():
    """The full cascade, at the gallery level.

    Leave once, fail to be recognised (so a second copy of the same person is
    stored under a new ID), leave again, come back: the *original* identity must
    be recovered. Before the duplicate check existed, the second return was
    rejected as ambiguous forever, because the two candidates it could not
    separate were both this person.
    """
    first, second, query = _pair_at(separation=0.45)
    config = DormantConfig(appearance_distance=0.42, ratio_test=0.85)

    # Visit 1 ends: the identity is stored.
    gallery = DormantGallery(config)
    gallery.admit(1, first[None, :], frame=0, hits=609)

    # Visit 2 begins: the gate misses by 0.03 and the person is minted afresh.
    assert gallery.match(second[None, :]) == [], "reproduces the observed 0.45 > 0.42 miss"
    assert gallery.attempts[-1].outcome == "rejected_gate"
    assert gallery.attempts[-1].ranked[0] == (1, pytest.approx(0.45, abs=1e-9))

    # Visit 2 ends: a duplicate of the same person joins the gallery.
    gallery.admit(2, second[None, :], frame=100, hits=40)

    # Visit 3: the deadlock would strike here.
    matches = gallery.match(query[None, :])
    assert [gid for _row, gid, _d in matches] == [1], (
        f"must recover the original identity, got {matches}"
    )
    assert gallery.n_collapsed == 1


def test_match_is_one_to_one():
    gallery = DormantGallery()
    gallery.admit(1, _identity(60), frame=0, hits=50)
    gallery.admit(2, _identity(61), frame=0, hits=50)

    queries = np.vstack([_identity(60)[:1], _identity(60)[1:2]])
    matches = gallery.match(queries)
    assigned = [gid for _row, gid, _d in matches]
    assert len(assigned) == len(set(assigned)), "one dormant id may serve at most one query"


def test_pop_removes_and_counts():
    gallery = DormantGallery()
    gallery.admit(5, _identity(70), frame=0, hits=50)
    entry = gallery.pop(5)
    assert entry.global_id == 5
    assert len(gallery) == 0
    assert gallery.n_resurrected == 1


def test_empty_gallery_matches_nothing():
    assert DormantGallery().match(_identity(80)) == []
