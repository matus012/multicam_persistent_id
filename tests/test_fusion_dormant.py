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
    gallery.expire(frame=400, dt=1 / 30)  # 13.3 s: id 1 is gone
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
