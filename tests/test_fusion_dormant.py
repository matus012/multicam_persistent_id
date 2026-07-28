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


def test_ratio_test_rejects_ambiguous_candidates():
    """Two stored identities that both fit equally well identify nobody."""
    gallery = DormantGallery(DormantConfig(appearance_distance=1.5, ratio_test=0.85))
    base = _identity(50, n=8)
    # Two near-duplicate identities: any query fits both about equally.
    gallery.admit(1, base, frame=0, hits=50)
    gallery.admit(2, base + 1e-6, frame=0, hits=50)

    assert gallery.match(base[:1]) == [], "ambiguous evidence must resurrect nothing"
    assert gallery.n_rejected_ambiguous >= 1


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
