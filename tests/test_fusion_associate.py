"""Tests for mcreid.fusion.associate — appearance gallery, cost matrix, Hungarian solve."""

from __future__ import annotations

import numpy as np
import pytest

from mcreid.fusion.associate import (
    INFEASIBLE,
    AppearanceGallery,
    AssociationConfig,
    build_cost_matrix,
    linear_assignment,
)

# --- AppearanceGallery ------------------------------------------------------------


def test_gallery_distance_empty_gallery_returns_all_ones() -> None:
    gallery = AppearanceGallery()
    dist = gallery.distance(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    assert np.allclose(dist, [1.0, 1.0])


def test_gallery_distance_identical_embedding_is_zero() -> None:
    gallery = AppearanceGallery()
    gallery.add("cam0", np.array([1.0, 0.0, 0.0]))
    dist = gallery.distance(np.array([[1.0, 0.0, 0.0]]))
    assert dist[0] == pytest.approx(0.0, abs=1e-9)


def test_gallery_distance_orthogonal_embedding_is_one() -> None:
    gallery = AppearanceGallery()
    gallery.add("cam0", np.array([1.0, 0.0, 0.0]))
    dist = gallery.distance(np.array([[0.0, 1.0, 0.0]]))
    assert dist[0] == pytest.approx(1.0, abs=1e-9)


def test_gallery_add_normalises_embeddings() -> None:
    gallery = AppearanceGallery()
    gallery.add("cam0", np.array([3.0, 0.0, 0.0]))  # not unit norm
    dist = gallery.distance(np.array([[1.0, 0.0, 0.0]]))
    assert dist[0] == pytest.approx(0.0, abs=1e-9)


def test_gallery_add_rejects_zero_embedding() -> None:
    gallery = AppearanceGallery()
    with pytest.raises(ValueError, match="zero embedding"):
        gallery.add("cam0", np.zeros(4))


def test_gallery_rejects_bad_constructor_params() -> None:
    with pytest.raises(ValueError, match="per_camera"):
        AppearanceGallery(per_camera=0)
    with pytest.raises(ValueError, match="ema_alpha"):
        AppearanceGallery(ema_alpha=1.0)
    with pytest.raises(ValueError, match="ema_alpha"):
        AppearanceGallery(ema_alpha=0.0)


def test_gallery_len_and_matrix_and_items() -> None:
    gallery = AppearanceGallery(per_camera=2)
    assert len(gallery) == 0
    assert gallery.matrix().shape == (0, 0)
    assert gallery.items() == []

    gallery.add("cam0", np.array([1.0, 0.0]))
    gallery.add("cam1", np.array([0.0, 1.0]))
    assert len(gallery) == 2
    # matrix stacks every stored vector plus the EMA.
    assert gallery.matrix().shape == (3, 2)
    assert len(gallery.items()) == 2
    assert gallery.cameras == ("cam0", "cam1")


def test_gallery_ring_buffer_respects_per_camera_cap() -> None:
    gallery = AppearanceGallery(per_camera=2)
    for i in range(5):
        vec = np.zeros(4)
        vec[i % 4] = 1.0
        gallery.add("cam0", vec)
    assert len(gallery) == 2, "per_camera=2 must cap the ring buffer at 2 entries"


def test_gallery_embedding_dim_mismatch_raises() -> None:
    gallery = AppearanceGallery()
    gallery.add("cam0", np.array([1.0, 0.0, 0.0]))
    with pytest.raises(ValueError, match="dim mismatch"):
        gallery.distance(np.array([[1.0, 0.0]]))


# --- AssociationConfig ----------------------------------------------------------------


def test_association_config_rejects_weights_not_summing_to_one() -> None:
    with pytest.raises(ValueError, match="weights must sum to 1"):
        AssociationConfig(weight_geometry=0.5, weight_appearance=0.6)


def test_association_config_rejects_non_positive_gate_params() -> None:
    with pytest.raises(ValueError, match="chi2_gate"):
        AssociationConfig(chi2_gate=0.0, weight_geometry=0.4, weight_appearance=0.6)


def test_association_config_default_is_valid() -> None:
    cfg = AssociationConfig()
    assert cfg.weight_geometry + cfg.weight_appearance == pytest.approx(1.0)


# --- build_cost_matrix -----------------------------------------------------------------


def test_build_cost_matrix_gates_on_mahalanobis_alone() -> None:
    cfg = AssociationConfig()
    maha = np.array([[cfg.chi2_gate + 1.0]])
    euclid = np.array([[0.1]])  # well within gate
    appearance = np.array([[0.05]])  # well within gate
    cost = build_cost_matrix(maha, euclid, appearance, cfg)
    assert cost[0, 0] == INFEASIBLE, "mahalanobis gate alone must make the cell infeasible"


def test_build_cost_matrix_gates_on_euclidean_alone() -> None:
    cfg = AssociationConfig()
    maha = np.array([[1.0]])
    euclid = np.array([[cfg.max_distance_m + 0.1]])
    appearance = np.array([[0.05]])
    cost = build_cost_matrix(maha, euclid, appearance, cfg)
    assert cost[0, 0] == INFEASIBLE, "euclidean gate alone must make the cell infeasible"


def test_build_cost_matrix_gates_on_appearance_alone() -> None:
    cfg = AssociationConfig()
    maha = np.array([[1.0]])
    euclid = np.array([[0.1]])
    appearance = np.array([[cfg.max_appearance_distance + 0.1]])
    cost = build_cost_matrix(maha, euclid, appearance, cfg)
    assert cost[0, 0] == INFEASIBLE, "appearance gate alone must make the cell infeasible"


def test_build_cost_matrix_blends_feasible_cells() -> None:
    cfg = AssociationConfig()
    maha = np.array([[1.0]])
    euclid = np.array([[0.1]])
    appearance = np.array([[0.1]])
    cost = build_cost_matrix(maha, euclid, appearance, cfg)
    expected_geo = min(1.0 / cfg.chi2_gate, 1.0)
    expected_app = min(0.1 / cfg.max_appearance_distance, 1.0)
    expected = cfg.weight_geometry * expected_geo + cfg.weight_appearance * expected_app
    assert cost[0, 0] == pytest.approx(expected)
    assert cost[0, 0] < INFEASIBLE


def test_build_cost_matrix_rejects_mismatched_shapes() -> None:
    cfg = AssociationConfig()
    with pytest.raises(ValueError, match="shapes disagree"):
        build_cost_matrix(np.zeros((2, 2)), np.zeros((2, 3)), np.zeros((2, 2)), cfg)


# --- linear_assignment -----------------------------------------------------------------


def test_linear_assignment_hand_built_cost_matrix() -> None:
    cost = np.array([[0.1, 0.9], [0.8, 0.2]])
    matches, unmatched_rows, unmatched_cols = linear_assignment(cost, max_cost=0.85)
    assert set(matches) == {(0, 0), (1, 1)}
    assert unmatched_rows == []
    assert unmatched_cols == []


def test_linear_assignment_respects_cost_ceiling() -> None:
    cost = np.array([[0.1, 0.9], [0.8, 0.2]])
    matches, unmatched_rows, unmatched_cols = linear_assignment(cost, max_cost=0.15)
    assert matches == [(0, 0)], "only the cell under the ceiling should be kept"
    assert unmatched_rows == [1]
    assert unmatched_cols == [1]


def test_linear_assignment_handles_empty_rows() -> None:
    cost = np.zeros((0, 3))
    matches, unmatched_rows, unmatched_cols = linear_assignment(cost, max_cost=1.0)
    assert matches == []
    assert unmatched_rows == []
    assert unmatched_cols == [0, 1, 2]


def test_linear_assignment_handles_empty_cols() -> None:
    cost = np.zeros((3, 0))
    matches, unmatched_rows, unmatched_cols = linear_assignment(cost, max_cost=1.0)
    assert matches == []
    assert unmatched_rows == [0, 1, 2]
    assert unmatched_cols == []


def test_linear_assignment_rejects_non_2d_cost() -> None:
    with pytest.raises(ValueError, match="2-D"):
        linear_assignment(np.zeros(3), max_cost=1.0)
