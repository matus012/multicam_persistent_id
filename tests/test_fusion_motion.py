"""Tests for mcreid.fusion.motion — the constant-velocity ground Kalman filter."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from mcreid.fusion.motion import GroundKalman

FloatArray = npt.NDArray[np.float64]

DT = 1.0 / 30.0


def _kf(**overrides: object) -> GroundKalman:
    kwargs: dict[str, object] = dict(process_noise=1.5, initial_velocity_var=4.0, max_speed_mps=4.0)
    kwargs.update(overrides)
    return GroundKalman(**kwargs)  # type: ignore[arg-type]


# --- constructor validation -------------------------------------------------------


def test_rejects_non_positive_process_noise() -> None:
    with pytest.raises(ValueError, match="process_noise"):
        _kf(process_noise=0.0)


def test_rejects_non_positive_max_speed() -> None:
    with pytest.raises(ValueError, match="max_speed_mps"):
        _kf(max_speed_mps=0.0)


# --- matrices ----------------------------------------------------------------------


def test_transition_matrix_applies_constant_velocity() -> None:
    F = GroundKalman.transition(DT)
    mean = np.array([1.0, 2.0, 0.5, -0.5])
    predicted = F @ mean
    assert np.allclose(predicted, [1.0 + 0.5 * DT, 2.0 - 0.5 * DT, 0.5, -0.5])


def test_process_covariance_is_symmetric_and_grows_with_dt() -> None:
    kf = _kf()
    small = kf.process_covariance(DT)
    large = kf.process_covariance(DT * 10)
    assert np.allclose(small, small.T)
    assert np.trace(large) > np.trace(small)


# --- initiate -----------------------------------------------------------------------


def test_initiate_sets_zero_velocity_and_position_covariance() -> None:
    kf = _kf()
    pos_cov = np.diag([0.05, 0.07])
    mean, cov = kf.initiate((1.0, 2.0), pos_cov)
    assert np.allclose(mean, [1.0, 2.0, 0.0, 0.0])
    assert np.allclose(cov[:2, :2], pos_cov)
    assert cov[2, 2] == pytest.approx(kf.initial_velocity_var)
    assert cov[3, 3] == pytest.approx(kf.initial_velocity_var)


# --- predict --------------------------------------------------------------------------


def test_predict_rejects_negative_dt() -> None:
    kf = _kf()
    mean, cov = kf.initiate((0.0, 0.0), np.eye(2) * 0.01)
    with pytest.raises(ValueError, match="dt must be non-negative"):
        kf.predict(mean, cov, -0.01)


def test_predict_clamps_velocity_to_max_speed() -> None:
    kf = _kf(max_speed_mps=4.0)
    mean = np.array([0.0, 0.0, 10.0, 10.0])  # speed ~14.1 m/s, way over the clamp
    cov = np.eye(4)
    new_mean, _ = kf.predict(mean, cov, DT)
    speed = float(np.linalg.norm(new_mean[2:]))
    assert speed == pytest.approx(kf.max_speed_mps, abs=1e-9)


def test_predict_covariance_grows_monotonically_with_no_updates() -> None:
    kf = _kf()
    mean, cov = kf.initiate((1.0, 1.0), np.eye(2) * 0.01)
    traces = []
    for _ in range(20):
        mean, cov = kf.predict(mean, cov, DT)
        traces.append(float(np.trace(cov)))
    assert all(
        traces[i] < traces[i + 1] for i in range(len(traces) - 1)
    ), f"covariance trace must grow monotonically under repeated predict(): {traces}"


# --- update: noiseless CV tracking -----------------------------------------------------


def test_update_tracks_noiseless_constant_velocity_trajectory() -> None:
    kf = _kf()
    true_vel = np.array([1.0, 0.5])
    pos0 = np.array([1.0, 1.0])
    tiny_r = np.eye(2) * 1e-8

    mean, cov = kf.initiate(pos0, np.eye(2) * 0.01)
    for k in range(1, 10):
        mean, cov = kf.predict(mean, cov, DT)
        true_pos = pos0 + true_vel * DT * k
        mean, cov = kf.update(mean, cov, true_pos, tiny_r)

    final_true_pos = pos0 + true_vel * DT * 9
    err = float(np.linalg.norm(mean[:2] - final_true_pos))
    assert err < 1e-6, f"noiseless CV trajectory should converge to <1e-6 m, got {err}"


def test_update_keeps_covariance_symmetric_positive_definite_over_many_steps() -> None:
    kf = _kf()
    rng = np.random.default_rng(0)
    pos0 = np.array([1.0, 1.0])
    mean, cov = kf.initiate(pos0, np.eye(2) * 0.05)
    r = np.eye(2) * 0.05

    for _ in range(200):
        mean, cov = kf.predict(mean, cov, DT)
        z = pos0 + rng.normal(scale=0.05, size=2)
        mean, cov = kf.update(mean, cov, z, r)

        assert np.allclose(cov, cov.T, atol=1e-10), "covariance must stay symmetric"
        eigenvalues = np.linalg.eigvalsh(cov)
        assert np.all(eigenvalues > 0), f"covariance must stay positive-definite: {eigenvalues}"


# --- project / mahalanobis_sq ------------------------------------------------------------


def test_project_returns_measurement_space_prediction() -> None:
    kf = _kf()
    mean, cov = kf.initiate((2.0, 3.0), np.eye(2) * 0.1)
    z_pred, S = kf.project(mean, cov, np.eye(2) * 0.2)
    assert np.allclose(z_pred, [2.0, 3.0])
    assert S.shape == (2, 2)
    assert np.allclose(S, S.T)


def test_mahalanobis_sq_zero_at_exact_prediction_and_grows_with_offset() -> None:
    kf = _kf()
    mean, cov = kf.initiate((0.0, 0.0), np.eye(2) * 0.1)
    covs = np.stack([np.eye(2) * 0.1, np.eye(2) * 0.1])
    measurements = np.array([[0.0, 0.0], [1.0, 1.0]])
    dist = kf.mahalanobis_sq(mean, cov, measurements, covs)
    assert dist[0] == pytest.approx(0.0, abs=1e-9)
    assert dist[1] > dist[0]


def test_mahalanobis_sq_rejects_count_mismatch() -> None:
    kf = _kf()
    mean, cov = kf.initiate((0.0, 0.0), np.eye(2) * 0.1)
    with pytest.raises(ValueError, match="count mismatch"):
        kf.mahalanobis_sq(mean, cov, np.zeros((2, 2)), np.stack([np.eye(2)]))
