"""Constant-velocity Kalman filter on the ground plane.

State is ``[x, y, vx, vy]`` in world metres / metres-per-second — deliberately
*not* P1's box-space filter, whose aspect-ratio state has no meaning once the
target is projected to the floor.

Coasting through a total occlusion is just repeated `predict()` with no
`update()`: the mean glides on constant velocity and the covariance grows, which
both keeps the BEV dot alive and automatically widens the re-association gate in
proportion to how long the target has been missing.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

# Chi-square 99% quantile, 2 degrees of freedom — the default gating threshold.
CHI2_2DOF_99 = 9.2103
CHI2_2DOF_95 = 5.9915


class GroundKalman:
    """4-state constant-velocity filter with per-measurement noise."""

    def __init__(
        self,
        process_noise: float = 1.5,
        initial_velocity_var: float = 4.0,
        max_speed_mps: float = 4.0,
    ) -> None:
        """
        Args:
            process_noise: white-noise acceleration density, m^2/s^3. A walking
                person accelerates at ~1-2 m/s^2; 1.5 keeps the filter responsive
                to direction changes without chasing detector jitter.
            initial_velocity_var: variance of the (unknown) initial velocity, m^2/s^2.
            max_speed_mps: hard clamp on the coasting velocity. Prevents a noisy
                birth from launching a coasted track across the room.
        """
        if process_noise <= 0.0:
            raise ValueError(f"process_noise must be positive, got {process_noise}")
        if max_speed_mps <= 0.0:
            raise ValueError(f"max_speed_mps must be positive, got {max_speed_mps}")
        self.process_noise = float(process_noise)
        self.initial_velocity_var = float(initial_velocity_var)
        self.max_speed_mps = float(max_speed_mps)
        self._H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)

    # --- matrices ---------------------------------------------------------

    @staticmethod
    def transition(dt: float) -> FloatArray:
        return np.array(
            [[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def process_covariance(self, dt: float) -> FloatArray:
        """Discretised white-noise-acceleration covariance."""
        q = self.process_noise
        dt2, dt3, dt4 = dt * dt, dt**3, dt**4
        block_pp = q * dt4 / 4.0
        block_pv = q * dt3 / 2.0
        block_vv = q * dt2
        return np.array(
            [
                [block_pp, 0.0, block_pv, 0.0],
                [0.0, block_pp, 0.0, block_pv],
                [block_pv, 0.0, block_vv, 0.0],
                [0.0, block_pv, 0.0, block_vv],
            ],
            dtype=np.float64,
        )

    # --- filter -----------------------------------------------------------

    def initiate(
        self, world_xy: npt.ArrayLike, world_cov: npt.ArrayLike
    ) -> tuple[FloatArray, FloatArray]:
        """Start a track from one ground measurement (velocity unknown)."""
        pos = np.asarray(world_xy, dtype=np.float64).reshape(2)
        pos_cov = np.asarray(world_cov, dtype=np.float64).reshape(2, 2)
        mean = np.array([pos[0], pos[1], 0.0, 0.0], dtype=np.float64)
        cov = np.zeros((4, 4), dtype=np.float64)
        cov[:2, :2] = pos_cov
        cov[2, 2] = cov[3, 3] = self.initial_velocity_var
        return mean, cov

    def predict(
        self, mean: FloatArray, cov: FloatArray, dt: float
    ) -> tuple[FloatArray, FloatArray]:
        """Constant-velocity prediction. ``dt`` in seconds."""
        if dt < 0.0:
            raise ValueError(f"dt must be non-negative, got {dt}")
        F = self.transition(dt)
        new_mean = F @ mean
        speed = float(np.linalg.norm(new_mean[2:]))
        if speed > self.max_speed_mps:
            new_mean[2:] *= self.max_speed_mps / speed
        new_cov = F @ cov @ F.T + self.process_covariance(dt)
        return new_mean, new_cov

    def project(
        self, mean: FloatArray, cov: FloatArray, measurement_cov: npt.ArrayLike
    ) -> tuple[FloatArray, FloatArray]:
        """Project state into measurement space. Returns (z_pred, innovation_cov S)."""
        R = np.asarray(measurement_cov, dtype=np.float64).reshape(2, 2)
        z_pred = self._H @ mean
        S = self._H @ cov @ self._H.T + R
        return z_pred, S

    def update(
        self,
        mean: FloatArray,
        cov: FloatArray,
        world_xy: npt.ArrayLike,
        measurement_cov: npt.ArrayLike,
    ) -> tuple[FloatArray, FloatArray]:
        """Standard Kalman update with a 2-D position measurement.

        Measurements from different cameras in the same frame are applied
        sequentially; given the state they are conditionally independent, so
        this is equivalent to a single stacked update.
        """
        z = np.asarray(world_xy, dtype=np.float64).reshape(2)
        z_pred, S = self.project(mean, cov, measurement_cov)
        kalman_gain = np.linalg.solve(S.T, (cov @ self._H.T).T).T
        innovation = z - z_pred
        new_mean = mean + kalman_gain @ innovation
        identity = np.eye(4, dtype=np.float64)
        factor = identity - kalman_gain @ self._H
        R = np.asarray(measurement_cov, dtype=np.float64).reshape(2, 2)
        # Joseph form — stays symmetric positive-definite under repeated updates.
        new_cov = factor @ cov @ factor.T + kalman_gain @ R @ kalman_gain.T
        return new_mean, 0.5 * (new_cov + new_cov.T)

    def mahalanobis_sq(
        self,
        mean: FloatArray,
        cov: FloatArray,
        measurements: npt.ArrayLike,
        measurement_covs: npt.ArrayLike,
    ) -> FloatArray:
        """Squared Mahalanobis distance from the predicted position to each of
        ``measurements`` (N, 2), each with its own (N, 2, 2) covariance."""
        z = np.asarray(measurements, dtype=np.float64).reshape(-1, 2)
        covs = np.asarray(measurement_covs, dtype=np.float64).reshape(-1, 2, 2)
        if z.shape[0] != covs.shape[0]:
            raise ValueError(f"count mismatch: {z.shape[0]} measurements, {covs.shape[0]} covs")

        z_pred = self._H @ mean
        base = self._H @ cov @ self._H.T
        out = np.empty(z.shape[0], dtype=np.float64)
        for i in range(z.shape[0]):
            S = base + covs[i]
            diff = z[i] - z_pred
            try:
                out[i] = float(diff @ np.linalg.solve(S, diff))
            except np.linalg.LinAlgError:  # pragma: no cover - singular S is pathological
                out[i] = np.inf
        return out
