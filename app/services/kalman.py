"""Pure NumPy/SciPy Kalman filter for 8-dimensional bounding box state estimation.

State Space:
    x = [x, y, a, h, vx, vy, va, vh]^T
    where:
        (x, y): center bounding box coordinates
        a: aspect ratio (width / height)
        h: height
        (vx, vy, va, vh): respective instantaneous velocities

Measurement Space:
    z = [x, y, a, h]^T
"""

from typing import Tuple
import numpy as np
import scipy.linalg


class KalmanFilterTracker:
    """Kalman filter for tracking bounding boxes in image space."""

    def __init__(self) -> None:
        """Initialize constant velocity state transition and measurement matrices."""
        ndim = 4
        dt = 1.0

        # Motion matrix (8x8)
        self._motion_mat = np.eye(2 * ndim, 2 * ndim, dtype=np.float32)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        # Measurement projection matrix (4x8)
        self._update_mat = np.eye(ndim, 2 * ndim, dtype=np.float32)

        # Standard deviations weighting process and measurement noises
        self._std_weight_position = 1.0 / 20.0
        self._std_weight_velocity = 1.0 / 160.0

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create track from unassociated measurement.

        Args:
            measurement: Bounding box coordinates (x, y, a, h) of shape (4,).

        Returns:
            mean: Initial state vector of shape (8,).
            covariance: Initial state covariance matrix of shape (8, 8).
        """
        mean_pos = measurement.astype(np.float32)
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        h = measurement[3]
        std = [
            2.0 * self._std_weight_position * h,
            2.0 * self._std_weight_position * h,
            1e-2,
            2.0 * self._std_weight_position * h,
            10.0 * self._std_weight_velocity * h,
            10.0 * self._std_weight_velocity * h,
            1e-5,
            10.0 * self._std_weight_velocity * h,
        ]
        covariance = np.diag(np.square(std)).astype(np.float32)
        return mean, covariance

    def predict(
        self, mean: np.ndarray, covariance: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run Kalman filter prediction step.

        Args:
            mean: State vector of shape (8,).
            covariance: State covariance matrix of shape (8, 8).

        Returns:
            mean: Predicted state vector of shape (8,).
            covariance: Predicted covariance matrix of shape (8, 8).
        """
        h = mean[3]
        std_pos = [
            self._std_weight_position * h,
            self._std_weight_position * h,
            1e-2,
            self._std_weight_position * h,
        ]
        std_vel = [
            self._std_weight_velocity * h,
            self._std_weight_velocity * h,
            1e-5,
            self._std_weight_velocity * h,
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel])).astype(np.float32)

        # mean = F @ mean
        mean = np.dot(self._motion_mat, mean)
        # covariance = F @ covariance @ F^T + Q
        covariance = np.linalg.multi_dot(
            [self._motion_mat, covariance, self._motion_mat.T]
        ) + motion_cov

        return mean, covariance

    def project(
        self, mean: np.ndarray, covariance: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Project state distribution to measurement space.

        Args:
            mean: State vector of shape (8,).
            covariance: State covariance of shape (8, 8).

        Returns:
            projected_mean: Projected measurement mean of shape (4,).
            projected_covariance: Projected measurement covariance of shape (4, 4).
        """
        h = mean[3]
        std = [
            self._std_weight_position * h,
            self._std_weight_position * h,
            1e-1,
            self._std_weight_position * h,
        ]
        meas_cov = np.diag(np.square(std)).astype(np.float32)

        projected_mean = np.dot(self._update_mat, mean)
        projected_covariance = np.linalg.multi_dot(
            [self._update_mat, covariance, self._update_mat.T]
        ) + meas_cov
        return projected_mean, projected_covariance

    def update(
        self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run Kalman filter correction step using measurement.

        Args:
            mean: Predicted state vector of shape (8,).
            covariance: Predicted covariance matrix of shape (8, 8).
            measurement: Measurement vector (x, y, a, h) of shape (4,).

        Returns:
            new_mean: Corrected state vector of shape (8,).
            new_covariance: Corrected state covariance of shape (8, 8).
        """
        projected_mean, projected_cov = self.project(mean, covariance)

        # Cholesky factorization for stable inversion of projected_cov
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False
        )
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            np.dot(covariance, self._update_mat.T).T,
            check_finite=False,
        ).T

        innovation = measurement - projected_mean
        new_mean = mean + np.dot(kalman_gain, innovation)

        # Joseph form update to guarantee symmetry and positive semi-definiteness:
        # P = (I - K @ H) @ P @ (I - K @ H)^T + K @ R @ K^T
        h = mean[3]
        std = [
            self._std_weight_position * h,
            self._std_weight_position * h,
            1e-1,
            self._std_weight_position * h,
        ]
        meas_cov = np.diag(np.square(std)).astype(np.float32)

        identity = np.eye(8, dtype=np.float32)
        i_kh = identity - np.dot(kalman_gain, self._update_mat)
        new_covariance = np.linalg.multi_dot([i_kh, covariance, i_kh.T]) + np.linalg.multi_dot(
            [kalman_gain, meas_cov, kalman_gain.T]
        )

        return new_mean, new_covariance

    def gating_distance(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements: np.ndarray,
        only_position: bool = False,
    ) -> np.ndarray:
        """Compute squared Mahalanobis distance between state and measurements.

        Args:
            mean: State vector of shape (8,).
            covariance: State covariance of shape (8, 8).
            measurements: Matrix of measurements of shape (N, 4).
            only_position: If True, only considers (x, y) coordinates.

        Returns:
            Squared Mahalanobis distances of shape (N,).
        """
        mean, cov = self.project(mean, covariance)
        if only_position:
            mean, cov = mean[:2], cov[:2, :2]
            measurements = measurements[:, :2]

        cholesky_factor = np.linalg.cholesky(cov)
        d = measurements - mean
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True
        )
        squared_distances = np.sum(z * z, axis=0)
        return squared_distances

