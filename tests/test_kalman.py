"""Unit tests for pure NumPy/SciPy Kalman Filter state estimation."""

import numpy as np
import pytest
from app.services.kalman import KalmanFilterTracker


def test_kalman_initiate() -> None:
    """Test Kalman filter state and covariance initialization."""
    kf = KalmanFilterTracker()
    meas = np.array([100.0, 200.0, 0.5, 80.0], dtype=np.float32)

    mean, cov = kf.initiate(meas)

    assert mean.shape == (8,)
    assert cov.shape == (8, 8)

    # Position components match measurement
    assert np.allclose(mean[:4], meas)
    # Velocity components are initialized to zero
    assert np.allclose(mean[4:], np.zeros(4))
    # Covariance matrix must be symmetric positive-definite
    assert np.all(np.linalg.eigvals(cov) > 0)


def test_kalman_predict_constant_velocity() -> None:
    """Test prediction step integrates velocity into position."""
    kf = KalmanFilterTracker()
    mean = np.array([100.0, 200.0, 0.5, 80.0, 5.0, -3.0, 0.0, 0.0], dtype=np.float32)
    cov = np.eye(8, dtype=np.float32)

    pred_mean, pred_cov = kf.predict(mean, cov)

    # x_new = 100 + 5 = 105
    assert pytest.approx(pred_mean[0], 0.01) == 105.0
    # y_new = 200 - 3 = 197
    assert pytest.approx(pred_mean[1], 0.01) == 197.0
    # Velocities should remain constant in prediction
    assert pytest.approx(pred_mean[4], 0.01) == 5.0
    assert pytest.approx(pred_mean[5], 0.01) == -3.0

    # Covariance uncertainty must increase after prediction due to process noise Q
    assert np.trace(pred_cov) > np.trace(cov)


def test_kalman_update_velocity_convergence() -> None:
    """Test that repeated measurements along a trajectory correctly estimate velocity."""
    kf = KalmanFilterTracker()
    # Initial object at x=100 moving with true velocity vx=4.0 px/frame
    true_vx = 4.0
    z0 = np.array([100.0, 150.0, 1.0, 50.0], dtype=np.float32)
    mean, cov = kf.initiate(z0)

    for i in range(1, 10):
        mean, cov = kf.predict(mean, cov)
        # Measurement at step i
        zi = np.array([100.0 + true_vx * i, 150.0, 1.0, 50.0], dtype=np.float32)
        mean, cov = kf.update(mean, cov, zi)

    # Estimated vx should closely converge to true_vx
    est_vx = mean[4]
    assert pytest.approx(est_vx, abs=0.5) == true_vx


def test_kalman_project_dimensions() -> None:
    """Test state projection to measurement space."""
    kf = KalmanFilterTracker()
    mean = np.array([50.0, 60.0, 0.8, 100.0, 1.0, 2.0, 0.0, 0.0], dtype=np.float32)
    cov = np.eye(8, dtype=np.float32)

    p_mean, p_cov = kf.project(mean, cov)

    assert p_mean.shape == (4,)
    assert p_cov.shape == (4, 4)
    assert np.allclose(p_mean, mean[:4])
    assert np.all(np.linalg.eigvals(p_cov) > 0)


def test_kalman_gating_distance() -> None:
    """Test squared Mahalanobis distance calculation."""
    kf = KalmanFilterTracker()
    mean = np.array([100.0, 100.0, 1.0, 50.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    cov = np.eye(8, dtype=np.float32)

    # Exact match measurement
    z_match = np.array([[100.0, 100.0, 1.0, 50.0]], dtype=np.float32)
    dist_zero = kf.gating_distance(mean, cov, z_match)
    assert float(dist_zero[0]) < 1e-3

    # Distant measurement
    z_far = np.array([[300.0, 300.0, 1.0, 50.0]], dtype=np.float32)
    dist_far = kf.gating_distance(mean, cov, z_far)
    assert float(dist_far[0]) > float(dist_zero[0])

