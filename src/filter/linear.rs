#![allow(non_snake_case)]
//! Standard (linear) Kalman filter.
//!
//! Uses a fixed F (transition) and H (observation) matrix pre-computed from
//! the symbolic dynamics during configuration.

use super::traits::{FilterState, KalmanFilter, StepResult, UpdateResult};
use nalgebra::{DMatrix, DVector};

/// Standard linear Kalman filter with constant transition and observation matrices.
pub struct LinearKF {
    /// Number of state variables.
    n: usize,
    /// Number of observation variables.
    m: usize,
    /// Fixed transition matrix F (n × n).
    F: DMatrix<f64>,
    /// Fixed observation matrix H (m × n).
    H: DMatrix<f64>,
    /// Process noise covariance Q (n × n).
    Q: DMatrix<f64>,
    /// Measurement noise covariance R (m × m).
    R: DMatrix<f64>,
    /// Current state vector x (n × 1).
    x: DVector<f64>,
    /// Current covariance matrix P (n × n).
    P: DMatrix<f64>,
    /// Identity matrix (n × n), cached.
    I: DMatrix<f64>,
}

impl LinearKF {
    /// Create a new linear Kalman filter.
    pub fn new(
        F: DMatrix<f64>,
        H: DMatrix<f64>,
        Q: DMatrix<f64>,
        R: DMatrix<f64>,
        x0: &[f64],
        P0: DMatrix<f64>,
    ) -> Self {
        let n = F.nrows();
        let m = H.nrows();
        let x = DVector::from_row_slice(x0);
        let I = DMatrix::identity(n, n);
        LinearKF {
            n,
            m,
            F,
            H,
            Q,
            R,
            x,
            P: P0,
            I,
        }
    }
}

impl KalmanFilter for LinearKF {
    fn predict(&mut self, _dt: f64) -> FilterState {
        // x = F * x
        self.x = &self.F * &self.x;
        // P = F * P * F^T + Q
        self.P = &self.F * &self.P * self.F.transpose() + &self.Q;

        FilterState {
            x: self.x.as_slice().to_vec(),
            P: self.P.clone(),
        }
    }

    fn update(&mut self, z: &[f64]) -> UpdateResult {
        let z_vec = DVector::from_row_slice(z);

        // y = z - H * x  (innovation / residual)
        let y = &z_vec - &self.H * &self.x;

        // S = H * P * H^T + R  (innovation covariance)
        let S = &self.H * &self.P * self.H.transpose() + &self.R;

        // K = P * H^T * S^-1  (Kalman gain)
        let S_inv = S
            .clone()
            .try_inverse()
            .expect("innovation covariance matrix is not invertible");
        let K = &self.P * self.H.transpose() * &S_inv;

        // x = x + K * y
        self.x = &self.x + &K * &y;

        // P = (I - K * H) * P
        self.P = (&self.I - &K * &self.H) * &self.P;

        // Convert Kalman gain and innovation covariance to nested Vecs
        let kalman_gain: Vec<Vec<f64>> = (0..self.n)
            .map(|i| (0..self.m).map(|j| K[(i, j)]).collect())
            .collect();

        let innovation_cov: Vec<Vec<f64>> = (0..self.m)
            .map(|i| (0..self.m).map(|j| S[(i, j)]).collect())
            .collect();

        UpdateResult {
            updated: FilterState {
                x: self.x.as_slice().to_vec(),
                P: self.P.clone(),
            },
            residual: y.as_slice().to_vec(),
            kalman_gain,
            innovation_cov,
        }
    }

    fn step(&mut self, dt: f64, z: &[f64]) -> StepResult {
        let predicted = self.predict(dt);
        let update = self.update(z);
        StepResult { predicted, update }
    }

    fn predict_only(&mut self, dt: f64) -> FilterState {
        self.predict(dt)
    }

    fn state(&self) -> &[f64] {
        self.x.as_slice()
    }

    fn covariance(&self) -> &DMatrix<f64> {
        &self.P
    }

    fn jacobian(&self, _x: &[f64], _dt: f64) -> DMatrix<f64> {
        self.F.clone()
    }
}
