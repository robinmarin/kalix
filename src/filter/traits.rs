#![allow(non_snake_case)]
//! Kalman filter trait and result types.

use nalgebra::DMatrix;

/// The result of a prediction step.
pub struct FilterState {
    /// State vector x (post-predict, pre-update).
    pub x: Vec<f64>,
    /// Covariance matrix P (post-predict, pre-update).
    pub P: DMatrix<f64>,
}

/// The result of an update (measurement incorporation) step.
pub struct UpdateResult {
    /// Updated filter state (post-update).
    pub updated: FilterState,
    /// Innovation (measurement residual): z - H * x_predicted.
    pub residual: Vec<f64>,
    /// Kalman gain K as row-major nested Vec.
    pub kalman_gain: Vec<Vec<f64>>,
    /// Innovation covariance S = H * P * H^T + R.
    pub innovation_cov: Vec<Vec<f64>>,
}

/// Combined result of a predict + update cycle.
pub struct StepResult {
    /// State after prediction, before measurement incorporation.
    pub predicted: FilterState,
    /// Result of incorporating the measurement.
    pub update: UpdateResult,
}

/// The Kalman filter interface implemented by both linear KF and EKF.
pub trait KalmanFilter {
    /// Advance state by `dt`. Returns predicted state before incorporating measurement.
    fn predict(&mut self, dt: f64) -> FilterState;

    /// Incorporate observation `z`. Must be called after `predict`.
    fn update(&mut self, z: &[f64]) -> UpdateResult;

    /// Convenience: predict + update in one call.
    fn step(&mut self, dt: f64, z: &[f64]) -> StepResult;

    /// Predict only — no measurement available (sensor dropout / missing bar).
    fn predict_only(&mut self, dt: f64) -> FilterState;

    /// Current state vector (post-update).
    fn state(&self) -> &[f64];

    /// Full covariance matrix (post-update).
    fn covariance(&self) -> &DMatrix<f64>;

    /// Jacobian of dynamics evaluated at the given state and `dt`.
    ///
    /// Linear KF returns the fixed F matrix regardless of state.
    /// EKF evaluates symbolically at the given point.
    fn jacobian(&self, x: &[f64], dt: f64) -> DMatrix<f64>;
}
