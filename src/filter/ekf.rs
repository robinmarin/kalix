#![allow(non_snake_case)]
//! Extended Kalman filter.
//!
//! Recomputes the Jacobian of the dynamics at the current state estimate each
//! prediction step. Uses symbolic differentiation from the expression AST.

use nalgebra::{DMatrix, DVector};
use std::collections::HashMap;

use super::traits::{FilterState, KalmanFilter, StepResult, UpdateResult};
use crate::expr::ast::Expr;
use crate::expr::diff::diff;
use crate::expr::eval::eval_with_map;

/// Extended Kalman filter with symbolically-derived Jacobians.
pub struct EKF {
    /// Number of state variables.
    n: usize,
    /// Number of observation variables.
    m: usize,
    /// Dynamics expressions, one per state variable.
    dynamics: Vec<Expr>,
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
    /// State variable names (for building bindings during Jacobian eval).
    state_vars: Vec<String>,
    /// Pre-computed symbolic Jacobian: [(row_i_var_j) -> Expr]
    /// Each entry is the symbolic partial derivative of dynamics[i] w.r.t. state_vars[j].
    symbolic_jacobian: Vec<Vec<Expr>>,
}

impl EKF {
    /// Create a new EKF.
    ///
    /// `dynamics` are the parsed dynamics expressions, one per state variable.
    /// `state_vars` are the state variable names.
    /// `H` is the observation matrix.
    /// `Q` and `R` are noise covariance matrices.
    pub fn new(
        dynamics: Vec<Expr>,
        state_vars: Vec<String>,
        H: DMatrix<f64>,
        Q: DMatrix<f64>,
        R: DMatrix<f64>,
        x0: &[f64],
        P0: DMatrix<f64>,
    ) -> Self {
        let n = state_vars.len();
        let m = H.nrows();

        // Pre-compute symbolic Jacobian
        let symbolic_jacobian: Vec<Vec<Expr>> = dynamics
            .iter()
            .map(|dyn_expr| state_vars.iter().map(|var| diff(dyn_expr, var)).collect())
            .collect();

        let x = DVector::from_row_slice(x0);
        let I = DMatrix::identity(n, n);

        EKF {
            n,
            m,
            dynamics,
            H,
            Q,
            R,
            x,
            P: P0,
            I,
            state_vars,
            symbolic_jacobian,
        }
    }

    /// Evaluate the dynamics expressions at current state + dt to get next state prediction.
    fn evaluate_dynamics(&self, dt: f64) -> DVector<f64> {
        let mut bindings = HashMap::new();
        bindings.insert("dt", dt);
        for (i, var) in self.state_vars.iter().enumerate() {
            bindings.insert(var.as_str(), self.x[i]);
        }

        let next: Vec<f64> = self
            .dynamics
            .iter()
            .map(|expr| eval_with_map(expr, &bindings).unwrap_or(f64::NAN))
            .collect();

        DVector::from_row_slice(&next)
    }

    /// Evaluate the symbolic Jacobian at the given state and dt.
    fn evaluate_jacobian_at(&self, x: &[f64], dt: f64) -> DMatrix<f64> {
        let mut bindings = HashMap::new();
        bindings.insert("dt", dt);
        for (i, var) in self.state_vars.iter().enumerate() {
            bindings.insert(var.as_str(), x[i]);
        }

        let mut jac = DMatrix::zeros(self.n, self.n);
        for i in 0..self.n {
            for j in 0..self.n {
                jac[(i, j)] =
                    eval_with_map(&self.symbolic_jacobian[i][j], &bindings).unwrap_or(f64::NAN);
            }
        }
        jac
    }
}

impl KalmanFilter for EKF {
    fn predict(&mut self, dt: f64) -> FilterState {
        // Compute Jacobian at current state
        let F = self.evaluate_jacobian_at(self.x.as_slice(), dt);

        // Predict state using non-linear dynamics
        self.x = self.evaluate_dynamics(dt);

        // P = F * P * F^T + Q
        self.P = &F * &self.P * F.transpose() + &self.Q;

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

    fn jacobian(&self, x: &[f64], dt: f64) -> DMatrix<f64> {
        self.evaluate_jacobian_at(x, dt)
    }
}
