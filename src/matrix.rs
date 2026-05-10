#![allow(non_snake_case)]
//! Matrix construction from derived Jacobians.
//!
//! Provides helpers to derive the F (transition) and H (observation) matrices
//! from the symbolic dynamics and observation expressions in a `Config`.

use crate::config::Config;
use nalgebra::DMatrix;

/// Derive the F (transition) matrix by evaluating symbolic partial derivatives
/// of dynamics expressions at the given `dt` value.
///
/// For linear systems, all partials evaluate to constants; for nonlinear systems,
/// the Jacobian depends on the state. This function returns the F matrix at the
/// given `dt` with state variables bound to 0.
pub fn derive_F(config: &Config, dt: f64) -> DMatrix<f64> {
    config.derive_F(dt)
}

/// Derive the H (observation) matrix by differentiating observation expressions.
pub fn derive_H(config: &Config) -> DMatrix<f64> {
    config.derive_H()
}
