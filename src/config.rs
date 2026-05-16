#![allow(non_snake_case)]
//! TOML config parsing, validation, and matrix derivation.
//!
//! The config defines the filter's state variables, dynamics expressions,
//! observation model, noise parameters, and initial conditions. On load,
//! the system validates all constraints, derives F and H matrices, and
//! detects whether the system is linear or requires an EKF.

use serde::Deserialize;
use thiserror::Error;

use crate::expr::ast::Expr;
use crate::expr::diff::diff;
use crate::expr::eval::eval_with_map;
use crate::expr::parser;
use std::collections::HashMap;

/// Errors that can occur during config loading or validation.
#[derive(Error, Debug)]
pub enum ConfigError {
    #[error("TOML parse error: {0}")]
    TomlParse(#[from] toml::de::Error),

    #[error("expression parse error: {0}")]
    ExprParse(String),

    #[error("unknown state variable in dynamics: '{name}' is not in state.variables")]
    UnknownStateVariable { name: String },

    #[error("unknown variable '{name}' in expression — must be a state variable or 'dt'")]
    UnknownVariable { name: String },

    #[error("dynamics missing entry for state variable '{name}'")]
    MissingDynamicsEntry { name: String },

    #[error("too many dynamics entries: got {got}, expected {expected} (one per state variable)")]
    WrongDynamicsCount { got: usize, expected: usize },

    #[error("Q must be {expected}x{expected}, got {got_rows}x{got_cols}")]
    WrongQSize {
        expected: usize,
        got_rows: usize,
        got_cols: usize,
    },

    #[error("R must be {expected}x{expected}, got {got_rows}x{got_cols}")]
    WrongRSize {
        expected: usize,
        got_rows: usize,
        got_cols: usize,
    },

    #[error("Q diagonal entries must all be >= 0 with at least one > 0")]
    InvalidQDiag,

    #[error("R diagonal entries must all be >= 0 with at least one > 0")]
    InvalidRDiag,

    #[error("initial state vector must have {expected} elements, got {got}")]
    WrongInitialStateSize { expected: usize, got: usize },

    #[error("initial covariance must be {expected}x{expected}, got {got_rows}x{got_cols}")]
    WrongInitialCovSize {
        expected: usize,
        got_rows: usize,
        got_cols: usize,
    },

    #[error("observation variables count ({nvars}) must match expressions count ({nexprs})")]
    ObservationCountMismatch { nvars: usize, nexprs: usize },

    #[error("observation expression references 'dt' — not allowed")]
    ObservationReferencesDt,

    #[error("--input requires backtest mode")]
    InputRequiresBacktest,
}

/// The filter variant: linear KF or EKF.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    Linear,
    Ekf,
}

/// Raw TOML structure for deserialization.
#[derive(Debug, Deserialize)]
struct RawConfig {
    filter: RawFilter,
    state: RawState,
    dynamics: RawDynamics,
    observation: RawObservation,
    noise: RawNoise,
    initial: RawInitial,
}

#[derive(Debug, Deserialize)]
struct RawFilter {
    name: String,
    #[serde(default)]
    description: String,
}

#[derive(Debug, Deserialize)]
struct RawState {
    variables: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct RawDynamics {
    #[serde(flatten)]
    entries: HashMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct RawObservation {
    variables: Vec<String>,
    expressions: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct RawNoise {
    process: Vec<Vec<f64>>,
    measurement: Vec<Vec<f64>>,
}

#[derive(Debug, Deserialize)]
struct RawInitial {
    state: Vec<f64>,
    covariance: Vec<Vec<f64>>,
}

/// A fully validated and parsed filter configuration.
#[derive(Debug, Clone)]
pub struct Config {
    /// Filter name (from `[filter].name`).
    pub name: String,
    /// Filter description.
    pub description: String,
    /// State variable names in declaration order.
    pub state_variables: Vec<String>,
    /// Parsed dynamics expressions, one per state variable (in declaration order).
    pub dynamics: Vec<Expr>,
    /// Observation variable names.
    pub observation_variables: Vec<String>,
    /// Parsed observation expressions.
    pub observation_expressions: Vec<Expr>,
    /// Process noise covariance Q (n × n).
    pub Q: nalgebra::DMatrix<f64>,
    /// Measurement noise covariance R (m × m).
    pub R: nalgebra::DMatrix<f64>,
    /// Initial state vector.
    pub x0: Vec<f64>,
    /// Initial covariance matrix.
    pub P0: nalgebra::DMatrix<f64>,
    /// Detected variant.
    pub variant: Variant,
    /// Whether to halt on input error (vs skip).
    pub halt_on_error: bool,
}

impl Config {
    /// Parse and validate a config from a TOML string.
    pub fn from_toml(toml_str: &str) -> Result<Self, ConfigError> {
        let raw: RawConfig = toml::from_str(toml_str)?;
        Config::from_raw(raw)
    }

    fn from_raw(raw: RawConfig) -> Result<Self, ConfigError> {
        let n = raw.state.variables.len();
        let m = raw.observation.variables.len();

        // ── Validate dynamics entries ────────────────────────────────
        let dynamics_order: Vec<String> = raw.state.variables.clone();

        // Check that dynamics has exactly the right number of entries
        let dynamics_count = raw.dynamics.entries.len();
        if dynamics_count != n {
            return Err(ConfigError::WrongDynamicsCount {
                got: dynamics_count,
                expected: n,
            });
        }

        // Parse dynamics expressions in state variable order
        let mut dynamics: Vec<Expr> = Vec::with_capacity(n);
        for var_name in &dynamics_order {
            let expr_str = raw.dynamics.entries.get(var_name).ok_or_else(|| {
                ConfigError::MissingDynamicsEntry {
                    name: var_name.clone(),
                }
            })?;

            let expr = parser::parse(expr_str).map_err(ConfigError::ExprParse)?;

            // Validate: every variable in the expression must be a state variable or "dt"
            validate_variables(&expr, &raw.state.variables)?;

            dynamics.push(expr);
        }

        // ── Parse observation expressions ────────────────────────────
        if raw.observation.variables.len() != raw.observation.expressions.len() {
            return Err(ConfigError::ObservationCountMismatch {
                nvars: raw.observation.variables.len(),
                nexprs: raw.observation.expressions.len(),
            });
        }

        let mut observation_expressions: Vec<Expr> = Vec::with_capacity(m);
        for expr_str in &raw.observation.expressions {
            let expr = parser::parse(expr_str).map_err(ConfigError::ExprParse)?;
            // Observation expressions must not reference dt
            if references_dt(&expr) {
                return Err(ConfigError::ObservationReferencesDt);
            }
            // Must only reference state variables
            validate_observation_variables(&expr, &raw.state.variables)?;
            observation_expressions.push(expr);
        }

        // ── Validate noise matrices ──────────────────────────────────
        let q_rows = raw.noise.process.len();
        let q_cols = raw.noise.process.first().map(|r| r.len()).unwrap_or(0);
        if q_rows != n || q_cols != n {
            return Err(ConfigError::WrongQSize {
                expected: n,
                got_rows: q_rows,
                got_cols: q_cols,
            });
        }

        let r_rows = raw.noise.measurement.len();
        let r_cols = raw.noise.measurement.first().map(|r| r.len()).unwrap_or(0);
        if r_rows != m || r_cols != m {
            return Err(ConfigError::WrongRSize {
                expected: m,
                got_rows: r_rows,
                got_cols: r_cols,
            });
        }

        // Q and R diagonal entries must all be >= 0, with at least one > 0 each
        let q_diag_ok = (0..n).all(|i| raw.noise.process[i][i] >= 0.0)
            && (0..n).any(|i| raw.noise.process[i][i] > 0.0);
        if !q_diag_ok {
            return Err(ConfigError::InvalidQDiag);
        }

        let r_diag_ok = (0..m).all(|i| raw.noise.measurement[i][i] >= 0.0)
            && (0..m).any(|i| raw.noise.measurement[i][i] > 0.0);
        if !r_diag_ok {
            return Err(ConfigError::InvalidRDiag);
        }

        // ── Validate initial state and covariance ────────────────────
        if raw.initial.state.len() != n {
            return Err(ConfigError::WrongInitialStateSize {
                expected: n,
                got: raw.initial.state.len(),
            });
        }

        let p0_rows = raw.initial.covariance.len();
        let p0_cols = raw.initial.covariance.first().map(|r| r.len()).unwrap_or(0);
        if p0_rows != n || p0_cols != n {
            return Err(ConfigError::WrongInitialCovSize {
                expected: n,
                got_rows: p0_rows,
                got_cols: p0_cols,
            });
        }

        // ── Build matrices ───────────────────────────────────────────
        let Q = vec_to_dmatrix(&raw.noise.process, n, n);
        let R = vec_to_dmatrix(&raw.noise.measurement, m, m);
        let P0 = vec_to_dmatrix(&raw.initial.covariance, n, n);

        // ── Detect variant ───────────────────────────────────────────
        let variant = detect_variant_from_exprs(&dynamics, &raw.state.variables);

        Ok(Config {
            name: raw.filter.name,
            description: raw.filter.description,
            state_variables: raw.state.variables,
            dynamics,
            observation_variables: raw.observation.variables,
            observation_expressions,
            Q,
            R,
            x0: raw.initial.state,
            P0,
            variant,
            halt_on_error: false,
        })
    }

    /// Derive the F matrix at a given dt by evaluating symbolic partials.
    pub fn derive_F(&self, dt: f64) -> nalgebra::DMatrix<f64> {
        let n = self.state_variables.len();
        let mut F = nalgebra::DMatrix::zeros(n, n);

        let bindings: HashMap<&str, f64> = self
            .state_variables
            .iter()
            .map(|v| (v.as_str(), 0.0))
            .chain(std::iter::once(("dt", dt)))
            .collect();

        for i in 0..n {
            for j in 0..n {
                let partial = diff(&self.dynamics[i], &self.state_variables[j]);
                F[(i, j)] = eval_with_map(&partial, &bindings).unwrap_or(f64::NAN);
            }
        }
        F
    }

    /// Derive the H matrix from observation expressions.
    ///
    /// For linear observations, partials are evaluated at 0 bindings for state vars
    /// (since the derivative of a linear expression is constant).
    pub fn derive_H(&self) -> nalgebra::DMatrix<f64> {
        let n = self.state_variables.len();
        let m = self.observation_expressions.len();
        let mut H = nalgebra::DMatrix::zeros(m, n);

        let bindings: HashMap<&str, f64> = self
            .state_variables
            .iter()
            .map(|v| (v.as_str(), 0.0))
            .collect();

        for i in 0..m {
            for j in 0..n {
                let partial = diff(&self.observation_expressions[i], &self.state_variables[j]);
                H[(i, j)] = eval_with_map(&partial, &bindings).unwrap_or(f64::NAN);
            }
        }
        H
    }
}

/// Validate that every variable in an expression is either a state variable or "dt".
fn validate_variables(expr: &Expr, state_vars: &[String]) -> Result<(), ConfigError> {
    match expr {
        Expr::Lit(_) => Ok(()),
        Expr::Var(name) => {
            if name == "dt" || state_vars.contains(name) {
                Ok(())
            } else {
                Err(ConfigError::UnknownVariable { name: name.clone() })
            }
        }
        Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) | Expr::Div(a, b) => {
            validate_variables(a, state_vars)?;
            validate_variables(b, state_vars)?;
            Ok(())
        }
        Expr::Pow(a, _) => validate_variables(a, state_vars),
        Expr::Sin(a) | Expr::Cos(a) | Expr::Log(a) | Expr::Exp(a) => {
            validate_variables(a, state_vars)
        }
    }
}

/// Validate that an observation expression only references state variables (not dt).
fn validate_observation_variables(expr: &Expr, state_vars: &[String]) -> Result<(), ConfigError> {
    match expr {
        Expr::Lit(_) => Ok(()),
        Expr::Var(name) => {
            if state_vars.contains(name) {
                Ok(())
            } else {
                Err(ConfigError::UnknownVariable { name: name.clone() })
            }
        }
        Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) | Expr::Div(a, b) => {
            validate_observation_variables(a, state_vars)?;
            validate_observation_variables(b, state_vars)?;
            Ok(())
        }
        Expr::Pow(a, _) => validate_observation_variables(a, state_vars),
        Expr::Sin(a) | Expr::Cos(a) | Expr::Log(a) | Expr::Exp(a) => {
            validate_observation_variables(a, state_vars)?;
            Ok(())
        }
    }
}

/// Check if an expression references `dt` anywhere.
fn references_dt(expr: &Expr) -> bool {
    match expr {
        Expr::Var(name) => name == "dt",
        Expr::Lit(_) => false,
        Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) | Expr::Div(a, b) => {
            references_dt(a) || references_dt(b)
        }
        Expr::Pow(a, _) => references_dt(a),
        Expr::Sin(a) | Expr::Cos(a) | Expr::Log(a) | Expr::Exp(a) => references_dt(a),
    }
}

/// Detect variant by checking if any partial derivative still contains a state variable.
fn detect_variant_from_exprs(dynamics: &[Expr], state_vars: &[String]) -> Variant {
    for expr in dynamics {
        for var in state_vars {
            let partial = diff(expr, var);
            if contains_state_var(&partial, state_vars) {
                return Variant::Ekf;
            }
        }
    }
    Variant::Linear
}

/// Check if a simplified expression still contains a state variable (not "dt").
fn contains_state_var(expr: &Expr, state_vars: &[String]) -> bool {
    match expr {
        Expr::Var(name) => state_vars.contains(name),
        Expr::Lit(_) => false,
        Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) | Expr::Div(a, b) => {
            contains_state_var(a, state_vars) || contains_state_var(b, state_vars)
        }
        Expr::Pow(a, _) => contains_state_var(a, state_vars),
        Expr::Sin(a) | Expr::Cos(a) | Expr::Log(a) | Expr::Exp(a) => {
            contains_state_var(a, state_vars)
        }
    }
}

/// Convert a `Vec<Vec<f64>>` to a nalgebra `DMatrix`.
fn vec_to_dmatrix(data: &[Vec<f64>], rows: usize, cols: usize) -> nalgebra::DMatrix<f64> {
    let mut mat = nalgebra::DMatrix::zeros(rows, cols);
    for i in 0..rows {
        for j in 0..cols {
            mat[(i, j)] = data[i][j];
        }
    }
    mat
}
