//! Kalix — Declarative Kalman filtering from dynamics expressions.
//!
//! Write the physics as symbolic expressions, and the system automatically derives
//! the F (transition) and H (observation) matrices, detects linearity, and selects
//! the appropriate filter variant (linear KF or EKF).
//!
//! # Example
//!
//! ```toml
//! # configs/trend_no_accel.toml
//! [filter]
//! name = "trend_no_accel"
//!
//! [state]
//! variables = ["pos", "vel"]
//!
//! [dynamics]
//! pos = "pos + vel*dt"
//! vel = "vel"
//!
//! [observation]
//! variables   = ["z"]
//! expressions = ["pos"]
//! ```
//!
//! ```rust,no_run
//! use kalix::config::{Config, Variant};
//!
//! let toml_str = std::fs::read_to_string("configs/trend_no_accel.toml").unwrap();
//! let config = Config::from_toml(&toml_str).unwrap();
//! assert_eq!(config.variant, Variant::Linear);
//! ```

pub mod config;
pub mod expr;
pub mod filter;
pub mod io;
pub mod log;
pub mod matrix;
