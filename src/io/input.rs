//! Input message deserialisation.
//!
//! Each input line is a JSON object with `t` (timestamp), `dt` (timestep),
//! and `z` (observation vector or null for predict-only).

use serde::Deserialize;

/// A single input step read from stdin or an input file.
#[derive(Debug, Deserialize, Clone)]
pub struct InputMessage {
    /// Informational timestamp — not used by the filter.
    pub t: f64,
    /// Timestep duration (must be > 0 and finite).
    pub dt: f64,
    /// Observation vector, or `null` for predict-only.
    pub z: Option<Vec<f64>>,
}

/// Errors that can occur when validating an input message.
#[derive(Debug, Clone)]
pub enum InputError {
    InvalidDt { dt: f64, reason: String },
    WrongZLength { got: usize, expected: usize },
    MalformedInput { reason: String },
}

impl InputMessage {
    /// Validate the message against the expected observation dimension.
    ///
    /// Returns `Ok(())` if valid, or an `InputError` describing the problem.
    pub fn validate(&self, expected_m: usize) -> Result<(), InputError> {
        // Validate dt
        if !self.dt.is_finite() {
            return Err(InputError::InvalidDt {
                dt: self.dt,
                reason: "non-finite value".to_string(),
            });
        }
        if self.dt <= 0.0 {
            return Err(InputError::InvalidDt {
                dt: self.dt,
                reason: format!("must be positive, got {}", self.dt),
            });
        }

        // Validate z length if observation is present
        if let Some(ref z) = self.z {
            if z.len() != expected_m {
                return Err(InputError::WrongZLength {
                    got: z.len(),
                    expected: expected_m,
                });
            }
        }

        Ok(())
    }
}

impl std::fmt::Display for InputError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InputError::InvalidDt { dt: _, reason } => {
                write!(f, "invalid dt: {}", reason)
            }
            InputError::WrongZLength { got, expected } => {
                write!(f, "z has {} elements, expected {}", got, expected)
            }
            InputError::MalformedInput { reason } => {
                write!(f, "malformed input: {}", reason)
            }
        }
    }
}
