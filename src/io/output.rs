#![allow(non_snake_case)]
//! Output message serialisation for live and backtest modes.
//!
//! Live mode emits compact JSON with named state and diagonal covariance.
//! Backtest mode emits full audit records with complete covariance matrices.

use crate::filter::traits::{FilterState, StepResult};
use serde::Serialize;

/// Ready event — emitted once after successful config load.
#[derive(Debug, Serialize)]
pub struct ReadyEvent {
    pub event: String,
    pub filter: String,
    pub variant: String,
    pub mode: String,
    pub state_variables: Vec<String>,
    pub observation_variables: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub F: Option<Vec<Vec<f64>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub H: Option<Vec<Vec<f64>>>,
}

/// Named state: maps variable names to values.
#[derive(Debug, Serialize)]
pub struct NamedState {
    #[serde(flatten)]
    pub values: serde_json::Map<String, serde_json::Value>,
}

/// Live-mode output — compact, one line per step.
#[derive(Debug, Serialize)]
pub struct LiveOutput {
    pub t: f64,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub predict_only: bool,
    pub x: NamedState,
    pub p_diag: Vec<f64>,
}

/// Backtest-mode output — full audit record per step.
#[derive(Debug, Serialize)]
pub struct BacktestOutput {
    pub t: f64,
    pub step: u64,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub predict_only: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub predict: Option<BacktestPredict>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub update: Option<BacktestUpdate>,
}

#[derive(Debug, Serialize)]
pub struct BacktestPredict {
    pub x: NamedState,
    pub P: Vec<Vec<f64>>,
}

#[derive(Debug, Serialize)]
pub struct BacktestUpdate {
    pub x: NamedState,
    pub P: Vec<Vec<f64>>,
    pub residual: NamedState,
    pub kalman_gain: Vec<Vec<f64>>,
    pub innovation_cov: Vec<Vec<f64>>,
}

/// Summary event — emitted at end of backtest mode.
#[derive(Debug, Serialize)]
pub struct SummaryEvent {
    pub event: String,
    pub steps: u64,
    pub predict_only_steps: u64,
    pub skipped_steps: u64,
    pub final_x: NamedState,
    pub final_p_diag: Vec<f64>,
}

/// Build a `NamedState` from variable names and values.
pub fn make_named_state(names: &[String], values: &[f64]) -> NamedState {
    let mut map = serde_json::Map::new();
    for (name, &val) in names.iter().zip(values.iter()) {
        map.insert(
            name.clone(),
            serde_json::Value::Number(
                serde_json::Number::from_f64(val)
                    .unwrap_or(serde_json::Number::from_f64(0.0).unwrap()),
            ),
        );
    }
    NamedState { values: map }
}

/// Convert a nalgebra DMatrix to Vec<Vec<f64>>.
pub fn matrix_to_vec(mat: &nalgebra::DMatrix<f64>) -> Vec<Vec<f64>> {
    (0..mat.nrows())
        .map(|i| (0..mat.ncols()).map(|j| mat[(i, j)]).collect())
        .collect()
}

/// Build a live-mode output line for a normal step.
pub fn build_live_output(
    t: f64,
    predict_only: bool,
    state_names: &[String],
    state: &[f64],
    P: &nalgebra::DMatrix<f64>,
) -> LiveOutput {
    let p_diag: Vec<f64> = (0..state.len()).map(|i| P[(i, i)]).collect();
    LiveOutput {
        t,
        predict_only,
        x: make_named_state(state_names, state),
        p_diag,
    }
}

/// Build a backtest-mode output line for a full step.
pub fn build_backtest_output(
    t: f64,
    step: u64,
    predict_only: bool,
    result: Option<&StepResult>,
    state_names: &[String],
    obs_names: &[String],
) -> BacktestOutput {
    let (predict, update) = if let Some(r) = result {
        let pred = BacktestPredict {
            x: make_named_state(state_names, &r.predicted.x),
            P: matrix_to_vec(&r.predicted.P),
        };
        let upd = BacktestUpdate {
            x: make_named_state(state_names, &r.update.updated.x),
            P: matrix_to_vec(&r.update.updated.P),
            residual: make_named_state(obs_names, &r.update.residual),
            kalman_gain: r.update.kalman_gain.clone(),
            innovation_cov: r.update.innovation_cov.clone(),
        };
        (Some(pred), if predict_only { None } else { Some(upd) })
    } else {
        (None, None)
    };

    BacktestOutput {
        t,
        step,
        predict_only,
        predict,
        update,
    }
}

/// Build a backtest output for a predict-only step.
pub fn build_backtest_predict_only(
    t: f64,
    step: u64,
    predicted: &FilterState,
    state_names: &[String],
) -> BacktestOutput {
    BacktestOutput {
        t,
        step,
        predict_only: true,
        predict: Some(BacktestPredict {
            x: make_named_state(state_names, &predicted.x),
            P: matrix_to_vec(&predicted.P),
        }),
        update: None,
    }
}

/// Build a summary event for backtest mode.
pub fn build_summary(
    steps: u64,
    predict_only_steps: u64,
    skipped_steps: u64,
    state_names: &[String],
    state: &[f64],
    P: &nalgebra::DMatrix<f64>,
) -> SummaryEvent {
    let p_diag: Vec<f64> = (0..state.len()).map(|i| P[(i, i)]).collect();
    SummaryEvent {
        event: "summary".to_string(),
        steps,
        predict_only_steps,
        skipped_steps,
        final_x: make_named_state(state_names, state),
        final_p_diag: p_diag,
    }
}

/// Build the ready event.
pub fn build_ready(
    filter_name: &str,
    variant: &str,
    mode: &str,
    state_vars: &[String],
    obs_vars: &[String],
    F: Option<&nalgebra::DMatrix<f64>>,
    H: Option<&nalgebra::DMatrix<f64>>,
) -> ReadyEvent {
    ReadyEvent {
        event: "ready".to_string(),
        filter: filter_name.to_string(),
        variant: variant.to_string(),
        mode: mode.to_string(),
        state_variables: state_vars.to_vec(),
        observation_variables: obs_vars.to_vec(),
        F: F.map(matrix_to_vec),
        H: H.map(matrix_to_vec),
    }
}
