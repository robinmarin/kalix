#![allow(non_snake_case)]
//! Kalix CLI — Declarative Kalman filter bridge.
//!
//! Reads JSON input lines from stdin (or a file), runs the filter, and emits
//! JSON output lines to stdout. Two modes:
//! - **live**: low-latency streaming, minimal output
//! - **backtest**: full audit trail with complete covariance matrices

use anyhow::{Context, Result};
use clap::Parser;
use std::io::{BufRead, BufReader, Write};
use tracing::{debug, error, info};

use kalix::config::Config;
use kalix::filter::traits::KalmanFilter;
use kalix::filter::{ekf::EKF, linear::LinearKF};
use kalix::io::input::InputMessage;
use kalix::io::output;

/// Kalix — Declarative Kalman filter from symbolic dynamics.
#[derive(Parser, Debug)]
#[command(name = "kalix", version = "0.1.0")]
struct Cli {
    /// Path to TOML config file.
    #[arg(short, long)]
    config: String,

    /// Run mode: live (streaming) or backtest (full audit).
    #[arg(short, long, default_value = "live")]
    mode: String,

    /// Input file path (implies --mode backtest).
    #[arg(short, long)]
    input: Option<String>,

    /// Error policy: skip (default) or halt.
    #[arg(long, default_value = "skip")]
    on_error: String,
}

fn main() -> Result<()> {
    kalix::log::init();

    let cli = Cli::parse();

    // Validate mode
    let mode = cli.mode.to_lowercase();
    if mode != "live" && mode != "backtest" {
        anyhow::bail!("invalid mode '{}': must be 'live' or 'backtest'", mode);
    }

    // --input requires backtest mode (or implies it)
    let (mode, input_path) = if let Some(ref path) = cli.input {
        if mode == "live" {
            anyhow::bail!("--input requires backtest mode");
        }
        ("backtest".to_string(), Some(path.clone()))
    } else {
        (mode, None)
    };

    // Error policy
    let halt_on_error = match cli.on_error.to_lowercase().as_str() {
        "skip" => false,
        "halt" => true,
        other => anyhow::bail!("invalid --on-error '{}': must be 'skip' or 'halt'", other),
    };

    // Load config
    let toml_str = std::fs::read_to_string(&cli.config)
        .with_context(|| format!("failed to read config file: {}", cli.config))?;

    let config =
        Config::from_toml(&toml_str).map_err(|e| anyhow::anyhow!("config error: {}", e))?;

    let n = config.state_variables.len();
    let m = config.observation_variables.len();

    // Derive F and H
    let dt_sample = 1.0; // dt for initial matrix display
    let F = config.derive_F(dt_sample);
    let H = config.derive_H();

    let variant_str = match config.variant {
        kalix::config::Variant::Linear => "linear",
        kalix::config::Variant::Ekf => "ekf",
    };

    info!(
        message = "config loaded",
        filter = %config.name,
        variant = variant_str,
    );
    info!(
        message = "derived F",
        F = ?output::matrix_to_vec(&F),
    );
    info!(
        message = "derived H",
        H = ?output::matrix_to_vec(&H),
    );

    // Build filter
    let mut filter: Box<dyn KalmanFilter> = match config.variant {
        kalix::config::Variant::Linear => Box::new(LinearKF::new(
            F.clone(),
            H.clone(),
            config.Q.clone(),
            config.R.clone(),
            &config.x0,
            config.P0.clone(),
        )),
        kalix::config::Variant::Ekf => Box::new(EKF::new(
            config.dynamics.clone(),
            config.state_variables.clone(),
            H.clone(),
            config.Q.clone(),
            config.R.clone(),
            &config.x0,
            config.P0.clone(),
        )),
    };

    // Emit ready event to stdout
    let ready = output::build_ready(
        &config.name,
        variant_str,
        &mode,
        &config.state_variables,
        &config.observation_variables,
        if variant_str == "linear" {
            Some(&F)
        } else {
            None
        },
        Some(&H),
    );
    println!("{}", serde_json::to_string(&ready)?);

    // ── Main loop ──────────────────────────────────────────────────────
    let reader: Box<dyn BufRead> = if let Some(ref path) = input_path {
        let file = std::fs::File::open(path)
            .with_context(|| format!("failed to open input file: {}", path))?;
        Box::new(BufReader::new(file))
    } else {
        Box::new(BufReader::new(std::io::stdin()))
    };

    let mut step_count: u64 = 0;
    let mut predict_only_count: u64 = 0;
    let mut skipped_count: u64 = 0;

    for line_result in reader.lines() {
        let line = line_result?;
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }

        // Parse input
        let msg: InputMessage = match serde_json::from_str(&line) {
            Ok(m) => m,
            Err(e) => {
                let err_msg = format!("malformed input: {}", e);
                eprintln!("{}", err_msg);
                error!(message = "input error", error = %err_msg);
                if halt_on_error {
                    std::process::exit(1);
                }
                skipped_count += 1;
                continue;
            }
        };

        // Validate input
        if let Err(validation_err) = msg.validate(m) {
            let err_msg = validation_err.to_string();
            eprintln!("{}", err_msg);
            error!(message = "input error", error = %err_msg);
            if halt_on_error {
                std::process::exit(1);
            }
            skipped_count += 1;
            continue;
        }

        let is_predict_only = msg.z.is_none();

        match mode.as_str() {
            "live" => {
                if is_predict_only {
                    let predicted = filter.predict_only(msg.dt);
                    predict_only_count += 1;
                    let output = output::build_live_output(
                        msg.t,
                        true,
                        &config.state_variables,
                        &predicted.x,
                        &predicted.P,
                    );
                    println!("{}", serde_json::to_string(&output)?);
                } else {
                    let result = filter.step(msg.dt, msg.z.as_ref().unwrap());
                    step_count += 1;
                    debug!(
                        message = "predict",
                        step = step_count,
                        x_prior = ?result.predicted.x,
                        x_post = ?result.update.updated.x,
                    );
                    debug!(
                        message = "kalman gain",
                        K = ?result.update.kalman_gain,
                    );
                    debug!(
                        message = "update",
                        residual = ?result.update.residual,
                        p_diag_post = ?(0..n).map(|i| result.update.updated.P[(i,i)]).collect::<Vec<_>>(),
                    );
                    let output = output::build_live_output(
                        msg.t,
                        false,
                        &config.state_variables,
                        &result.update.updated.x,
                        &result.update.updated.P,
                    );
                    println!("{}", serde_json::to_string(&output)?);
                }
            }
            "backtest" => {
                if is_predict_only {
                    let predicted = filter.predict_only(msg.dt);
                    predict_only_count += 1;
                    let output = output::build_backtest_predict_only(
                        msg.t,
                        step_count + predict_only_count + skipped_count,
                        &predicted,
                        &config.state_variables,
                    );
                    println!("{}", serde_json::to_string(&output)?);
                } else {
                    step_count += 1;
                    let result = filter.step(msg.dt, msg.z.as_ref().unwrap());
                    let output = output::build_backtest_output(
                        msg.t,
                        step_count + predict_only_count + skipped_count,
                        false,
                        Some(&result),
                        &config.state_variables,
                        &config.observation_variables,
                    );
                    println!("{}", serde_json::to_string(&output)?);
                }
            }
            _ => unreachable!(),
        }

        // Flush stdout after each line for live streaming responsiveness
        std::io::stdout().flush()?;
    }

    // ── Emit summary for backtest mode ─────────────────────────────────
    if mode == "backtest" {
        let summary = output::build_summary(
            step_count,
            predict_only_count,
            skipped_count,
            &config.state_variables,
            filter.state(),
            filter.covariance(),
        );
        println!("{}", serde_json::to_string(&summary)?);
    }

    Ok(())
}
