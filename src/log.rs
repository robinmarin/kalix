//! Structured JSON tracing setup via `tracing` + `tracing-subscriber`.
//!
//! All diagnostic output goes to stderr as structured JSON. stdout is the data
//! channel exclusively. Control verbosity with `RUST_LOG`.

use tracing_subscriber::EnvFilter;

/// Initialise the tracing subscriber for JSON-formatted stderr logging.
///
/// Respects `RUST_LOG` for verbosity control (default: `info`).
pub fn init() {
    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(env_filter)
        .json()
        .init();
}
