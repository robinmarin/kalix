# Kalix — Declarative Kalman Filter with Language-Agnostic CLI Bridge

## Mission

**Kalix** is a production-grade Kalman filter library in Rust with a CLI entry point
driven via stdin/stdout JSON — callable from any language that can spawn a process
(Python, TypeScript, Ruby, Go, shell, etc.). The crate name is `kalix` throughout —
in `Cargo.toml`, the binary name, and all documentation.

The filter is **declaratively configured via TOML** — the user writes dynamics as symbolic
expressions, and the system automatically derives the F (transition) and H (observation)
matrices, detects linearity, and selects the appropriate filter variant (linear KF or EKF).

The CLI supports two modes: **live** (low-latency streaming, minimal output) and
**backtest** (full audit trail, named state fields, complete covariance matrices). Both
modes read the same input format; output verbosity differs.

---

## Repository Layout

```
kalman/
├── Cargo.toml
├── README.md
├── src/
│   ├── main.rs               # CLI entry point
│   ├── lib.rs                # Public library surface
│   ├── config.rs             # TOML config parsing and validation
│   ├── expr/
│   │   ├── mod.rs
│   │   ├── ast.rs            # Expression AST nodes
│   │   ├── parser.rs         # String → AST parser
│   │   ├── diff.rs           # Symbolic differentiation
│   │   └── eval.rs           # AST evaluator (numeric, given variable bindings)
│   ├── filter/
│   │   ├── mod.rs
│   │   ├── linear.rs         # Standard Kalman filter
│   │   ├── ekf.rs            # Extended Kalman filter
│   │   └── traits.rs         # KalmanFilter trait
│   ├── io/
│   │   ├── mod.rs
│   │   ├── input.rs          # Input message deserialisation
│   │   └── output.rs         # Output message serialisation (live vs backtest)
│   ├── matrix.rs             # Matrix construction from derived Jacobians
│   └── log.rs                # Structured JSON tracing setup
├── configs/
│   ├── trend_no_accel.toml
│   ├── trend_with_accel.toml
│   ├── constant_velocity.toml
│   └── pendulum.toml
└── tests/
    ├── test_expr.rs
    ├── test_diff.rs
    ├── test_config.rs
    ├── test_linear_kf.rs
    └── test_ekf.rs
```

---

## Dependencies (`Cargo.toml`)

```toml
[package]
name        = "kalix"
version     = "0.2.1"
edition     = "2021"
description = "Declarative Kalman filtering from dynamics expressions. Write the physics, derive the filter."
license     = "MIT"

[[bin]]
name = "kalix"
path = "src/main.rs"

[dependencies]
nalgebra = { version = "0.33", features = ["serde-serialize"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
toml = "0.8"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["json", "env-filter"] }
anyhow = "1"
thiserror = "1"

[dev-dependencies]
approx = "0.5"
```

Do **not** use external expression or CAS crates. Implement the AST, parser, and symbolic
differentiator from scratch — the supported operation set is deliberately small.

---

## Expression Language

Expressions are polynomial in state variables and the special token `dt`. Supported:

| Syntax            | Meaning                                         |
| ----------------- | ----------------------------------------------- |
| `pos`, `vel`, `x` | State variable reference                        |
| `dt`              | Timestep — a runtime parameter, not a state var |
| `3.14`, `0.5`     | Numeric literal                                 |
| `a + b`           | Addition                                        |
| `a - b`           | Subtraction                                     |
| `a * b`           | Multiplication                                  |
| `a / b`           | Division — right-hand side must be a literal    |
| `a ^ 2`           | Integer power (positive integers only)          |
| `sin(expr)`       | Sine                                            |
| `cos(expr)`       | Cosine                                          |
| `log(expr)`       | Natural logarithm                               |
| `exp(expr)`       | Exponential (`e^expr`)                          |
| `(expr)`          | Grouping                                        |

No division by state variables — the RHS of `/` must be a numeric literal.
Unsupported functions return a descriptive parse error.

### AST Definition

```rust
pub enum Expr {
    Lit(f64),
    Var(String),                    // state variable name, or "dt"
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),      // right side must be Lit after parsing
    Pow(Box<Expr>, u32),
    Sin(Box<Expr>),
    Cos(Box<Expr>),
    Log(Box<Expr>),                 // natural log
    Exp(Box<Expr>),
}
```

### Symbolic Differentiation Rules

Implement `diff(expr: &Expr, var: &str) -> Expr`:

- `d/dx(lit) = 0`
- `d/dx(x) = 1`, `d/dx(y) = 0` for `y != x`
- `d/dx(dt) = 0` — dt is a parameter, not a state variable
- Standard sum, difference, product, and power rules
- Chain rule for transcendental functions:
  `d/dx(sin(u)) = cos(u) * u'`, `d/dx(cos(u)) = -sin(u) * u'`
  `d/dx(log(u)) = u' / u`, `d/dx(exp(u)) = exp(u) * u'`
- After differentiation, run a simplification pass:
  `0 + x -> x`, `x + 0 -> x`, `1 * x -> x`, `x * 1 -> x`,
  `0 * x -> 0`, `x * 0 -> 0`, `x ^ 1 -> x`, `x ^ 0 -> Lit(1)`

### Linearity Detection

After computing all partial derivatives `df_i/dx_j`, check whether any resulting
expression still contains a `Var` node whose name is a **state variable** (not `dt`).
If any partial is nonlinear -> EKF. Otherwise -> linear KF, F built once at startup.

---

## TOML Config Format

```toml
[filter]
name        = "trend_with_accel"
description = "Position/velocity/acceleration trend follower"

[state]
variables = ["pos", "vel", "acc"]

[dynamics]
# One entry per state variable — how it evolves each timestep
pos = "pos + vel*dt + 0.5*acc*dt^2"
vel = "vel + acc*dt"
acc = "acc"

[observation]
# Names and expressions for each measurement channel
variables   = ["z"]
expressions = ["pos"]    # z measures pos directly

[noise]
process     = [[0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]]   # Q — n x n
measurement = [[1.0]]                                         # R — m x m

[initial]
state      = [0.0, 0.0, 0.0]
covariance = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
```

### Config Validation (fail loudly at load time)

- `[dynamics]` must have exactly one entry per state variable, in the declared order
- Every variable in a dynamics expression must be in `[state].variables` or be `dt`
- Every observation expression must reference only state variables (not `dt`)
- F -> `n x n`, H -> `m x n`, Q -> `n x n`, R -> `m x m` — dimension mismatches are errors
- Q and R diagonal entries must all be >= 0, with at least one > 0 each

---

## KalmanFilter Trait

```rust
pub trait KalmanFilter {
    /// Advance state by dt. Returns predicted state before incorporating measurement.
    fn predict(&mut self, dt: f64) -> FilterState;

    /// Incorporate observation z. Must be called after predict.
    fn update(&mut self, z: &[f64]) -> UpdateResult;

    /// Convenience: predict + update in one call.
    fn step(&mut self, dt: f64, z: &[f64]) -> StepResult;

    /// Predict only — no measurement available (sensor dropout / missing bar).
    fn predict_only(&mut self, dt: f64) -> FilterState;

    /// Current state vector (post-update).
    fn state(&self) -> &[f64];

    /// Full covariance matrix (post-update).
    fn covariance(&self) -> &nalgebra::DMatrix<f64>;

    /// Jacobian of dynamics evaluated at the given state and dt.
    /// Linear KF returns the fixed F matrix regardless of state.
    /// EKF evaluates symbolically at the given point.
    fn jacobian(&self, x: &[f64], dt: f64) -> nalgebra::DMatrix<f64>;
}

pub struct FilterState {
    pub x: Vec<f64>,
    pub P: nalgebra::DMatrix<f64>,
}

pub struct UpdateResult {
    pub updated:        FilterState,
    pub residual:       Vec<f64>,           // z - H*x_predicted
    pub kalman_gain:    Vec<Vec<f64>>,      // K as row-major nested vec
    pub innovation_cov: Vec<Vec<f64>>,      // S = H*P*H' + R
}

pub struct StepResult {
    pub predicted: FilterState,
    pub update:    UpdateResult,
}
```

---

## CLI

### Invocation

```bash
# Live mode — reads from stdin, minimal output
./kalix --config configs/trend_with_accel.toml --mode live

# Backtest mode — reads from stdin, full audit output
./kalix --config configs/trend_with_accel.toml --mode backtest

# Backtest from file (--input implies --mode backtest)
./kalix --config configs/trend_with_accel.toml --input prices.jsonl

# --input combined with --mode live is a validation error — exit 1
```

### Startup Event (stdout, both modes)

Emitted once after successful config load, before reading any input:

```json
{
  "event": "ready",
  "filter": "trend_with_accel",
  "variant": "linear",
  "mode": "backtest",
  "state_variables": ["pos", "vel", "acc"],
  "observation_variables": ["z"],
  "F": [
    [1, 1, 0.5],
    [0, 1, 1],
    [0, 0, 1]
  ],
  "H": [[1, 0, 0]]
}
```

For EKF: `"variant": "ekf"` and `F` is omitted (recomputed per step).

---

## Input Format

One JSON object per line. Both modes share the same input format.

**Normal step** — observation available:

```json
{ "t": 1746912000.0, "dt": 1.0, "z": [10.3] }
```

**Predict-only step** — sensor dropout or missing bar:

```json
{ "t": 1746912001.0, "dt": 1.0, "z": null }
```

`t` is informational only — the filter does not infer `dt` from timestamps. Callers must
supply `dt` explicitly to handle irregular bars (weekends, missing ticks) correctly.

### Input Validation

Validate every incoming message before processing. On error, emit a structured JSON line
to **stderr** and apply the `--on-error` policy (default `skip`; `halt` exits with code 1):

```bash
./kalix --config trend.toml --mode live --on-error halt
```

| Condition          | Error message                              |
| ------------------ | ------------------------------------------ |
| `dt <= 0`          | `"invalid dt: must be positive, got {dt}"` |
| `dt` is NaN or inf | `"invalid dt: non-finite value"`           |
| wrong `z` length   | `"z has {n} elements, expected {m}"`       |
| unparseable JSON   | `"malformed input: {reason}"`              |

---

## Output Format

### Live Mode

One compact JSON line per input line. State named using `[state].variables`:

```json
{
  "t": 1746912000.0,
  "x": { "pos": 10.1, "vel": 0.31, "acc": 0.02 },
  "p_diag": [0.48, 0.19, 0.01]
}
```

For predict-only steps, shape is identical with `"predict_only": true` added:

```json
{
  "t": 1746912001.0,
  "predict_only": true,
  "x": { "pos": 10.41, "vel": 0.31, "acc": 0.02 },
  "p_diag": [0.51, 0.2, 0.011]
}
```

### Backtest Mode

One JSON object per input line. Full covariance matrices and named fields throughout.
`P` is a 2D array; variable order matches `[state].variables`.
Residual is named using `[observation].variables`.

```json
{
  "t": 1746912000.0,
  "step": 42,
  "predict": {
    "x": { "pos": 10.0, "vel": 0.3, "acc": 0.019 },
    "P": [
      [0.51, 0.01, 0.0],
      [0.01, 0.21, 0.0],
      [0.0, 0.0, 0.011]
    ]
  },
  "update": {
    "x": { "pos": 10.1, "vel": 0.31, "acc": 0.02 },
    "P": [
      [0.48, 0.009, 0.0],
      [0.009, 0.19, 0.0],
      [0.0, 0.0, 0.01]
    ],
    "residual": { "z": 0.2 },
    "kalman_gain": [[0.48], [0.09], [0.01]],
    "innovation_cov": [[1.48]]
  }
}
```

For predict-only steps in backtest mode, `"update"` is omitted and `"predict_only": true`
is added at the top level.

### Summary Event (backtest mode only)

Emitted to **stdout** when stdin reaches EOF or the input file is exhausted:

```json
{
  "event": "summary",
  "steps": 1000,
  "predict_only_steps": 12,
  "skipped_steps": 3,
  "final_x": { "pos": 10.1, "vel": 0.3, "acc": 0.019 },
  "final_p_diag": [0.48, 0.19, 0.01]
}
```

---

## Logging (stderr)

All diagnostic output goes to **stderr** as structured JSON via `tracing` +
`tracing-subscriber` with the JSON formatter. stdout is the data channel exclusively.

Control verbosity with `RUST_LOG`:

**`RUST_LOG=info`** — startup summary only:

```json
{"timestamp":"...","level":"INFO","message":"config loaded","filter":"trend_with_accel","variant":"linear"}
{"timestamp":"...","level":"INFO","message":"derived F","F":[[1,1,0.5],[0,1,1],[0,0,1]]}
{"timestamp":"...","level":"INFO","message":"derived H","H":[[1,0,0]]}
```

**`RUST_LOG=debug`** — per-step internals:

```json
{"timestamp":"...","level":"DEBUG","message":"predict","step":42,"x_prior":[10,0.3,0.02],"x_post":[10.3,0.3,0.02]}
{"timestamp":"...","level":"DEBUG","message":"kalman gain","K":[[0.48],[0.09],[0.01]]}
{"timestamp":"...","level":"DEBUG","message":"update","residual":[0.2],"p_diag_post":[0.48,0.19,0.01]}
```

EKF additionally logs the Jacobian evaluated at the current state each step at DEBUG level.

---

## Example Configs

### `configs/trend_no_accel.toml`

```toml
[filter]
name = "trend_no_accel"

[state]
variables = ["pos", "vel"]

[dynamics]
pos = "pos + vel*dt"
vel = "vel"

[observation]
variables   = ["z"]
expressions = ["pos"]

[noise]
process     = [[0.01, 0], [0, 0.01]]
measurement = [[1.0]]

[initial]
state      = [0.0, 0.0]
covariance = [[1, 0], [0, 1]]
```

### `configs/trend_with_accel.toml`

```toml
[filter]
name = "trend_with_accel"

[state]
variables = ["pos", "vel", "acc"]

[dynamics]
pos = "pos + vel*dt + 0.5*acc*dt^2"
vel = "vel + acc*dt"
acc = "acc"

[observation]
variables   = ["z"]
expressions = ["pos"]

[noise]
process     = [[0.01,0,0],[0,0.01,0],[0,0,0.01]]
measurement = [[1.0]]

[initial]
state      = [0.0, 0.0, 0.0]
covariance = [[1,0,0],[0,1,0],[0,0,1]]
```

### `configs/constant_velocity.toml`

```toml
[filter]
name = "constant_velocity"

[state]
variables = ["pos", "vel"]

[dynamics]
pos = "pos + vel*dt"
vel = "vel"

[observation]
variables   = ["z"]
expressions = ["pos"]

[noise]
process     = [[0.1, 0], [0, 0.1]]
measurement = [[5.0]]

[initial]
state      = [0.0, 0.0]
covariance = [[10, 0], [0, 10]]
```

### Pendulum (EKF with trig)

```toml
[filter]
name = "pendulum"

[state]
variables = ["theta", "omega"]

[dynamics]
theta = "theta + omega*dt"
omega = "omega - 9.81*sin(theta)*dt"

[observation]
variables   = ["z"]
expressions = ["theta"]

[noise]
process     = [[0.001, 0], [0, 0.001]]
measurement = [[0.1]]

[initial]
state      = [0.1, 0.0]
covariance = [[0.1, 0], [0, 0.1]]
```

---

## Language Integration

The CLI communicates exclusively via stdin/stdout JSON lines — no language-specific
bindings required. Any language that can spawn a child process works identically.

### Python

```python
import subprocess, json

proc = subprocess.Popen(
    ["./target/release/kalix", "--config", "configs/trend_with_accel.toml", "--mode", "live"],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
)

# Read the ready event
ready = json.loads(proc.stdout.readline())
print(f"Filter ready: {ready['filter']} ({ready['variant']})")

# Normal observation
proc.stdin.write(json.dumps({"t": 1000.0, "dt": 1.0, "z": [10.3]}) + "\n")
proc.stdin.flush()
result = json.loads(proc.stdout.readline())
print(f"pos={result['x']['pos']:.3f}, vel={result['x']['vel']:.3f}")

# Predict-only (sensor dropout)
proc.stdin.write(json.dumps({"t": 1001.0, "dt": 1.0, "z": None}) + "\n")
proc.stdin.flush()
result = json.loads(proc.stdout.readline())
print(f"predict-only: pos={result['x']['pos']:.3f}")

proc.stdin.close()
proc.wait()
```

### Python — backtest from file

```python
import subprocess, json

proc = subprocess.Popen(
    ["./target/release/kalix", "--config", "configs/trend_with_accel.toml", "--input", "prices.jsonl"],
    stdout=subprocess.PIPE, text=True
)

results = []
for line in proc.stdout:
    msg = json.loads(line)
    if msg.get("event") == "ready":
        continue
    if msg.get("event") == "summary":
        print(f"Done: {msg['steps']} steps, {msg['predict_only_steps']} predict-only")
        break
    results.append(msg)  # full per-step audit record

proc.wait()
```

### TypeScript

```typescript
import { spawn } from "child_process";
import * as readline from "readline";

const proc = spawn("./target/release/kalix", [
  "--config",
  "configs/trend_with_accel.toml",
  "--mode",
  "live",
]);

const rl = readline.createInterface({ input: proc.stdout });

rl.on("line", (line) => {
  const msg = JSON.parse(line);
  if (msg.event === "ready") {
    console.log("Filter ready:", msg.filter, msg.variant);
    return;
  }
  console.log("pos:", msg.x.pos, "vel:", msg.x.vel);
});

// Normal observation
proc.stdin.write(
  JSON.stringify({ t: Date.now() / 1000, dt: 1.0, z: [10.3] }) + "\n",
);

// Predict-only (sensor dropout)
proc.stdin.write(
  JSON.stringify({ t: Date.now() / 1000 + 1, dt: 1.0, z: null }) + "\n",
);
```

---

## Tests

Two clearly separated test concerns:

**Section 1 — Design**: verify TOML -> matrix derivation. Uses **exact equality** —
safe because all F/H entries are integers or IEEE-754-exact fractions (0.5, 0.125).

**Section 2 — Filter numerics**: verify running filter output. Uses
`approx::assert_abs_diff_eq!` throughout. **Never use exact float equality on filter
state.** Shape assertions (lengths, matrix dimensions) may use `assert_eq!`.

---

## Section 1 — Design: Matrix Derivation

### `tests/test_expr.rs` — Parser

```rust
// Literal
parse("3.14") -> Expr::Lit(3.14)

// Variable
parse("vel") -> Expr::Var("vel")

// Addition
parse("pos + vel") -> Add(Var("pos"), Var("vel"))

// Multiplication
parse("vel * dt") -> Mul(Var("vel"), Var("dt"))

// Integer power
parse("dt^2") -> Pow(Var("dt"), 2)

// Complex expression — must parse without error
parse("pos + vel*dt + 0.5*acc*dt^2") -> Ok(...)

// Trig and transcendental functions
parse("sin(pos)")  -> Sin(Var("pos"))
parse("cos(vel)")  -> Cos(Var("vel"))
parse("log(pos)")  -> Log(Var("pos"))
parse("exp(vel)")  -> Exp(Var("vel"))

// Unsupported — must return Err with descriptive message
parse("tan(pos)")  -> Err(...)    // unknown function
parse("pos / vel") -> Err(...)    // division by variable disallowed
```

---

### `tests/test_diff.rs` — Symbolic Differentiation

Differentiate symbolically then evaluate at concrete bindings. All values below are
IEEE-754 exact — assert with `epsilon = 1e-15`.

```rust
let expr = "pos + vel*dt + 0.5*acc*dt^2";

// d/d(pos) = 1.0 — independent of bindings
eval(diff(expr, "pos"), {pos:0, vel:0, acc:0, dt:1.0}) == 1.0
eval(diff(expr, "pos"), {pos:5, vel:3, acc:1, dt:0.5}) == 1.0

// d/d(vel) = dt
eval(diff(expr, "vel"), {dt: 1.0}) == 1.0
eval(diff(expr, "vel"), {dt: 0.5}) == 0.5

// d/d(acc) = 0.5 * dt^2
eval(diff(expr, "acc"), {dt: 1.0}) == 0.5    // 0.5 * 1.0 — exact
eval(diff(expr, "acc"), {dt: 0.5}) == 0.125  // 0.5 * 0.25 — exact

// Simplification: d/d(pos) of "pos" must reduce to Lit(1), not a compound expr
matches!(diff("pos", "pos"), Expr::Lit(v) if v == 1.0)

// d/d(vel) and d/d(acc) of "vel + acc*dt"
eval(diff("vel + acc*dt", "vel"), {dt: 1.0}) == 1.0
eval(diff("vel + acc*dt", "acc"), {dt: 1.0}) == 1.0
eval(diff("vel + acc*dt", "acc"), {dt: 0.5}) == 0.5

// ── Transcendental chain rule ────────────────────────────────────────
eval(diff("sin(x)", "x"), {x: 0.0}) == 1.0     (cos(0) = 1)
eval(diff("cos(x)", "x"), {x: π/6}) == -0.5     (-sin(π/6) = -0.5)
eval(diff("log(x)", "x"), {x: 2.0}) == 0.5      (1/2 = 0.5)
eval(diff("exp(x)", "x"), {x: 0.0}) == 1.0      (exp(0) = 1)
eval(diff("sin(2*x)", "x"), {x: 0.0}) == 2.0    (2*cos(0) = 2)
```

---

### `tests/test_config.rs` — Config Validation and Matrix Derivation

Exact equality on F and H is correct — all entries are 0, 1, or IEEE-754-exact fractions.

```rust
// ── trend_no_accel at dt=1.0 ──────────────────────────────────────────
// d(pos + vel*dt)/d(pos) = 1,  d(...)/d(vel) = dt = 1
// d(vel)/d(pos)          = 0,  d(vel)/d(vel) = 1
let F = derive_F(&config_no_accel, dt=1.0);
assert_eq!(F, [[1.0, 1.0],
               [0.0, 1.0]]);

let H = derive_H(&config_no_accel);
assert_eq!(H, [[1.0, 0.0]]);

assert_eq!(detect_variant(&config_no_accel), Variant::Linear);

// ── trend_with_accel at dt=1.0 ────────────────────────────────────────
// d(pos+vel*dt+0.5*acc*dt^2)/d(pos,vel,acc) = [1, 1,   0.5]
// d(vel+acc*dt)/d(pos,vel,acc)               = [0, 1,   1.0]
// d(acc)/d(pos,vel,acc)                      = [0, 0,   1.0]
let F = derive_F(&config_with_accel, dt=1.0);
assert_eq!(F, [[1.0, 1.0, 0.5],
               [0.0, 1.0, 1.0],
               [0.0, 0.0, 1.0]]);

// Same config at dt=0.5 — 0.5 and 0.125 are IEEE-754 exact
let F = derive_F(&config_with_accel, dt=0.5);
assert_eq!(F, [[1.0, 0.5, 0.125],
               [0.0, 1.0, 0.5  ],
               [0.0, 0.0, 1.0  ]]);

let H = derive_H(&config_with_accel);
assert_eq!(H, [[1.0, 0.0, 0.0]]);

assert_eq!(detect_variant(&config_with_accel), Variant::Linear);

// ── EKF detection (nonlinear dynamics) ────────────────────────────────
// Drag: vel_next = vel - 0.01*vel^2*dt  -> d/d(vel) = 1 - 0.02*vel*dt
assert_eq!(detect_variant(&config_drag), Variant::Ekf);

// Pendulum: omega_next = omega - 9.81*sin(theta)*dt  -> contains sin/theta
assert_eq!(detect_variant(&config_pendulum), Variant::Ekf);

// Exponential growth: x_next = x + 0.1*exp(x)*dt  -> contains exp
assert_eq!(detect_variant(&config_exp_growth), Variant::Ekf);

// ── Validation errors ─────────────────────────────────────────────────
config_with_bad_dynamics_key()       -> Err(msg contains "unknown state variable")
config_with_unknown_var_in_expr()    -> Err(msg contains "unknown variable")
config_with_wrong_Q_size()           -> Err(msg contains "Q must be 3x3")
config_with_input_and_live_mode()    -> Err(msg contains "--input requires backtest mode")
```

---

## Section 2 — Filter Numerics

All assertions use `approx::assert_abs_diff_eq!(actual, expected, epsilon = 1e-3)`
unless stated otherwise. Shape assertions use `assert_eq!`. Never use exact equality
on filter state.

All expected values were computed analytically and verified with a Python reference
implementation before inclusion in this prompt.

### `tests/test_linear_kf.rs`

Config: `trend_no_accel`, `x=[0,0]`, `P=I`, `Q=0.01*I`, `R=[[1.0]]`, `dt=1.0`.

```rust
// ── Step 1: z = [10.0] ───────────────────────────────────────────────
let result = filter.step(dt=1.0, z=[10.0]);

// Predicted (F*[0,0] = [0,0])
assert_abs_diff_eq!(result.predicted.x[0], 0.0, epsilon=1e-3);
assert_abs_diff_eq!(result.predicted.x[1], 0.0, epsilon=1e-3);

// Residual = z - H*x_predicted = 10.0 - 0.0
assert_abs_diff_eq!(result.update.residual[0], 10.0, epsilon=1e-3);

// Kalman gain shape: n x m = 2 x 1
assert_eq!(result.update.kalman_gain.len(),    2);
assert_eq!(result.update.kalman_gain[0].len(), 1);

// Kalman gain values: K ~= [0.6678, 0.3322]
// Gain is split across both state variables because vel feeds into pos prediction
assert_abs_diff_eq!(result.update.kalman_gain[0][0], 0.6678, epsilon=1e-3);
assert_abs_diff_eq!(result.update.kalman_gain[1][0], 0.3322, epsilon=1e-3);

// Updated state
assert_abs_diff_eq!(result.update.updated.x[0], 6.6777, epsilon=1e-3);
assert_abs_diff_eq!(result.update.updated.x[1], 3.3223, epsilon=1e-3);

// ── Step 2: z = [11.0] ───────────────────────────────────────────────
let result = filter.step(dt=1.0, z=[11.0]);

// Predicted: pos = 6.6777 + 3.3223*1.0 = 10.0, vel = 3.3223
assert_abs_diff_eq!(result.predicted.x[0], 10.0,   epsilon=1e-3);
assert_abs_diff_eq!(result.predicted.x[1],  3.3223, epsilon=1e-3);

// Residual = 11.0 - 10.0 = 1.0
assert_abs_diff_eq!(result.update.residual[0], 1.0, epsilon=1e-3);

// Updated
assert_abs_diff_eq!(result.update.updated.x[0], 10.6689, epsilon=1e-3);
assert_abs_diff_eq!(result.update.updated.x[1],  3.6567, epsilon=1e-3);

// ── Predict-only step ─────────────────────────────────────────────────
// After step 2, call predict_only — state advances but P must grow (no measurement)
let p_post_diag = result.update.updated.P.diagonal().clone();
let predicted = filter.predict_only(dt=1.0);
// pos_predicted ~= 10.6689 + 3.6567 = 14.3256
assert_abs_diff_eq!(predicted.x[0], 14.3256, epsilon=1e-2);
// Covariance diagonal must be >= post-update diagonal — uncertainty grows without data
for i in 0..2 {
    assert!(predicted.P[(i,i)] >= p_post_diag[i] - 1e-9);
}

// ── Convergence: 100 steps, z=10.0, fresh start ───────────────────────
// epsilon=0.01 — we care about convergence, not a precise settled value
assert_abs_diff_eq!(x[0], 10.0, epsilon=0.01);
assert_abs_diff_eq!(x[1],  0.0, epsilon=0.01);

// ── Named output — backtest serialisation ────────────────────────────
// Confirm JSON output uses state variable names, not integer indices
let json = serde_json::to_value(&backtest_step_output).unwrap();
assert!(json["update"]["x"]["pos"].is_number());
assert!(json["update"]["x"]["vel"].is_number());
assert!(json["update"]["residual"]["z"].is_number());
```

---

### `tests/test_ekf.rs`

Define the config as an inline `&str` constant inside the test file. Parse it with the
normal config loader. Do not load from disk.

```toml
[filter]
name = "drag"

[state]
variables = ["pos", "vel"]

[dynamics]
pos = "pos + vel*dt"
vel = "vel - 0.01*vel^2*dt"    # drag — nonlinear in vel

[observation]
variables   = ["z"]
expressions = ["pos"]

[noise]
process     = [[0.01, 0], [0, 0.01]]
measurement = [[1.0]]

[initial]
state      = [0.0, 1.0]
covariance = [[1, 0], [0, 1]]
```

```rust
// ── Variant detection ─────────────────────────────────────────────────
assert_eq!(detect_variant(&config), Variant::Ekf);

// ── Jacobian at state [10.0, 5.0], dt=1.0 ────────────────────────────
// pos_next = pos + vel*dt  ->  d/dpos=1, d/dvel=dt=1
// vel_next = vel - 0.01*vel^2*dt  ->  d/dpos=0, d/dvel = 1 - 0.02*vel*dt
// At vel=5.0, dt=1.0:  F[1][1] = 1 - 0.02*5.0*1.0 = 0.9  (IEEE-754 exact)
// Use tight epsilon here — these are exact symbolic evaluations, not accumulated ops
let jac = ekf.jacobian(&[10.0, 5.0], 1.0);
assert_abs_diff_eq!(jac[(0,0)], 1.0, epsilon=1e-10);
assert_abs_diff_eq!(jac[(0,1)], 1.0, epsilon=1e-10);
assert_abs_diff_eq!(jac[(1,0)], 0.0, epsilon=1e-10);
assert_abs_diff_eq!(jac[(1,1)], 0.9, epsilon=1e-10);

// ── Single step — must not panic, all outputs must be finite ──────────
let result = ekf.step(1.0, &[1.0]);
assert!(result.update.residual[0].is_finite());
assert!(result.update.updated.x.iter().all(|v| v.is_finite()));
assert!(result.update.kalman_gain.iter().flatten().all(|v| v.is_finite()));

// ── Stability: 50 steps, state must remain bounded ────────────────────
for _ in 0..50 {
    ekf.step(1.0, &[5.0]);
}
assert!(ekf.state().iter().all(|v| v.is_finite()));
assert!(ekf.state()[0].abs() < 1000.0);
assert!(ekf.state()[1].abs() < 1000.0);

// ── Predict-only in EKF — Jacobian must be re-evaluated at current state
let predicted = ekf.predict_only(1.0);
assert!(predicted.x.iter().all(|v| v.is_finite()));
assert!(predicted.P.iter().all(|v| v.is_finite()));

// ── Pendulum (trigonometric dynamics) ─────────────────────────────────
// Jacobian at theta=0, dt=1:  F = [[1, 1], [-9.81, 1]]
let jac = pendulum_ekf.jacobian(&[0.0, 0.0], 1.0);
assert_abs_diff_eq!(jac[(1,0)], -9.81, epsilon=1e-10);

// At theta=π/6: F[1][0] = -9.81*cos(π/6) ≈ -8.495
let jac = pendulum_ekf.jacobian(&[π/6, 0.0], 1.0);
assert_abs_diff_eq!(jac[(1,0)], -8.495, epsilon=1e-2);

// ── Exponential growth ───────────────────────────────────────────────
// x_next = x + 0.1*exp(x)*dt  ->  F = 1 + 0.1*exp(x)*dt
let jac = exp_ekf.jacobian(&[0.0], 1.0);
assert_abs_diff_eq!(jac[(0,0)], 1.1, epsilon=1e-10);
```

---

## Quality Bar

- `cargo test` — zero failures
- `cargo clippy -- -D warnings` — zero warnings
- All public types and trait methods must have doc comments
- `anyhow` for application-layer errors (main, CLI), `thiserror` for library errors
  (config, expr, filter) — do not mix
- No `unwrap()` or `expect()` in library code — propagate errors explicitly
- The expression evaluator and symbolic differentiator must be purely functional —
  no mutation of AST nodes, no interior mutability
- `src/io/` owns all serialisation logic — filter structs must not contain serde derives;
  the IO layer maps them to output shapes for each mode
