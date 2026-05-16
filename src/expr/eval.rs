//! Numeric evaluator for expression ASTs.
//!
//! Given an expression and a map of variable bindings, computes the numeric result.

use super::ast::Expr;
use std::collections::HashMap;

/// Evaluate an expression given variable bindings.
///
/// `bindings` maps variable names to their numeric values. Every variable
/// referenced in the expression must have a binding, or evaluation returns
/// an error.
pub fn eval(expr: &Expr, bindings: &[(&str, f64)]) -> Result<f64, String> {
    let map: HashMap<&str, f64> = bindings.iter().copied().collect();
    eval_with_map(expr, &map)
}

/// Evaluate with a pre-built `HashMap` of bindings.
pub fn eval_with_map(expr: &Expr, bindings: &HashMap<&str, f64>) -> Result<f64, String> {
    match expr {
        Expr::Lit(v) => Ok(*v),
        Expr::Var(name) => bindings
            .get(name.as_str())
            .copied()
            .ok_or_else(|| format!("undefined variable: {}", name)),
        Expr::Add(a, b) => Ok(eval_with_map(a, bindings)? + eval_with_map(b, bindings)?),
        Expr::Sub(a, b) => Ok(eval_with_map(a, bindings)? - eval_with_map(b, bindings)?),
        Expr::Mul(a, b) => Ok(eval_with_map(a, bindings)? * eval_with_map(b, bindings)?),
        Expr::Div(a, b) => {
            let denom = eval_with_map(b, bindings)?;
            if denom == 0.0 {
                return Err("division by zero".to_string());
            }
            Ok(eval_with_map(a, bindings)? / denom)
        }
        Expr::Pow(base, n) => {
            let val = eval_with_map(base, bindings)?;
            Ok(val.powi(*n as i32))
        }
        Expr::Sin(arg) => {
            let val = eval_with_map(arg, bindings)?;
            Ok(val.sin())
        }
        Expr::Cos(arg) => {
            let val = eval_with_map(arg, bindings)?;
            Ok(val.cos())
        }
        Expr::Log(arg) => {
            let val = eval_with_map(arg, bindings)?;
            if val <= 0.0 {
                return Err("log of non-positive number".to_string());
            }
            Ok(val.ln())
        }
        Expr::Exp(arg) => {
            let val = eval_with_map(arg, bindings)?;
            Ok(val.exp())
        }
    }
}
