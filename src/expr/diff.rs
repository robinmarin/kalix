//! Symbolic differentiation of expression ASTs.
//!
//! Implements `diff(expr, var)` and a post-differentiation simplification pass.
//! All operations are purely functional — no mutation of AST nodes.

use super::ast::Expr;

/// Symbolically differentiate `expr` with respect to `var`.
///
/// Returns a new (unsimplified) expression. Call `simplify` afterwards
/// to collapse trivial forms (e.g. `0 + x → x`, `1 * x → x`).
pub fn diff(expr: &Expr, var: &str) -> Expr {
    let raw = diff_raw(expr, var);
    simplify(raw)
}

/// Raw differentiation without simplification.
fn diff_raw(expr: &Expr, var: &str) -> Expr {
    match expr {
        Expr::Lit(_) => Expr::Lit(0.0),

        Expr::Var(name) => {
            if name == var {
                Expr::Lit(1.0)
            } else {
                Expr::Lit(0.0)
            }
        }

        Expr::Add(a, b) => {
            let da = diff_raw(a, var);
            let db = diff_raw(b, var);
            Expr::add(da, db)
        }

        Expr::Sub(a, b) => {
            let da = diff_raw(a, var);
            let db = diff_raw(b, var);
            Expr::sub(da, db)
        }

        Expr::Mul(a, b) => {
            // d/dx(u*v) = u'v + uv'
            let da = diff_raw(a, var);
            let db = diff_raw(b, var);
            Expr::add(
                Expr::mul(da, b.as_ref().clone()),
                Expr::mul(a.as_ref().clone(), db),
            )
        }

        Expr::Div(a, b) => {
            // d/dx(u/v) = (u'v - uv') / v^2
            // Note: v must be a literal, so v' = 0, simplifying to u'/v
            let da = diff_raw(a, var);
            // Since b is always a literal per validation, db = 0
            if let Expr::Lit(v) = b.as_ref() {
                Expr::div(da, Expr::Lit(*v))
            } else {
                // Fallback: full quotient rule (shouldn't happen after validation)
                let db = diff_raw(b, var);
                Expr::div(
                    Expr::sub(
                        Expr::mul(da, b.as_ref().clone()),
                        Expr::mul(a.as_ref().clone(), db),
                    ),
                    Expr::pow(b.as_ref().clone(), 2),
                )
            }
        }

        Expr::Pow(base, n) => {
            if *n == 0 {
                // d/dx(1) = 0
                Expr::Lit(0.0)
            } else {
                // d/dx(u^n) = n * u^(n-1) * u'
                let dbase = diff_raw(base, var);
                Expr::mul(
                    Expr::mul(
                        Expr::Lit(*n as f64),
                        Expr::pow(base.as_ref().clone(), n - 1),
                    ),
                    dbase,
                )
            }
        }

        Expr::Sin(arg) => {
            // d/dx(sin(u)) = cos(u) * u'
            let darg = diff_raw(arg, var);
            Expr::mul(Expr::cos(arg.as_ref().clone()), darg)
        }

        Expr::Cos(arg) => {
            // d/dx(cos(u)) = -sin(u) * u'
            let darg = diff_raw(arg, var);
            Expr::mul(
                Expr::mul(Expr::Lit(-1.0), Expr::sin(arg.as_ref().clone())),
                darg,
            )
        }

        Expr::Log(arg) => {
            // d/dx(log(u)) = u' / u
            let darg = diff_raw(arg, var);
            Expr::div(darg, arg.as_ref().clone())
        }

        Expr::Exp(arg) => {
            // d/dx(exp(u)) = exp(u) * u'
            let darg = diff_raw(arg, var);
            Expr::mul(Expr::exp(arg.as_ref().clone()), darg)
        }
    }
}

/// Simplify an expression by collapsing trivial algebraic forms.
///
/// Rules applied bottom-up:
/// - `0 + x → x`, `x + 0 → x`
/// - `0 - x → -1 * x`, `x - 0 → x`
/// - `1 * x → x`, `x * 1 → x`, `0 * x → 0`, `x * 0 → 0`
/// - `x / 1 → x`, `0 / x → 0`
/// - `x ^ 0 → 1`, `x ^ 1 → x`
/// - Constant folding: `Lit(a) + Lit(b) → Lit(a+b)`, etc.
pub fn simplify(expr: Expr) -> Expr {
    match expr {
        // Leaves — no simplification needed
        Expr::Lit(_) | Expr::Var(_) => expr,

        Expr::Add(a, b) => {
            let a = simplify(*a);
            let b = simplify(*b);
            match (&a, &b) {
                // 0 + x → x
                (Expr::Lit(v), _) if *v == 0.0 => b,
                // x + 0 → x
                (_, Expr::Lit(v)) if *v == 0.0 => a,
                // Lit + Lit → Lit
                (Expr::Lit(va), Expr::Lit(vb)) => Expr::Lit(va + vb),
                _ => Expr::add(a, b),
            }
        }

        Expr::Sub(a, b) => {
            let a = simplify(*a);
            let b = simplify(*b);
            match (&a, &b) {
                // x - 0 → x
                (_, Expr::Lit(v)) if *v == 0.0 => a,
                // 0 - x → -1 * x
                (Expr::Lit(v), _) if *v == 0.0 => Expr::mul(Expr::Lit(-1.0), b),
                // Lit - Lit → Lit
                (Expr::Lit(va), Expr::Lit(vb)) => Expr::Lit(va - vb),
                _ => Expr::sub(a, b),
            }
        }

        Expr::Mul(a, b) => {
            let a = simplify(*a);
            let b = simplify(*b);
            match (&a, &b) {
                // 0 * x → 0
                (Expr::Lit(v), _) if *v == 0.0 => Expr::Lit(0.0),
                // x * 0 → 0
                (_, Expr::Lit(v)) if *v == 0.0 => Expr::Lit(0.0),
                // 1 * x → x
                (Expr::Lit(v), _) if *v == 1.0 => b,
                // x * 1 → x
                (_, Expr::Lit(v)) if *v == 1.0 => a,
                // Lit * Lit → Lit
                (Expr::Lit(va), Expr::Lit(vb)) => Expr::Lit(va * vb),
                _ => Expr::mul(a, b),
            }
        }

        Expr::Div(a, b) => {
            let a = simplify(*a);
            let b = simplify(*b);
            match (&a, &b) {
                // 0 / x → 0
                (Expr::Lit(v), _) if *v == 0.0 => Expr::Lit(0.0),
                // x / 1 → x
                (_, Expr::Lit(v)) if *v == 1.0 => a,
                // Lit / Lit → Lit
                (Expr::Lit(va), Expr::Lit(vb)) if *vb != 0.0 => Expr::Lit(va / vb),
                _ => Expr::div(a, b),
            }
        }

        Expr::Pow(base, n) => {
            let base = simplify(*base);
            match n {
                1 => base,           // x ^ 1 → x
                0 => Expr::Lit(1.0), // x ^ 0 → 1
                _ => Expr::pow(base, n),
            }
        }

        Expr::Sin(arg) => Expr::sin(simplify(*arg)),
        Expr::Cos(arg) => Expr::cos(simplify(*arg)),
        Expr::Log(arg) => Expr::log(simplify(*arg)),
        Expr::Exp(arg) => Expr::exp(simplify(*arg)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::eval::eval;
    use crate::expr::parser::parse;

    fn diff_and_eval(expr_str: &str, var: &str, bindings: &[(&str, f64)]) -> f64 {
        let expr = parse(expr_str).unwrap();
        let d = diff(&expr, var);
        eval(&d, bindings).unwrap()
    }

    #[test]
    fn test_diff_pos_in_trend() {
        let expr = "pos + vel*dt + 0.5*acc*dt^2";

        // d/d(pos) = 1.0 — independent of bindings
        let v = diff_and_eval(
            expr,
            "pos",
            &[("pos", 0.0), ("vel", 0.0), ("acc", 0.0), ("dt", 1.0)],
        );
        assert!((v - 1.0).abs() < 1e-15);

        let v = diff_and_eval(
            expr,
            "pos",
            &[("pos", 5.0), ("vel", 3.0), ("acc", 1.0), ("dt", 0.5)],
        );
        assert!((v - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_diff_vel_in_trend() {
        let expr = "pos + vel*dt + 0.5*acc*dt^2";

        // d/d(vel) = dt
        let v = diff_and_eval(
            expr,
            "vel",
            &[("pos", 0.0), ("vel", 0.0), ("acc", 0.0), ("dt", 1.0)],
        );
        assert!((v - 1.0).abs() < 1e-15);

        let v = diff_and_eval(
            expr,
            "vel",
            &[("pos", 0.0), ("vel", 0.0), ("acc", 0.0), ("dt", 0.5)],
        );
        assert!((v - 0.5).abs() < 1e-15);
    }

    #[test]
    fn test_diff_acc_in_trend() {
        let expr = "pos + vel*dt + 0.5*acc*dt^2";

        // d/d(acc) = 0.5 * dt^2
        let v = diff_and_eval(
            expr,
            "acc",
            &[("pos", 0.0), ("vel", 0.0), ("acc", 0.0), ("dt", 1.0)],
        );
        assert!((v - 0.5).abs() < 1e-15);

        let v = diff_and_eval(
            expr,
            "acc",
            &[("pos", 0.0), ("vel", 0.0), ("acc", 0.0), ("dt", 0.5)],
        );
        assert!((v - 0.125).abs() < 1e-15);
    }

    #[test]
    fn test_diff_pos_simple() {
        let expr = parse("pos").unwrap();
        let d = diff(&expr, "pos");
        // Must reduce to Lit(1), not a compound expr
        assert!(matches!(d, Expr::Lit(v) if v == 1.0));
    }

    #[test]
    fn test_diff_vel_acc_expr() {
        let expr = "vel + acc*dt";

        let v = diff_and_eval(expr, "vel", &[("vel", 0.0), ("acc", 0.0), ("dt", 1.0)]);
        assert!((v - 1.0).abs() < 1e-15);

        let v = diff_and_eval(expr, "acc", &[("vel", 0.0), ("acc", 0.0), ("dt", 1.0)]);
        assert!((v - 1.0).abs() < 1e-15);

        let v = diff_and_eval(expr, "acc", &[("vel", 0.0), ("acc", 0.0), ("dt", 0.5)]);
        assert!((v - 0.5).abs() < 1e-15);
    }

    #[test]
    fn test_simplify_zero_plus_x() {
        let e = Expr::add(Expr::Lit(0.0), Expr::var("x"));
        assert_eq!(simplify(e), Expr::var("x"));
    }

    #[test]
    fn test_simplify_one_times_x() {
        let e = Expr::mul(Expr::Lit(1.0), Expr::var("x"));
        assert_eq!(simplify(e), Expr::var("x"));
    }

    #[test]
    fn test_simplify_zero_times_x() {
        let e = Expr::mul(Expr::Lit(0.0), Expr::var("x"));
        assert_eq!(simplify(e), Expr::Lit(0.0));
    }

    #[test]
    fn test_simplify_pow_one() {
        let e = Expr::pow(Expr::var("x"), 1);
        assert_eq!(simplify(e), Expr::var("x"));
    }

    #[test]
    fn test_simplify_pow_zero() {
        let e = Expr::pow(Expr::var("x"), 0);
        assert_eq!(simplify(e), Expr::Lit(1.0));
    }
}
