#[cfg(test)]
mod tests {
    use kalix::expr::diff::diff;
    use kalix::expr::eval::eval;
    use kalix::expr::parser::parse;

    fn diff_and_eval(expr_str: &str, var: &str, bindings: &[(&str, f64)]) -> f64 {
        let expr = parse(expr_str).unwrap();
        let d = diff(&expr, var);
        eval(&d, bindings).unwrap()
    }

    #[test]
    fn test_diff_pos_independent_of_bindings() {
        let expr = "pos + vel*dt + 0.5*acc*dt^2";

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
    fn test_diff_vel_equals_dt() {
        let expr = "pos + vel*dt + 0.5*acc*dt^2";

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
    fn test_diff_acc_equals_half_dt_squared() {
        let expr = "pos + vel*dt + 0.5*acc*dt^2";

        // d/d(acc) = 0.5 * dt^2
        let v = diff_and_eval(
            expr,
            "acc",
            &[("pos", 0.0), ("vel", 0.0), ("acc", 0.0), ("dt", 1.0)],
        );
        assert!((v - 0.5).abs() < 1e-15);

        // 0.5 * (0.5)^2 = 0.5 * 0.25 = 0.125
        let v = diff_and_eval(
            expr,
            "acc",
            &[("pos", 0.0), ("vel", 0.0), ("acc", 0.0), ("dt", 0.5)],
        );
        assert!((v - 0.125).abs() < 1e-15);
    }

    #[test]
    fn test_diff_pos_simplifies_to_lit_one() {
        use kalix::expr::ast::Expr;
        let expr = parse("pos").unwrap();
        let d = diff(&expr, "pos");
        assert!(matches!(d, Expr::Lit(v) if v == 1.0));
    }

    #[test]
    fn test_diff_vel_and_acc_in_vel_plus_acc_dt() {
        let expr = "vel + acc*dt";

        let v = diff_and_eval(expr, "vel", &[("vel", 0.0), ("acc", 0.0), ("dt", 1.0)]);
        assert!((v - 1.0).abs() < 1e-15);

        let v = diff_and_eval(expr, "acc", &[("vel", 0.0), ("acc", 0.0), ("dt", 1.0)]);
        assert!((v - 1.0).abs() < 1e-15);

        let v = diff_and_eval(expr, "acc", &[("vel", 0.0), ("acc", 0.0), ("dt", 0.5)]);
        assert!((v - 0.5).abs() < 1e-15);
    }
}
