#[cfg(test)]
mod tests {
    use kalix::expr::ast::Expr;
    use kalix::expr::parser::parse;

    #[test]
    fn test_literal() {
        assert_eq!(parse("3.14"), Ok(Expr::Lit(3.14)));
    }

    #[test]
    fn test_variable() {
        assert_eq!(parse("vel"), Ok(Expr::Var("vel".to_string())));
    }

    #[test]
    fn test_addition() {
        assert_eq!(
            parse("pos + vel"),
            Ok(Expr::add(Expr::var("pos"), Expr::var("vel")))
        );
    }

    #[test]
    fn test_multiplication() {
        assert_eq!(
            parse("vel * dt"),
            Ok(Expr::mul(Expr::var("vel"), Expr::var("dt")))
        );
    }

    #[test]
    fn test_integer_power() {
        assert_eq!(parse("dt^2"), Ok(Expr::pow(Expr::var("dt"), 2)));
    }

    #[test]
    fn test_complex_expression_parses() {
        let result = parse("pos + vel*dt + 0.5*acc*dt^2");
        assert!(result.is_ok(), "Expected OK, got {:?}", result);
    }

    #[test]
    fn test_sin_parses() {
        assert_eq!(parse("sin(pos)"), Ok(Expr::sin(Expr::var("pos"))));
    }

    #[test]
    fn test_cos_parses() {
        assert_eq!(parse("cos(vel)"), Ok(Expr::cos(Expr::var("vel"))));
    }

    #[test]
    fn test_sin_with_arith() {
        assert_eq!(
            parse("sin(vel * dt)"),
            Ok(Expr::sin(Expr::mul(Expr::var("vel"), Expr::var("dt")))),
        );
    }

    #[test]
    fn test_unknown_function_rejected() {
        assert!(parse("tan(pos)").is_err());
        assert!(parse("sqrt(pos)").is_err());
    }

    #[test]
    fn test_log_parses() {
        assert_eq!(parse("log(pos)"), Ok(Expr::log(Expr::var("pos"))),);
    }

    #[test]
    fn test_exp_parses() {
        assert_eq!(parse("exp(vel)"), Ok(Expr::exp(Expr::var("vel"))),);
    }

    #[test]
    fn test_division_by_variable_disallowed() {
        let result = parse("pos / vel");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("division"));
    }
}
