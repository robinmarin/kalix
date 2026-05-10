#![allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use kalix::config::{Config, Variant};

    const TREND_NO_ACCEL: &str = r#"
[filter]
name = "trend_no_accel"
description = "Position/velocity trend follower (no acceleration)"

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
"#;

    const TREND_WITH_ACCEL: &str = r#"
[filter]
name = "trend_with_accel"
description = "Position/velocity/acceleration trend follower"

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
process     = [[0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]]
measurement = [[1.0]]

[initial]
state      = [0.0, 0.0, 0.0]
covariance = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
"#;

    #[test]
    fn test_trend_no_accel_F_dt_1() {
        let config = Config::from_toml(TREND_NO_ACCEL).unwrap();
        let F = config.derive_F(1.0);
        // d(pos+vel*dt)/d(pos) = 1,  d(...)/d(vel) = dt = 1
        // d(vel)/d(pos)          = 0,  d(vel)/d(vel) = 1
        assert_eq!(F[(0, 0)], 1.0);
        assert_eq!(F[(0, 1)], 1.0);
        assert_eq!(F[(1, 0)], 0.0);
        assert_eq!(F[(1, 1)], 1.0);
    }

    #[test]
    fn test_trend_no_accel_H() {
        let config = Config::from_toml(TREND_NO_ACCEL).unwrap();
        let H = config.derive_H();
        assert_eq!(H[(0, 0)], 1.0);
        assert_eq!(H[(0, 1)], 0.0);
    }

    #[test]
    fn test_trend_no_accel_is_linear() {
        let config = Config::from_toml(TREND_NO_ACCEL).unwrap();
        assert_eq!(config.variant, Variant::Linear);
    }

    #[test]
    fn test_trend_with_accel_F_dt_1() {
        let config = Config::from_toml(TREND_WITH_ACCEL).unwrap();
        let F = config.derive_F(1.0);
        // d(pos+vel*dt+0.5*acc*dt^2)/d(pos,vel,acc) = [1, 1,   0.5]
        // d(vel+acc*dt)/d(pos,vel,acc)               = [0, 1,   1.0]
        // d(acc)/d(pos,vel,acc)                      = [0, 0,   1.0]
        assert_eq!(F[(0, 0)], 1.0);
        assert_eq!(F[(0, 1)], 1.0);
        assert_eq!(F[(0, 2)], 0.5);
        assert_eq!(F[(1, 0)], 0.0);
        assert_eq!(F[(1, 1)], 1.0);
        assert_eq!(F[(1, 2)], 1.0);
        assert_eq!(F[(2, 0)], 0.0);
        assert_eq!(F[(2, 1)], 0.0);
        assert_eq!(F[(2, 2)], 1.0);
    }

    #[test]
    fn test_trend_with_accel_F_dt_0_5() {
        let config = Config::from_toml(TREND_WITH_ACCEL).unwrap();
        let F = config.derive_F(0.5);
        assert_eq!(F[(0, 0)], 1.0);
        assert_eq!(F[(0, 1)], 0.5);
        assert_eq!(F[(0, 2)], 0.125);
        assert_eq!(F[(1, 0)], 0.0);
        assert_eq!(F[(1, 1)], 1.0);
        assert_eq!(F[(1, 2)], 0.5);
        assert_eq!(F[(2, 0)], 0.0);
        assert_eq!(F[(2, 1)], 0.0);
        assert_eq!(F[(2, 2)], 1.0);
    }

    #[test]
    fn test_trend_with_accel_H() {
        let config = Config::from_toml(TREND_WITH_ACCEL).unwrap();
        let H = config.derive_H();
        assert_eq!(H[(0, 0)], 1.0);
        assert_eq!(H[(0, 1)], 0.0);
        assert_eq!(H[(0, 2)], 0.0);
    }

    #[test]
    fn test_trend_with_accel_is_linear() {
        let config = Config::from_toml(TREND_WITH_ACCEL).unwrap();
        assert_eq!(config.variant, Variant::Linear);
    }

    #[test]
    fn test_ekf_detection_drag() {
        const DRAG_CONFIG: &str = r#"
[filter]
name = "drag"

[state]
variables = ["pos", "vel"]

[dynamics]
pos = "pos + vel*dt"
vel = "vel - 0.01*vel^2*dt"

[observation]
variables   = ["z"]
expressions = ["pos"]

[noise]
process     = [[0.01, 0], [0, 0.01]]
measurement = [[1.0]]

[initial]
state      = [0.0, 1.0]
covariance = [[1, 0], [0, 1]]
"#;
        let config = Config::from_toml(DRAG_CONFIG).unwrap();
        assert_eq!(config.variant, Variant::Ekf);
    }

    #[test]
    fn test_unknown_variable_in_expr() {
        const BAD_CONFIG: &str = r#"
[filter]
name = "bad"

[state]
variables = ["pos"]

[dynamics]
pos = "pos + unknown*dt"

[observation]
variables   = ["z"]
expressions = ["pos"]

[noise]
process     = [[0.01]]
measurement = [[1.0]]

[initial]
state      = [0.0]
covariance = [[1.0]]
"#;
        let result = Config::from_toml(BAD_CONFIG);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("unknown variable"));
    }

    #[test]
    fn test_wrong_q_size_is_error() {
        const BAD_CONFIG: &str = r#"
[filter]
name = "bad"

[state]
variables = ["pos", "vel", "acc"]

[dynamics]
pos = "pos + vel*dt"
vel = "vel"
acc = "acc"

[observation]
variables   = ["z"]
expressions = ["pos"]

[noise]
process     = [[0.01, 0], [0, 0.01]]
measurement = [[1.0]]

[initial]
state      = [0.0, 0.0, 0.0]
covariance = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
"#;
        let result = Config::from_toml(BAD_CONFIG);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Q must be 3x3"));
    }
}
