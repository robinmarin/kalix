#![allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use kalix::config::Config;
    use kalix::filter::linear::LinearKF;
    use kalix::filter::traits::KalmanFilter;

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

    fn build_filter() -> LinearKF {
        let config = Config::from_toml(TREND_NO_ACCEL).unwrap();
        let F = config.derive_F(1.0);
        let H = config.derive_H();
        LinearKF::new(
            F,
            H,
            config.Q.clone(),
            config.R.clone(),
            &config.x0,
            config.P0.clone(),
        )
    }

    #[test]
    fn test_step_1() {
        let mut filter = build_filter();
        let result = filter.step(1.0, &[10.0]);

        // Predicted: F*[0,0] = [0,0]
        assert_abs_diff_eq!(result.predicted.x[0], 0.0, epsilon = 1e-3);
        assert_abs_diff_eq!(result.predicted.x[1], 0.0, epsilon = 1e-3);

        // Residual = z - H*x_predicted = 10.0 - 0.0
        assert_abs_diff_eq!(result.update.residual[0], 10.0, epsilon = 1e-3);

        // Kalman gain shape: n x m = 2 x 1
        assert_eq!(result.update.kalman_gain.len(), 2);
        assert_eq!(result.update.kalman_gain[0].len(), 1);

        // Kalman gain values: K ~= [0.6678, 0.3322]
        assert_abs_diff_eq!(result.update.kalman_gain[0][0], 0.6678, epsilon = 1e-3);
        assert_abs_diff_eq!(result.update.kalman_gain[1][0], 0.3322, epsilon = 1e-3);

        // Updated state
        assert_abs_diff_eq!(result.update.updated.x[0], 6.6777, epsilon = 1e-3);
        assert_abs_diff_eq!(result.update.updated.x[1], 3.3223, epsilon = 1e-3);
    }

    #[test]
    fn test_step_2() {
        let mut filter = build_filter();
        filter.step(1.0, &[10.0]);
        let result = filter.step(1.0, &[11.0]);

        // Predicted: pos = 6.6777 + 3.3223*1.0 = 10.0, vel = 3.3223
        assert_abs_diff_eq!(result.predicted.x[0], 10.0, epsilon = 1e-3);
        assert_abs_diff_eq!(result.predicted.x[1], 3.3223, epsilon = 1e-3);

        // Residual = 11.0 - 10.0 = 1.0
        assert_abs_diff_eq!(result.update.residual[0], 1.0, epsilon = 1e-3);

        // Updated
        assert_abs_diff_eq!(result.update.updated.x[0], 10.6689, epsilon = 1e-3);
        assert_abs_diff_eq!(result.update.updated.x[1], 3.6567, epsilon = 1e-3);
    }

    #[test]
    fn test_predict_only_covariance_grows() {
        let mut filter = build_filter();
        let result = filter.step(1.0, &[10.0]);
        let p_post_diag: Vec<f64> = (0..2).map(|i| result.update.updated.P[(i, i)]).collect();

        let predicted = filter.predict_only(1.0);
        // pos_predicted ~= 6.6777 + 3.3223 = 10.0
        assert_abs_diff_eq!(predicted.x[0], 10.0, epsilon = 1e-2);
        // vel_predicted ~= 3.3223
        assert_abs_diff_eq!(predicted.x[1], 3.3223, epsilon = 1e-2);

        // Covariance diagonal must be >= post-update diagonal — uncertainty grows without data
        for i in 0..2 {
            assert!(predicted.P[(i, i)] >= p_post_diag[i] - 1e-9);
        }
    }

    #[test]
    fn test_convergence_100_steps() {
        let mut filter = build_filter();
        for _ in 0..100 {
            filter.step(1.0, &[10.0]);
        }
        // After 100 steps observing z=10.0, the filter should converge near z=10.0, vel=0.0
        assert_abs_diff_eq!(filter.state()[0], 10.0, epsilon = 0.01);
        assert_abs_diff_eq!(filter.state()[1], 0.0, epsilon = 0.01);
    }

    #[test]
    fn test_named_output_uses_state_variable_names() {
        use kalix::io::output;
        let mut filter = build_filter();
        let result = filter.step(1.0, &[10.0]);

        let named_x = output::make_named_state(
            &["pos".to_string(), "vel".to_string()],
            &result.update.updated.x,
        );
        let json = serde_json::to_value(&named_x).unwrap();
        assert!(json["pos"].is_number());
        assert!(json["vel"].is_number());
    }
}
