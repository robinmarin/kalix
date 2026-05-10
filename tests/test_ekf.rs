#![allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use kalix::config::Config;
    use kalix::filter::ekf::EKF;
    use kalix::filter::traits::KalmanFilter;

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

    fn build_ekf() -> EKF {
        let config = Config::from_toml(DRAG_CONFIG).unwrap();
        let H = config.derive_H();
        EKF::new(
            config.dynamics,
            config.state_variables,
            H,
            config.Q,
            config.R,
            &config.x0,
            config.P0,
        )
    }

    #[test]
    fn test_variant_is_ekf() {
        let config = Config::from_toml(DRAG_CONFIG).unwrap();
        assert_eq!(config.variant, kalix::config::Variant::Ekf);
    }

    #[test]
    fn test_jacobian_at_state_10_5() {
        let ekf = build_ekf();
        // pos_next = pos + vel*dt  ->  d/dpos=1, d/dvel=dt=1
        // vel_next = vel - 0.01*vel^2*dt  ->  d/dpos=0, d/dvel = 1 - 0.02*vel*dt
        // At vel=5.0, dt=1.0:  F[1][1] = 1 - 0.02*5.0*1.0 = 0.9
        let jac = ekf.jacobian(&[10.0, 5.0], 1.0);
        assert_abs_diff_eq!(jac[(0, 0)], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(jac[(0, 1)], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(jac[(1, 0)], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(jac[(1, 1)], 0.9, epsilon = 1e-10);
    }

    #[test]
    fn test_single_step_outputs_finite() {
        let mut ekf = build_ekf();
        let result = ekf.step(1.0, &[1.0]);
        assert!(result.update.residual[0].is_finite());
        assert!(result.update.updated.x.iter().all(|v| v.is_finite()));
        assert!(result
            .update
            .kalman_gain
            .iter()
            .flatten()
            .all(|v| v.is_finite()));
    }

    #[test]
    fn test_stability_50_steps() {
        let mut ekf = build_ekf();
        for _ in 0..50 {
            ekf.step(1.0, &[5.0]);
        }
        assert!(ekf.state().iter().all(|v| v.is_finite()));
        assert!(ekf.state()[0].abs() < 1000.0);
        assert!(ekf.state()[1].abs() < 1000.0);
    }

    #[test]
    fn test_predict_only_ekf_finite() {
        let mut ekf = build_ekf();
        let predicted = ekf.predict_only(1.0);
        assert!(predicted.x.iter().all(|v| v.is_finite()));
        assert!(predicted.P.iter().all(|v| v.is_finite()));
    }
}
