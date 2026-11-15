//! Caller-defined stopping conditions for the shared Anderson engine.

use super::*;
use std::cell::Cell;

#[test]
fn custom_acceptance_ignores_builtin_tolerance() -> TestResult {
    let (device, client) = setup();
    let initial = Tensor::<CpuRuntime>::from_slice(&[1e16_f64], &[1], &device)?;
    let mapped = Tensor::<CpuRuntime>::from_slice(&[1.0_f64], &[1], &device)?;
    let mut options = picard_options(2, 1.0);
    options.base.tol = 0.0;
    let map_calls = Cell::new(0);
    let stop_calls = Cell::new(0);
    let result = anderson_with_convergence_impl(
        &client,
        |_| {
            map_calls.set(map_calls.get() + 1);
            Ok(mapped.clone())
        },
        &initial,
        &options,
        |x, gx, fx| {
            stop_calls.set(stop_calls.get() + 1);
            assert_eq!(x.item::<f64>()?, 1e16);
            assert_eq!(gx.item::<f64>()?, 1.0);
            assert_eq!(fx.item::<f64>()?, -1e16);
            Ok(AndersonConvergence {
                residual_norm: 123.0,
                converged: true,
            })
        },
    )?;

    assert!(result.converged);
    assert_eq!(result.iterations, 0);
    assert_eq!(result.residual_norm, 123.0);
    assert_eq!(result.x.item::<f64>()?.to_bits(), 1e16_f64.to_bits());
    assert_eq!(result.fun.item::<f64>()?, -1e16);
    assert_eq!(map_calls.get(), 1);
    assert_eq!(stop_calls.get(), 1);
    Ok(())
}

#[test]
fn custom_rejection_overrides_builtin_convergence() -> TestResult {
    let (device, client) = setup();
    let initial = Tensor::<CpuRuntime>::from_slice(&[0.0_f64], &[1], &device)?;
    let mut options = picard_options(2, 1.0);
    options.base.tol = f64::INFINITY;
    let map_calls = Cell::new(0);
    let stop_calls = Cell::new(0);
    let result = anderson_with_convergence_impl(
        &client,
        |x| {
            map_calls.set(map_calls.get() + 1);
            Ok(x.clone())
        },
        &initial,
        &options,
        |_, _, _| {
            stop_calls.set(stop_calls.get() + 1);
            Ok(AndersonConvergence {
                residual_norm: 0.0,
                converged: false,
            })
        },
    )?;

    assert!(!result.converged);
    assert_eq!(result.iterations, 2);
    assert_eq!(result.residual_norm, 0.0);
    assert_eq!(map_calls.get(), 3);
    assert_eq!(stop_calls.get(), 3);
    Ok(())
}

#[test]
fn exact_flag_agreement_controls_stopping() -> TestResult {
    let (device, client) = setup();
    let initial = Tensor::<CpuRuntime>::from_slice(&[1e16_f64, 0.0], &[2], &device)?;
    let mapped = Tensor::<CpuRuntime>::from_slice(&[1.0_f64, 1.0], &[2], &device)?;
    let mut options = picard_options(2, 1.0);
    options.base.tol = f64::INFINITY;
    let result = anderson_with_convergence_impl(
        &client,
        |_| Ok(mapped.clone()),
        &initial,
        &options,
        |x, gx, _| {
            let same = x.narrow(0, 1, 1)?.item::<f64>()? == gx.narrow(0, 1, 1)?.item::<f64>()?;
            Ok(AndersonConvergence {
                residual_norm: if same { 0.0 } else { 1.0 },
                converged: same,
            })
        },
    )?;

    assert!(result.converged);
    assert_eq!(result.iterations, 1);
    assert_eq!(result.x.try_to_vec::<f64>()?, vec![1.0, 1.0]);
    Ok(())
}

#[test]
fn convergence_callback_errors_retain_check_stage() -> TestResult {
    let (device, client) = setup();
    let initial = Tensor::<CpuRuntime>::from_slice(&[0.0_f64], &[1], &device)?;
    for error_at in [0, 1, 2] {
        let calls = Cell::new(0);
        let result = anderson_with_convergence_impl(
            &client,
            |x| client.add_scalar(x, 1.0),
            &initial,
            &picard_options(2, 1.0),
            |_, _, _| {
                let call = calls.get();
                calls.set(call + 1);
                if call == error_at {
                    return Err(numr::error::Error::InvalidArgument {
                        arg: "convergence",
                        reason: "declared stopping error".into(),
                    });
                }
                Ok(AndersonConvergence {
                    residual_norm: 1.0,
                    converged: false,
                })
            },
        );
        let expected = if error_at == 2 {
            "final check after 2 updates".to_owned()
        } else {
            format!("iteration {error_at}")
        };
        assert!(matches!(
            result,
            Err(OptimizeError::NumericalError { ref message })
                if message.contains("declared stopping error") && message.contains(&expected)
        ));
        assert_eq!(calls.get(), error_at + 1);
    }
    Ok(())
}

#[test]
fn invalid_custom_metrics_reject_at_iteration_and_final_checks() -> TestResult {
    let (device, client) = setup();
    let initial = Tensor::<CpuRuntime>::from_slice(&[0.0_f64], &[1], &device)?;
    let metrics = [
        (f64::NAN, false),
        (f64::NAN, true),
        (-1.0, false),
        (-1.0, true),
        (f64::NEG_INFINITY, false),
        (f64::NEG_INFINITY, true),
        (f64::INFINITY, true),
    ];
    for max_iter in [0, 1] {
        for (residual_norm, converged) in metrics {
            let result = anderson_with_convergence_impl(
                &client,
                |x| Ok(x.clone()),
                &initial,
                &picard_options(max_iter, 1.0),
                |_, _, _| {
                    Ok(AndersonConvergence {
                        residual_norm,
                        converged,
                    })
                },
            );
            assert!(matches!(
                result,
                Err(OptimizeError::NumericalError { ref message })
                    if message.contains("diagnostic residual_norm=")
            ));
        }
    }
    Ok(())
}

#[test]
fn infinite_metric_is_allowed_for_unconverged_iterates() -> TestResult {
    let (device, client) = setup();
    let initial = Tensor::<CpuRuntime>::from_slice(&[0.0_f64], &[1], &device)?;
    let result = anderson_with_convergence_impl(
        &client,
        |x| client.add_scalar(x, 1.0),
        &initial,
        &picard_options(2, 1.0),
        |_, _, _| {
            Ok(AndersonConvergence {
                residual_norm: f64::INFINITY,
                converged: false,
            })
        },
    )?;
    assert!(!result.converged);
    assert_eq!(result.iterations, 2);
    assert_eq!(result.residual_norm, f64::INFINITY);
    Ok(())
}

#[test]
fn equivalent_custom_norm_preserves_default_results() -> TestResult {
    let (device, client) = setup();
    let initial = Tensor::<CpuRuntime>::from_slice(&[-3.0_f64, 4.0, 1.0], &[3], &device)?;
    for (m, alpha) in [(0, 1.0), (0, 0.5), (3, 0.75)] {
        let mut options = picard_options(10, alpha);
        options.m = m;
        options.base.tol = 1e-10;
        let map = |x: &Tensor<CpuRuntime>| client.add_scalar(&client.mul_scalar(x, 0.25)?, 1.0);
        let default = anderson_impl(&client, map, &initial, &options)?;
        let custom =
            anderson_with_convergence_impl(&client, map, &initial, &options, |_, _, fx| {
                let residual_norm = tensor_norm(&client, fx)?;
                Ok(AndersonConvergence {
                    residual_norm,
                    converged: residual_norm < options.base.tol,
                })
            })?;
        assert_eq!(default.converged, custom.converged);
        assert_eq!(default.iterations, custom.iterations);
        assert_eq!(
            default.residual_norm.to_bits(),
            custom.residual_norm.to_bits()
        );
        for (left, right) in [(&default.x, &custom.x), (&default.fun, &custom.fun)] {
            assert_eq!(left.shape(), right.shape());
            for (left, right) in left
                .try_to_vec::<f64>()?
                .iter()
                .zip(right.try_to_vec::<f64>()?)
            {
                assert_eq!(left.to_bits(), right.to_bits());
            }
        }
    }
    Ok(())
}

#[test]
fn exhaustion_retains_evaluation_counts_and_never_accepts_final_decision() -> TestResult {
    let (device, client) = setup();
    let initial = Tensor::<CpuRuntime>::from_slice(&[0.0_f64], &[1], &device)?;
    for max_iter in [0, 2] {
        let map_calls = Cell::new(0);
        let stop_calls = Cell::new(0);
        let result = anderson_with_convergence_impl(
            &client,
            |x| {
                map_calls.set(map_calls.get() + 1);
                client.add_scalar(x, 1.0)
            },
            &initial,
            &picard_options(max_iter, 1.0),
            |_, _, _| {
                let call = stop_calls.get();
                stop_calls.set(call + 1);
                Ok(AndersonConvergence {
                    residual_norm: 42.0,
                    converged: call == max_iter,
                })
            },
        )?;
        assert!(!result.converged);
        assert_eq!(result.iterations, max_iter);
        assert_eq!(result.x.item::<f64>()?, max_iter as f64);
        assert_eq!(result.fun.item::<f64>()?, 1.0);
        assert_eq!(result.residual_norm, 42.0);
        assert_eq!(map_calls.get(), max_iter + 1);
        assert_eq!(stop_calls.get(), max_iter + 1);
    }
    Ok(())
}

#[test]
fn default_norm_keeps_legacy_nonfinite_diagnostics() -> TestResult {
    let (device, client) = setup();
    let initial = Tensor::<CpuRuntime>::from_slice(&[0.0_f64], &[1], &device)?;
    for value in [f64::NAN, f64::INFINITY] {
        let mapped = Tensor::<CpuRuntime>::from_slice(&[value], &[1], &device)?;
        let result = anderson_impl(
            &client,
            |_| Ok(mapped.clone()),
            &initial,
            &picard_options(0, 1.0),
        )?;
        assert!(!result.converged);
        assert_eq!(result.iterations, 0);
        if value.is_nan() {
            assert!(result.residual_norm.is_nan());
        } else {
            assert_eq!(result.residual_norm, value);
        }
    }
    Ok(())
}
