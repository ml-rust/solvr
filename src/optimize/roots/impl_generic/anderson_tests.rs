use super::*;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

fn setup() -> (CpuDevice, CpuClient) {
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    (device, client)
}

#[test]
fn test_anderson_cosine_fixed_point() {
    let (device, client) = setup();

    // Fixed point of g(x) = cos(x): x* ≈ 0.7390851332
    let x0 = Tensor::<CpuRuntime>::from_slice(&[1.0f64], &[1], &device).unwrap();

    let result = anderson_impl(
        &client,
        |x| {
            let data: Vec<f64> = x.to_vec();
            let result: Vec<f64> = data.iter().map(|xi| xi.cos()).collect();
            Tensor::from_slice(&result, x.shape(), x.device())
        },
        &x0,
        &AndersonOptions::default(),
    )
    .unwrap();

    assert!(result.converged, "Anderson did not converge");
    let sol: Vec<f64> = result.x.to_vec();
    assert!((sol[0] - 0.7390851332).abs() < 1e-6, "sol = {}", sol[0]);
}

#[test]
fn test_anderson_linear_fixed_point() {
    let (device, client) = setup();

    // g(x) = 0.5 * x + 1, fixed point at x = 2
    let x0 = Tensor::<CpuRuntime>::from_slice(&[0.0f64], &[1], &device).unwrap();

    let result = anderson_impl(
        &client,
        |x| {
            let data: Vec<f64> = x.to_vec();
            let result: Vec<f64> = data.iter().map(|xi| 0.5 * xi + 1.0).collect();
            Tensor::from_slice(&result, x.shape(), x.device())
        },
        &x0,
        &AndersonOptions::default(),
    )
    .unwrap();

    assert!(result.converged);
    let sol: Vec<f64> = result.x.to_vec();
    assert!((sol[0] - 2.0).abs() < 1e-6);
}

#[test]
fn test_anderson_identical_residual_history() {
    // g(x) = x + d has a constant residual d. Every residual difference
    // is zero, so the history matrix has identical (all-zero) columns.
    // The min-norm solve gives theta = 0 and each step adds alpha * d.
    let (device, client) = setup();
    let d = [1.0f64, -2.0, 0.5];
    let x0 = Tensor::<CpuRuntime>::from_slice(&[0.0f64; 3], &[3], &device).unwrap();
    let mut options = AndersonOptions {
        m: 3,
        alpha: 0.5,
        ..Default::default()
    };
    options.base.max_iter = 6;

    let result = anderson_impl(
        &client,
        |x| {
            let data: Vec<f64> = x.to_vec();
            let out: Vec<f64> = data.iter().zip(d.iter()).map(|(a, b)| a + b).collect();
            Tensor::from_slice(&out, x.shape(), x.device())
        },
        &x0,
        &options,
    )
    .unwrap();

    assert!(!result.converged);
    let sol: Vec<f64> = result.x.to_vec();
    for (s, di) in sol.iter().zip(d.iter()) {
        assert!(s.is_finite());
        assert!((s - 3.0 * di).abs() < 1e-12, "sol = {:?}", sol);
    }
    let norm = (1.0f64 + 4.0 + 0.25).sqrt();
    assert!((result.residual_norm - norm).abs() < 1e-12);
}

#[test]
fn test_anderson_history_shorter_than_unknowns() {
    // Six unknowns with history depth two: at most two difference
    // columns, fewer than the unknown count. Fixed point of
    // g_i(x) = a_i x_i + 1 is x_i = 1 / (1 - a_i).
    let (device, client) = setup();
    let a = [0.1f64, 0.2, 0.3, 0.4, 0.5, 0.6];
    let x0 = Tensor::<CpuRuntime>::from_slice(&[0.0f64; 6], &[6], &device).unwrap();
    let mut options = AndersonOptions {
        m: 2,
        ..Default::default()
    };
    options.base.max_iter = 500;

    let result = anderson_impl(
        &client,
        |x| {
            let data: Vec<f64> = x.to_vec();
            let out: Vec<f64> = data
                .iter()
                .zip(a.iter())
                .map(|(v, c)| c * v + 1.0)
                .collect();
            Tensor::from_slice(&out, x.shape(), x.device())
        },
        &x0,
        &options,
    )
    .unwrap();

    assert!(result.converged, "residual = {}", result.residual_norm);
    let sol: Vec<f64> = result.x.to_vec();
    for (s, c) in sol.iter().zip(a.iter()) {
        assert!((s - 1.0 / (1.0 - c)).abs() < 1e-6, "sol = {:?}", sol);
    }
}

type TestResult = std::result::Result<(), Box<dyn std::error::Error>>;

fn picard_options(max_iter: usize, alpha: f64) -> AndersonOptions {
    let mut options = AndersonOptions {
        m: 0,
        alpha,
        ..Default::default()
    };
    options.base.max_iter = max_iter;
    options.base.tol = 1e-12;
    options
}

macro_rules! check_picard_map_bits {
    ($runtime:ty, $client:expr, $dtype:ty, $large:expr) => {{
        use crate::optimize::roots::traits::anderson::AndersonAlgorithms;

        let client = &$client;
        let initial = Tensor::<$runtime>::from_slice(
            &[$large, -$large, $large, -$large],
            &[4],
            client.device(),
        )?;
        let expected: [$dtype; 4] = [1.0, -1.0, 0.0, -0.0];
        let mapped = Tensor::<$runtime>::from_slice(&expected, &[4], client.device())?;
        let result = AndersonAlgorithms::<$runtime>::anderson(
            client,
            |_| Ok(mapped.clone()),
            &initial,
            &picard_options(2, 1.0),
        )?;

        let custom = AndersonAlgorithms::<$runtime>::anderson_with_convergence(
            client,
            |_| Ok(mapped.clone()),
            &initial,
            &picard_options(2, 1.0),
            |_, _, fx| {
                let residual_norm = tensor_norm(client, fx)?;
                Ok(AndersonConvergence {
                    residual_norm,
                    converged: residual_norm == 0.0,
                })
            },
        )?;
        for result in [result, custom] {
            assert!(result.converged);
            assert_eq!(result.iterations, 1);
            let actual = result.x.try_to_vec::<$dtype>()?;
            for (actual, expected) in actual.iter().zip(expected) {
                assert_eq!(actual.to_bits(), expected.to_bits());
            }
        }
    }};
}

#[test]
fn undamped_picard_preserves_original_f64_map_bits() -> TestResult {
    let (_, client) = setup();
    check_picard_map_bits!(CpuRuntime, client, f64, 1e16_f64);
    Ok(())
}

#[test]
fn undamped_picard_preserves_original_f32_map_bits() -> TestResult {
    let (_, client) = setup();
    check_picard_map_bits!(CpuRuntime, client, f32, 1e8_f32);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_undamped_picard_preserves_original_f32_and_f64_map_bits() -> TestResult {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};

    let client = CudaClient::new(CudaDevice::new(0))?;
    check_picard_map_bits!(CudaRuntime, client, f32, 1e8_f32);
    check_picard_map_bits!(CudaRuntime, client, f64, 1e16_f64);
    Ok(())
}

#[cfg(feature = "wgpu")]
#[test]
fn wgpu_undamped_picard_preserves_original_f32_map_bits() -> TestResult {
    use numr::runtime::wgpu::{WgpuClient, WgpuDevice, WgpuRuntime};

    let client = WgpuClient::new(WgpuDevice::new(0))?;
    check_picard_map_bits!(WgpuRuntime, client, f32, 1e8_f32);
    Ok(())
}

#[test]
fn damped_picard_retains_residual_update_arithmetic() -> TestResult {
    let (device, client) = setup();
    let original = [1e16_f64, -3.0, 0.0];
    let target = [1.0_f64, 2.0, -1.0];
    let initial = Tensor::<CpuRuntime>::from_slice(&original, &[3], &device)?;
    let mapped = Tensor::<CpuRuntime>::from_slice(&target, &[3], &device)?;
    let result = anderson_impl(
        &client,
        |_| Ok(mapped.clone()),
        &initial,
        &picard_options(1, 0.5),
    )?;

    assert!(!result.converged);
    assert_eq!(result.iterations, 1);
    let actual = result.x.try_to_vec::<f64>()?;
    for ((actual, original), target) in actual.iter().zip(original).zip(target) {
        let expected = original + 0.5 * (target - original);
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
    Ok(())
}

#[test]
fn invalid_initial_shapes_reject_before_callback() -> TestResult {
    use std::cell::Cell;

    let (device, client) = setup();
    for shape in [vec![], vec![0], vec![2, 2]] {
        let initial = Tensor::<CpuRuntime>::zeros(&shape, DType::F64, &device)?;
        let called = Cell::new(false);
        let result = anderson_impl(
            &client,
            |x| {
                called.set(true);
                Ok(x.clone())
            },
            &initial,
            &picard_options(2, 1.0),
        );
        assert!(matches!(result, Err(OptimizeError::InvalidInput { .. })));
        assert!(!called.get());
    }
    Ok(())
}

#[test]
fn mismatched_map_shapes_reject_before_broadcasting() -> TestResult {
    use std::cell::Cell;

    let (device, client) = setup();
    let initial = Tensor::<CpuRuntime>::from_slice(&[1.0_f64, 2.0], &[2], &device)?;
    let valid = Tensor::<CpuRuntime>::from_slice(&[3.0_f64, 4.0], &[2], &device)?;
    for shape in [vec![], vec![0], vec![1], vec![1, 2]] {
        let mapped = Tensor::<CpuRuntime>::zeros(&shape, DType::F64, &device)?;
        for error_at in [0, 1] {
            let calls = Cell::new(0);
            let result = anderson_impl(
                &client,
                |_| {
                    let call = calls.get();
                    calls.set(call + 1);
                    Ok(if call == error_at {
                        mapped.clone()
                    } else {
                        valid.clone()
                    })
                },
                &initial,
                &picard_options(2, 1.0),
            );
            assert!(matches!(result, Err(OptimizeError::InvalidInput { .. })));
            assert_eq!(calls.get(), error_at + 1);
        }
    }
    Ok(())
}

#[test]
fn callback_errors_retain_initial_and_iteration_context() -> TestResult {
    use std::cell::Cell;

    let (device, client) = setup();
    let initial = Tensor::<CpuRuntime>::from_slice(&[0.0_f64], &[1], &device)?;
    let mapped = Tensor::<CpuRuntime>::from_slice(&[1.0_f64], &[1], &device)?;
    for error_at in [0, 1] {
        let calls = Cell::new(0);
        let result = anderson_impl(
            &client,
            |_| {
                let call = calls.get();
                calls.set(call + 1);
                if call == error_at {
                    return Err(numr::error::Error::InvalidArgument {
                        arg: "map",
                        reason: "declared callback error".into(),
                    });
                }
                Ok(mapped.clone())
            },
            &initial,
            &picard_options(2, 1.0),
        );
        assert!(matches!(
            result,
            Err(OptimizeError::NumericalError { ref message })
                if message.contains("declared callback error")
                    && message.contains(if error_at == 0 { "initial g eval" } else { "g eval iter 0" })
        ));
        assert_eq!(calls.get(), error_at + 1);
    }
    Ok(())
}

#[test]
fn squared_sum_reads_f32_and_f64_scalars_without_reinterpretation() -> TestResult {
    use crate::optimize::impl_generic::utils::compute_cost;

    let (device, client) = setup();
    let f32_values = Tensor::<CpuRuntime>::from_slice(&[3.0_f32, 4.0], &[2], &device)?;
    let f64_values = Tensor::<CpuRuntime>::from_slice(&[3.0_f64, 4.0], &[2], &device)?;
    for values in [&f32_values, &f64_values] {
        assert_eq!(tensor_norm(&client, values)?, 5.0);
        assert_eq!(compute_cost(&client, values)?, 25.0);
    }
    Ok(())
}

#[test]
fn nonscalar_squared_sum_returns_an_error() -> TestResult {
    use crate::optimize::impl_generic::utils::compute_cost;

    let (device, client) = setup();
    let values = Tensor::<CpuRuntime>::from_slice(&[1.0_f64, 2.0, 3.0, 4.0], &[2, 2], &device)?;
    assert!(tensor_norm(&client, &values).is_err());
    assert!(compute_cost(&client, &values).is_err());
    Ok(())
}

#[path = "anderson_convergence_tests.rs"]
mod convergence_tests;
