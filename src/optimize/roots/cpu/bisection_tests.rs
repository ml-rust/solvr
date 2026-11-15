//! CPU coverage for independent bracketed roots.

use super::*;
use crate::DType;
use crate::optimize::OptimizeError;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, ScalarOps, TypeConversionOps, UnaryOps, UtilityOps,
};
use numr::runtime::cpu::CpuDevice;
use std::cell::RefCell;

fn setup() -> (CpuDevice, CpuClient) {
    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());
    (device, client)
}

fn tensor(
    client: &CpuClient,
    device: &CpuDevice,
    values: &[f64],
    dtype: DType,
) -> Tensor<CpuRuntime> {
    client
        .cast(
            &Tensor::from_slice(values, &[values.len()], device).unwrap(),
            dtype,
        )
        .unwrap()
}

#[test]
fn signed_functions_support_f32_and_f64() {
    let (device, client) = setup();
    for dtype in [DType::F32, DType::F64] {
        for sign in [1.0, -1.0] {
            let lower = tensor(&client, &device, &[-3.0, 0.0], dtype);
            let upper = tensor(&client, &device, &[0.0, 3.0], dtype);
            let targets = tensor(&client, &device, &[-2.0, 2.0], dtype);
            let result = client
                .bisect_batched(
                    |x| client.mul_scalar(&client.sub(x, &targets)?, sign),
                    &lower,
                    &upper,
                    &BatchedBisectionOptions::default(),
                )
                .unwrap();
            let lo: Vec<f64> = client.cast(&result.lower, DType::F64).unwrap().to_vec();
            let hi: Vec<f64> = client.cast(&result.upper, DType::F64).unwrap().to_vec();
            for (index, root) in [-2.0, 2.0].into_iter().enumerate() {
                assert!(lo[index] <= root && root <= hi[index]);
                assert!(hi[index] - lo[index] <= 3e-6);
            }
            assert!(result.iterations > 0);
            assert_eq!(result.expansions, 0);
        }
    }
}

#[test]
fn expansion_preserves_lower_and_collapses_exact_upper() {
    let (device, client) = setup();
    let lower = tensor(&client, &device, &[-5.0, 1.0], DType::F64);
    let upper = tensor(&client, &device, &[-4.0, 2.0], DType::F64);
    let target = tensor(&client, &device, &[3.0, 1.0], DType::F64);
    let calls = RefCell::new(Vec::<Vec<f64>>::new());
    let result = client
        .bisect_batched(
            |x| {
                calls.borrow_mut().push(x.to_vec());
                client.sub(x, &target)
            },
            &lower,
            &upper,
            &BatchedBisectionOptions {
                max_expansions: 3,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(result.expansions, 3);
    assert_eq!(result.iterations, 0);
    assert_eq!(result.lower.to_vec::<f64>(), vec![3.0, 1.0]);
    assert_eq!(result.upper.to_vec::<f64>(), vec![3.0, 1.0]);
    assert_eq!(result.f_lower.to_vec::<f64>(), vec![0.0, 0.0]);
    assert_eq!(result.f_upper.to_vec::<f64>(), vec![0.0, 0.0]);
    assert_eq!(
        *calls.borrow(),
        vec![
            vec![-5.0, 1.0],
            vec![-4.0, 1.0],
            vec![-3.0, 1.0],
            vec![-1.0, 1.0],
            vec![3.0, 1.0]
        ]
    );
}

#[test]
fn heterogeneous_lanes_freeze_after_width_convergence() {
    let (device, client) = setup();
    let lower = tensor(&client, &device, &[1.0, 0.0], DType::F64);
    let upper = tensor(&client, &device, &[1.0 + 1e-8, 3.0], DType::F64);
    let targets = tensor(&client, &device, &[1.0 + 5e-9, 2.0], DType::F64);
    let result = client
        .bisect_batched(
            |x| client.sub(x, &targets),
            &lower,
            &upper,
            &Default::default(),
        )
        .unwrap();
    assert_eq!(result.lower.to_vec::<f64>()[0], 1.0);
    assert_eq!(result.upper.to_vec::<f64>()[0], 1.0 + 1e-8);
    assert_eq!(result.f_lower.to_vec::<f64>()[0], 1.0 - (1.0 + 5e-9));
    assert!(result.iterations > 0);
}

#[test]
fn degenerate_intervals_require_exact_roots() {
    let (device, client) = setup();
    let point = tensor(&client, &device, &[2.0], DType::F64);
    let result = client
        .bisect_batched(
            |x| client.sub_scalar(x, 2.0),
            &point,
            &point,
            &Default::default(),
        )
        .unwrap();
    assert_eq!(result.iterations, 0);
    assert_eq!(result.lower.to_vec::<f64>(), vec![2.0]);
    assert!(
        client
            .bisect_batched(|x| Ok(x.clone()), &point, &point, &Default::default())
            .is_err()
    );
}

#[test]
fn lower_and_upper_endpoint_roots_collapse() {
    let (device, client) = setup();
    let lower = tensor(&client, &device, &[0.0, -2.0], DType::F64);
    let upper = tensor(&client, &device, &[3.0, 0.0], DType::F64);
    let result = client
        .bisect_batched(|x| Ok(x.clone()), &lower, &upper, &Default::default())
        .unwrap();
    assert_eq!(result.lower.to_vec::<f64>(), vec![0.0, 0.0]);
    assert_eq!(result.upper.to_vec::<f64>(), vec![0.0, 0.0]);
    assert_eq!(result.iterations, 0);
}

#[test]
fn same_sign_brackets_obey_expansion_limits() {
    let (device, client) = setup();
    let lower = tensor(&client, &device, &[0.0], DType::F64);
    let upper = tensor(&client, &device, &[1.0], DType::F64);
    for max_expansions in [0, 2] {
        let options = BatchedBisectionOptions {
            max_expansions,
            ..Default::default()
        };
        assert!(
            client
                .bisect_batched(|x| client.sub_scalar(x, 10.0), &lower, &upper, &options)
                .is_err()
        );
    }
}

#[test]
fn malformed_bounds_and_callback_outputs_return_errors() {
    let (device, client) = setup();
    let lower = tensor(&client, &device, &[0.0, 0.0], DType::F64);
    let upper = tensor(&client, &device, &[1.0, 1.0], DType::F64);
    let options = BatchedBisectionOptions::default();
    let reshaped = upper.reshape(&[2, 1]).unwrap();
    let wrong_dtype = client.cast(&upper, DType::F32).unwrap();
    let integers = client.cast(&upper, DType::I32).unwrap();
    assert!(
        client
            .bisect_batched(|x| Ok(x.clone()), &lower, &reshaped, &options)
            .is_err()
    );
    assert!(
        client
            .bisect_batched(|x| Ok(x.clone()), &lower, &wrong_dtype, &options)
            .is_err()
    );
    assert!(
        client
            .bisect_batched(|x| Ok(x.clone()), &integers, &integers, &options)
            .is_err()
    );
    assert!(
        client
            .bisect_batched(|x| Ok(x.clone()), &upper, &lower, &options)
            .is_err()
    );
    assert!(
        client
            .bisect_batched(|x| x.reshape(&[2, 1]), &lower, &upper, &options)
            .is_err()
    );
    assert!(
        client
            .bisect_batched(|x| client.cast(x, DType::F32), &lower, &upper, &options)
            .is_err()
    );
}

#[test]
fn nonfinite_bounds_and_residuals_return_errors() {
    let (device, client) = setup();
    let lower = tensor(&client, &device, &[0.0], DType::F64);
    let upper = tensor(&client, &device, &[1.0], DType::F64);
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let nonfinite = tensor(&client, &device, &[value], DType::F64);
        assert!(
            client
                .bisect_batched(|x| Ok(x.clone()), &lower, &nonfinite, &Default::default())
                .is_err()
        );
        assert!(
            client
                .bisect_batched(
                    |_| Ok(nonfinite.clone()),
                    &lower,
                    &upper,
                    &Default::default()
                )
                .is_err()
        );
        assert!(
            client
                .bisect_batched(
                    |x| {
                        let value: f64 = x.item()?;
                        client.fill(
                            x.shape(),
                            if value == 0.5 { f64::NAN } else { value - 0.25 },
                            x.dtype(),
                        )
                    },
                    &lower,
                    &upper,
                    &Default::default()
                )
                .is_err()
        );
    }
}

#[test]
fn invalid_options_return_errors() {
    let (device, client) = setup();
    let lower = tensor(&client, &device, &[-1.0], DType::F64);
    let upper = tensor(&client, &device, &[1.0], DType::F64);
    let mut cases = vec![
        BatchedBisectionOptions {
            atol: 0.0,
            rtol: 0.0,
            ..Default::default()
        },
        BatchedBisectionOptions {
            max_iter: 0,
            ..Default::default()
        },
        BatchedBisectionOptions {
            expansion_factor: 1.0,
            ..Default::default()
        },
    ];
    for value in [f64::NAN, f64::INFINITY, -1.0] {
        cases.push(BatchedBisectionOptions {
            atol: value,
            ..Default::default()
        });
        cases.push(BatchedBisectionOptions {
            rtol: value,
            ..Default::default()
        });
        cases.push(BatchedBisectionOptions {
            expansion_factor: value,
            ..Default::default()
        });
    }
    for options in cases {
        assert!(
            client
                .bisect_batched(|x| Ok(x.clone()), &lower, &upper, &options)
                .is_err()
        );
    }
}

#[test]
fn iteration_limit_returns_did_not_converge() {
    let (device, client) = setup();
    let lower = tensor(&client, &device, &[0.0], DType::F64);
    let upper = tensor(&client, &device, &[2.0], DType::F64);
    let options = BatchedBisectionOptions {
        max_iter: 1,
        ..Default::default()
    };
    let error = client
        .bisect_batched(
            |x| client.sub_scalar(&client.square(x)?, 2.0),
            &lower,
            &upper,
            &options,
        )
        .unwrap_err();
    assert!(matches!(
        error,
        OptimizeError::DidNotConverge { iterations: 1, .. }
    ));
}

#[test]
fn adjacent_f32_endpoints_report_unrepresentable_tolerance() {
    let (device, client) = setup();
    let lower = tensor(&client, &device, &[1.0], DType::F32);
    let upper = tensor(
        &client,
        &device,
        &[f32::from_bits(1.0f32.to_bits() + 1) as f64],
        DType::F32,
    );
    let options = BatchedBisectionOptions {
        atol: f64::from_bits(1),
        rtol: 0.0,
        ..Default::default()
    };
    let error = client
        .bisect_batched(
            |x| {
                client.where_cond(
                    &client.eq(x, &lower)?,
                    &client.fill(&[1], -1.0, DType::F32)?,
                    &client.fill(&[1], 1.0, DType::F32)?,
                )
            },
            &lower,
            &upper,
            &options,
        )
        .unwrap_err();
    assert!(matches!(error, OptimizeError::NumericalError { .. }));
}

#[test]
fn expansion_and_width_overflow_return_errors() {
    let (device, client) = setup();
    let options = BatchedBisectionOptions {
        max_expansions: 3,
        ..Default::default()
    };
    let lower = tensor(&client, &device, &[0.0], DType::F64);
    let upper = tensor(&client, &device, &[f64::MAX], DType::F64);
    assert!(
        client
            .bisect_batched(
                |x| client.fill(x.shape(), -1.0, x.dtype()),
                &lower,
                &upper,
                &options
            )
            .is_err()
    );
    let lower = tensor(&client, &device, &[-f64::MAX], DType::F64);
    assert!(
        client
            .bisect_batched(|x| client.sign(x), &lower, &upper, &options)
            .is_err()
    );
}

#[test]
fn nonlinear_signed_roots_retain_brackets_and_residuals() {
    let (device, client) = setup();
    for dtype in [DType::F32, DType::F64] {
        for sign in [1.0, -1.0] {
            let lower = tensor(&client, &device, &[0.0, 0.0], dtype);
            let upper = tensor(&client, &device, &[2.0, 3.0], dtype);
            let targets = tensor(&client, &device, &[2.0, 3.0], dtype);
            let result = client
                .bisect_batched(
                    |x| client.mul_scalar(&client.sub(&client.square(x)?, &targets)?, sign),
                    &lower,
                    &upper,
                    &Default::default(),
                )
                .unwrap();
            let lo = client
                .cast(&result.lower, DType::F64)
                .unwrap()
                .to_vec::<f64>();
            let hi = client
                .cast(&result.upper, DType::F64)
                .unwrap()
                .to_vec::<f64>();
            let flo = client
                .cast(&result.f_lower, DType::F64)
                .unwrap()
                .to_vec::<f64>();
            let fhi = client
                .cast(&result.f_upper, DType::F64)
                .unwrap()
                .to_vec::<f64>();
            for index in 0..2 {
                let root = (index as f64 + 2.0).sqrt();
                assert!((lo[index] - root).abs() < 4e-6);
                assert!((hi[index] - root).abs() < 4e-6);
                assert!(hi[index] - lo[index] <= 4e-6);
                assert!(flo[index] * sign <= 0.0);
                assert!(fhi[index] * sign >= 0.0);
            }
        }
    }
}
