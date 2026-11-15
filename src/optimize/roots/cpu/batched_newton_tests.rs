//! CPU coverage for strict independent correction iterations.

use super::*;
use crate::DType;
use numr::ops::{BinaryOps, ScalarOps, TypeConversionOps, UnaryOps, UtilityOps};
use numr::runtime::cpu::{CpuDevice, ParallelismConfig};
use std::cell::Cell;

fn setup() -> (CpuDevice, CpuClient) {
    let device = CpuDevice::new();
    let client =
        CpuClient::new(device.clone()).with_parallelism(ParallelismConfig::new(Some(1), Some(1)));
    (device, client)
}

fn tensor(device: &CpuDevice, values: &[f64]) -> Tensor<CpuRuntime> {
    Tensor::from_slice(values, &[values.len()], device).unwrap()
}

fn options(tolerance: f64, max_iter: usize) -> BatchedNewtonOptions {
    BatchedNewtonOptions {
        correction_tolerance: tolerance,
        max_iter,
    }
}

#[test]
fn analytic_roots_support_both_dtypes_and_configured_client() {
    let (device, client) = setup();
    for dtype in [DType::F32, DType::F64] {
        let initial = client.cast(&tensor(&device, &[1.0, 2.0]), dtype).unwrap();
        let targets = client.cast(&tensor(&device, &[2.0, 3.0]), dtype).unwrap();
        let result = client
            .newton_batched(
                |x| {
                    client.div(
                        &client.sub(&client.square(x)?, &targets)?,
                        &client.mul_scalar(x, 2.0)?,
                    )
                },
                &initial,
                &options(1e-6, 20),
            )
            .unwrap();
        let values = client.cast(&result.x, DType::F64).unwrap().to_vec::<f64>();
        assert!((values[0] - 2.0f64.sqrt()).abs() < 2e-6);
        assert!((values[1] - 3.0f64.sqrt()).abs() < 2e-6);
        assert_eq!(result.x.dtype(), dtype);
        assert_eq!(
            client
                .cast(&result.converged, DType::F64)
                .unwrap()
                .to_vec::<f64>(),
            vec![1.0, 1.0]
        );
        assert_eq!(
            client
                .cast(&result.nonfinite, DType::F64)
                .unwrap()
                .to_vec::<f64>(),
            vec![0.0, 0.0]
        );
    }
}

#[test]
fn strict_equality_updates_and_small_correction_stops_before_update() {
    let (device, client) = setup();
    let initial = tensor(&device, &[4.0, 4.0]);
    let corrections = tensor(&device, &[0.5, 0.25]);
    let result = client
        .newton_batched(|_| Ok(corrections.clone()), &initial, &options(0.5, 1))
        .unwrap();
    assert_eq!(result.x.to_vec::<f64>(), vec![3.5, 4.0]);
    assert_eq!(result.converged.to_vec::<f64>(), vec![0.0, 1.0]);
    assert_eq!(result.correction.to_vec::<f64>(), vec![0.5, 0.25]);
    assert_eq!(result.iterations, 1);
}

#[test]
fn mixed_lanes_freeze_state_and_last_correction_independently() {
    let (device, client) = setup();
    let calls = Cell::new(0);
    let initial = tensor(&device, &[10.0, 20.0, 30.0, 40.0]);
    let result = client
        .newton_batched(
            |_| {
                let n = calls.get();
                calls.set(n + 1);
                Ok(tensor(
                    &device,
                    if n == 0 {
                        &[0.01, f64::NAN, 1.0, f64::INFINITY]
                    } else {
                        &[f64::NAN, 5.0, 0.02, 5.0]
                    },
                ))
            },
            &initial,
            &options(0.1, 4),
        )
        .unwrap();
    assert_eq!(result.x.to_vec::<f64>(), vec![10.0, 20.0, 29.0, 40.0]);
    assert_eq!(result.converged.to_vec::<f64>(), vec![1.0, 0.0, 1.0, 0.0]);
    assert_eq!(result.nonfinite.to_vec::<f64>(), vec![0.0, 1.0, 0.0, 1.0]);
    let correction = result.correction.to_vec::<f64>();
    assert_eq!(correction[0], 0.01);
    assert!(correction[1].is_nan());
    assert_eq!(correction[2], 0.02);
    assert_eq!(correction[3], f64::INFINITY);
    assert_eq!(result.iterations, 2);
}

#[test]
fn last_evaluation_converges_but_exhaustion_returns_masks_zero() {
    let (device, client) = setup();
    let initial = tensor(&device, &[3.0]);
    for limit in [1, 2] {
        let result = client
            .newton_batched(
                |x| client.sub_scalar(x, 2.0),
                &initial,
                &options(0.1, limit),
            )
            .unwrap();
        assert_eq!(result.x.to_vec::<f64>(), vec![2.0]);
        assert_eq!(
            result.converged.to_vec::<f64>(),
            vec![if limit == 2 { 1.0 } else { 0.0 }]
        );
        assert_eq!(result.nonfinite.to_vec::<f64>(), vec![0.0]);
        assert_eq!(result.iterations, limit);
    }
}

#[test]
fn candidate_overflow_retains_last_finite_iterate() {
    let (device, client) = setup();
    let initial = tensor(&device, &[f64::MAX, 3.0]);
    let correction = tensor(&device, &[-f64::MAX, 0.0]);
    let result = client
        .newton_batched(|_| Ok(correction.clone()), &initial, &options(0.1, 5))
        .unwrap();
    assert_eq!(result.x.to_vec::<f64>(), vec![f64::MAX, 3.0]);
    assert_eq!(result.nonfinite.to_vec::<f64>(), vec![1.0, 0.0]);
    assert_eq!(result.converged.to_vec::<f64>(), vec![0.0, 1.0]);
    assert_eq!(result.correction.to_vec::<f64>(), vec![-f64::MAX, 0.0]);
}

#[test]
fn rounded_stagnation_never_implies_convergence() {
    let (device, client) = setup();
    for dtype in [DType::F32, DType::F64] {
        let initial = client.cast(&tensor(&device, &[1e20]), dtype).unwrap();
        let result = client
            .newton_batched(
                |x| client.fill(x.shape(), 1.0, x.dtype()),
                &initial,
                &options(0.5, 3),
            )
            .unwrap();
        assert_eq!(
            client.cast(&result.x, DType::F64).unwrap().to_vec::<f64>(),
            client.cast(&initial, DType::F64).unwrap().to_vec::<f64>()
        );
        assert_eq!(result.iterations, 3);
        assert_eq!(
            client
                .cast(&result.converged, DType::F64)
                .unwrap()
                .to_vec::<f64>(),
            vec![0.0]
        );
    }
}

#[test]
fn scalar_multidimensional_and_noncontiguous_shapes_are_preserved() {
    let (device, client) = setup();
    let matrix = tensor(&device, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .reshape(&[2, 3])
        .unwrap();
    let transposed = matrix.transpose(0, 1).unwrap();
    let scalar = tensor(&device, &[2.0]).reshape(&[]).unwrap();
    for initial in [matrix, transposed, scalar] {
        let result = client
            .newton_batched(|x| client.sub_scalar(x, 2.0), &initial, &options(1e-6, 2))
            .unwrap();
        for value in [
            &result.x,
            &result.correction,
            &result.converged,
            &result.nonfinite,
        ] {
            assert_eq!(value.shape(), initial.shape());
        }
        assert!(result.x.to_vec::<f64>().iter().all(|&x| x == 2.0));
        assert!(result.converged.to_vec::<f64>().iter().all(|&x| x == 1.0));
    }
}

#[test]
fn invalid_initial_and_options_return_errors_before_callback() {
    let (device, client) = setup();
    let initial = tensor(&device, &[1.0]);
    let callback = |_: &Tensor<CpuRuntime>| -> numr::error::Result<Tensor<CpuRuntime>> {
        panic!("invalid inputs reached callback")
    };
    for value in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(
            client
                .newton_batched(callback, &initial, &options(value, 1))
                .is_err()
        );
    }
    assert!(
        client
            .newton_batched(callback, &initial, &options(1e-6, 0))
            .is_err()
    );
    let f32_initial = client.cast(&initial, DType::F32).unwrap();
    for value in [f64::from_bits(1), f64::MAX] {
        assert!(
            client
                .newton_batched(callback, &f32_initial, &options(value, 1))
                .is_err()
        );
    }
    for values in [
        vec![f64::NAN],
        vec![f64::INFINITY],
        vec![f64::NEG_INFINITY],
        vec![],
    ] {
        assert!(
            client
                .newton_batched(callback, &tensor(&device, &values), &Default::default())
                .is_err()
        );
    }
    let integers = client.cast(&initial, DType::I32).unwrap();
    assert!(
        client
            .newton_batched(callback, &integers, &Default::default())
            .is_err()
    );
}

#[test]
fn callback_errors_and_metadata_errors_propagate() {
    let (device, client) = setup();
    let initial = tensor(&device, &[1.0, 2.0]);
    assert!(
        client
            .newton_batched(|x| x.reshape(&[2, 1]), &initial, &Default::default())
            .is_err()
    );
    assert!(
        client
            .newton_batched(
                |x| client.cast(x, DType::F32),
                &initial,
                &Default::default()
            )
            .is_err()
    );
    // Invalid reshape returns the original numr callback error.
    let error = client
        .newton_batched(|x| x.reshape(&[3]), &initial, &Default::default())
        .unwrap_err();
    assert!(matches!(
        error,
        crate::optimize::OptimizeError::NumrError(_)
    ));
}
