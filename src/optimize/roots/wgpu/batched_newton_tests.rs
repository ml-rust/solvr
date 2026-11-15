//! Wgpu parity for strict batched Newton lane semantics.

use super::*;
use numr::ops::{BinaryOps, ScalarOps, UnaryOps};
use numr::runtime::wgpu::WgpuDevice;

#[test]
fn batched_newton_matches_analytic_roots_and_lane_isolation() {
    let device = WgpuDevice::new(0);
    let Ok(client) = WgpuClient::new(device.clone()) else {
        eprintln!("Skipping wgpu batched Newton: no device");
        return;
    };
    let initial = Tensor::<WgpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device).unwrap();
    let targets = Tensor::<WgpuRuntime>::from_slice(&[2.0f32, 3.0], &[2], &device).unwrap();
    let result = client
        .newton_batched(
            |x| {
                client.div(
                    &client.sub(&client.square(x)?, &targets)?,
                    &client.mul_scalar(x, 2.0)?,
                )
            },
            &initial,
            &BatchedNewtonOptions {
                correction_tolerance: 1e-6,
                max_iter: 20,
            },
        )
        .unwrap();
    let values = result.x.to_vec::<f32>();
    assert!((values[0] - 2.0f32.sqrt()).abs() < 2e-6);
    assert!((values[1] - 3.0f32.sqrt()).abs() < 2e-6);
    assert_eq!(result.converged.to_vec::<f32>(), vec![1.0, 1.0]);
    assert_eq!(result.nonfinite.to_vec::<f32>(), vec![0.0, 0.0]);
    let corrections = Tensor::<WgpuRuntime>::from_slice(&[f32::NAN, 0.0], &[2], &device).unwrap();
    let result = client
        .newton_batched(|_| Ok(corrections.clone()), &initial, &Default::default())
        .unwrap();
    assert_eq!(result.x.to_vec::<f32>(), vec![1.0, 2.0]);
    assert_eq!(result.converged.to_vec::<f32>(), vec![0.0, 1.0]);
    assert_eq!(result.nonfinite.to_vec::<f32>(), vec![1.0, 0.0]);
    assert!(result.correction.to_vec::<f32>()[0].is_nan());
    let corrections = Tensor::<WgpuRuntime>::from_slice(&[0.5f32, 0.25], &[2], &device).unwrap();
    let result = client
        .newton_batched(
            |_| Ok(corrections.clone()),
            &initial,
            &BatchedNewtonOptions {
                correction_tolerance: 0.5,
                max_iter: 1,
            },
        )
        .unwrap();
    assert_eq!(result.x.to_vec::<f32>(), vec![0.5, 2.0]);
    assert_eq!(result.converged.to_vec::<f32>(), vec![0.0, 1.0]);
}

#[test]
fn batched_newton_rejects_f64_before_callback() {
    let device = WgpuDevice::new(0);
    let Ok(client) = WgpuClient::new(device.clone()) else {
        eprintln!("Skipping wgpu batched Newton F64 admission: no device");
        return;
    };
    let initial = Tensor::<WgpuRuntime>::from_slice(&[1.0f64], &[1], &device).unwrap();
    assert!(
        client
            .newton_batched(|x| Ok(x.clone()), &initial, &Default::default())
            .is_err()
    );
}
