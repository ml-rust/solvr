//! Cuda parity for strict batched Newton lane semantics.

use super::*;
use numr::ops::{BinaryOps, ScalarOps, UnaryOps};
use numr::runtime::cuda::CudaDevice;

#[test]
fn batched_newton_matches_analytic_roots_and_lane_isolation() {
    let device = CudaDevice::new(0);
    let Ok(client) = CudaClient::new(device.clone()) else {
        eprintln!("Skipping cuda batched Newton: no device");
        return;
    };
    let initial = Tensor::<CudaRuntime>::from_slice(&[1.0f64, 2.0], &[2], &device).unwrap();
    let targets = Tensor::<CudaRuntime>::from_slice(&[2.0f64, 3.0], &[2], &device).unwrap();
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
    let values = result.x.to_vec::<f64>();
    assert!((values[0] - 2.0f64.sqrt()).abs() < 2e-6);
    assert!((values[1] - 3.0f64.sqrt()).abs() < 2e-6);
    assert_eq!(result.converged.to_vec::<f64>(), vec![1.0, 1.0]);
    assert_eq!(result.nonfinite.to_vec::<f64>(), vec![0.0, 0.0]);
    let corrections = Tensor::<CudaRuntime>::from_slice(&[f64::NAN, 0.0], &[2], &device).unwrap();
    let result = client
        .newton_batched(|_| Ok(corrections.clone()), &initial, &Default::default())
        .unwrap();
    assert_eq!(result.x.to_vec::<f64>(), vec![1.0, 2.0]);
    assert_eq!(result.converged.to_vec::<f64>(), vec![0.0, 1.0]);
    assert_eq!(result.nonfinite.to_vec::<f64>(), vec![1.0, 0.0]);
    assert!(result.correction.to_vec::<f64>()[0].is_nan());
    let corrections = Tensor::<CudaRuntime>::from_slice(&[0.5f64, 0.25], &[2], &device).unwrap();
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
    assert_eq!(result.x.to_vec::<f64>(), vec![0.5, 2.0]);
    assert_eq!(result.converged.to_vec::<f64>(), vec![0.0, 1.0]);
}
