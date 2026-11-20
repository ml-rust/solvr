#[cfg(any(feature = "cuda", feature = "wgpu"))]
use super::fixture::{self, Result};
#[cfg(any(feature = "cuda", feature = "wgpu"))]
use numr::prelude::*;
#[cfg(any(feature = "cuda", feature = "wgpu"))]
use solvr::optimize::roots::{CertifiedBisectionAlgorithms, CertifiedBisectionOptions};
#[cfg(any(feature = "cuda", feature = "wgpu"))]
use std::cell::Cell;

#[cfg(feature = "cuda")]
#[test]
fn actual_cuda_roots_contain_all_independent_references() -> Result {
    let device = CudaDevice::new(0);
    let client = CudaClient::new(device)?;
    super::suite::run::<CudaRuntime, _>(&client, false)?;
    super::suite::run::<CudaRuntime, _>(&client, true)?;
    super::admission::run::<CudaRuntime, _>(&client)?;
    super::budget::run::<CudaRuntime, _>(&client)?;
    super::ownership::run::<CudaRuntime, _>(&client)?;
    let foreign = Tensor::<CudaRuntime>::empty(&[0], DType::F32, &CudaDevice::new(1))?;
    let called = Cell::new(false);
    assert!(
        client
            .bisect_certified(
                |_| {
                    called.set(true);
                    Err(fixture::sentinel())
                },
                &foreign,
                &foreign,
                &foreign,
                &CertifiedBisectionOptions::default()
            )
            .is_err()
    );
    assert!(!called.get());
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn actual_cuda_slack_secant_matches_cpu_bits() -> Result {
    let cpu = CpuRuntime::default_client(&CpuDevice::new());
    let cuda = CudaClient::new(CudaDevice::new(0))?;
    super::slack_secant::run::<CudaRuntime, _>(&cuda)?;
    let expected = super::slack_secant::trace::<CpuRuntime, _>(&cpu)?;
    let actual = super::slack_secant::trace::<CudaRuntime, _>(&cuda)?;
    assert_eq!(expected.len(), actual.len());
    for (index, (expected, actual)) in expected.iter().zip(&actual).enumerate() {
        assert_eq!(expected.bits, actual.bits, "trace {index} bits");
        assert_eq!(expected, actual, "trace {index}");
    }
    Ok(())
}

#[cfg(feature = "wgpu")]
#[test]
fn actual_wgpu_rejects_undeclared_arithmetic_before_callback_work() -> Result {
    let device = WgpuDevice::new(0);
    let client = WgpuClient::new(device.clone())?;
    for dtype in [DType::F32, DType::F64] {
        let empty = Tensor::<WgpuRuntime>::empty(&[0], dtype, &device)?;
        let called = Cell::new(false);
        assert!(
            client
                .bisect_certified(
                    |_| {
                        called.set(true);
                        Err(fixture::sentinel())
                    },
                    &empty,
                    &empty,
                    &empty,
                    &CertifiedBisectionOptions::default()
                )
                .is_err()
        );
        assert!(!called.get());
    }
    let lower = Tensor::<WgpuRuntime>::from_slice(&[0.0f32], &[1], &device)?;
    let upper = Tensor::<WgpuRuntime>::from_slice(&[1.0f32], &[1], &device)?;
    let called = Cell::new(false);
    assert!(
        client
            .bisect_certified(
                |_| {
                    called.set(true);
                    Err(fixture::sentinel())
                },
                &lower,
                &upper,
                &upper,
                &CertifiedBisectionOptions::default()
            )
            .is_err()
    );
    assert!(!called.get());
    Ok(())
}
