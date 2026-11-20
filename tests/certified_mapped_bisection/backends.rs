use super::fixture::Result;
use numr::prelude::*;
#[cfg(any(feature = "cuda", feature = "wgpu"))]
use solvr::optimize::roots::{CertifiedBisectionOptions, CertifiedMappedBisectionAlgorithms};
#[cfg(any(feature = "cuda", feature = "wgpu"))]
use std::cell::Cell;

#[test]
fn configured_cpu_mapped_roots_preserve_continuation_and_certificates() -> Result {
    use numr::runtime::cpu::ParallelismConfig;
    let device = CpuDevice::new();
    let original = CpuRuntime::default_client(&device);
    let before = original.parallelism();
    let config = ParallelismConfig::new(Some(2), Some(257));
    let client = original.with_parallelism(config);
    super::behavior::run::<CpuRuntime, _>(&client)?;
    super::resume::run::<CpuRuntime, _>(&client)?;
    super::ownership::run::<CpuRuntime, _>(&client)?;
    super::admission::run::<CpuRuntime, _>(&client)?;
    super::tolerance::run::<CpuRuntime, _>(&client)?;
    assert_eq!(client.parallelism(), config);
    assert_eq!(original.parallelism(), before);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn actual_cuda_mapped_roots_preserve_continuation_and_certificates() -> Result {
    let client = CudaClient::new(CudaDevice::new(0))?;
    super::behavior::run::<CudaRuntime, _>(&client)?;
    super::resume::run::<CudaRuntime, _>(&client)?;
    super::ownership::run::<CudaRuntime, _>(&client)?;
    super::admission::run::<CudaRuntime, _>(&client)?;
    super::tolerance::run::<CudaRuntime, _>(&client)?;
    let foreign = Tensor::<CudaRuntime>::empty(&[0], DType::F32, &CudaDevice::new(1))?;
    super::tolerance::foreign_device(&client, &foreign)?;
    let called = Cell::new(false);
    assert!(
        client
            .bisect_certified_mapped(
                |_| {
                    called.set(true);
                    Err(super::fixture::sentinel())
                },
                |_| {
                    called.set(true);
                    Err(super::fixture::sentinel())
                },
                &foreign,
                &foreign,
                &foreign,
                &CertifiedBisectionOptions {
                    max_points: 0,
                    ..Default::default()
                },
            )
            .is_err()
    );
    assert!(!called.get());
    Ok(())
}

#[test]
fn configured_cpu_slack_secant_continuations_keep_probe_state() -> Result {
    use numr::runtime::cpu::ParallelismConfig;
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device)
        .with_parallelism(ParallelismConfig::new(Some(2), Some(257)));
    super::slack_secant::run::<CpuRuntime, _>(&client)
}

#[cfg(feature = "cuda")]
#[test]
fn actual_cuda_slack_secant_continuations_match_cpu_bits() -> Result {
    let cpu = CpuRuntime::default_client(&CpuDevice::new());
    let cuda = CudaClient::new(CudaDevice::new(0))?;
    super::slack_secant::run::<CudaRuntime, _>(&cuda)?;
    for dtype in [DType::F32, DType::F64] {
        let expected = super::slack_secant::trace::<CpuRuntime, _>(&cpu, dtype)?;
        let actual = super::slack_secant::trace::<CudaRuntime, _>(&cuda, dtype)?;
        assert_eq!(expected, actual, "{dtype:?}");
    }
    Ok(())
}

#[cfg(feature = "wgpu")]
#[test]
fn actual_wgpu_mapped_roots_retain_arithmetic_qualification() -> Result {
    let device = WgpuDevice::new(0);
    let client = WgpuClient::new(device.clone())?;
    for dtype in [DType::F32, DType::F64] {
        let empty = Tensor::<WgpuRuntime>::empty(&[0], dtype, &device)?;
        let called = Cell::new(false);
        assert!(
            client
                .bisect_certified_mapped(
                    |_| {
                        called.set(true);
                        Err(super::fixture::sentinel())
                    },
                    |_| {
                        called.set(true);
                        Err(super::fixture::sentinel())
                    },
                    &empty,
                    &empty,
                    &empty,
                    &CertifiedBisectionOptions {
                        max_points: 0,
                        ..Default::default()
                    },
                )
                .is_err()
        );
        assert!(!called.get());
    }
    Ok(())
}
