use super::fixture::Result;
use numr::prelude::*;

fn run<R, C>(client: &C) -> Result
where
    R: Runtime<DType = DType>,
    C: super::fixture::Client<R> + solvr::optimize::roots::CertifiedUpperBracketAlgorithms<R>,
{
    super::search::run(client)?;
    super::budget::run(client)?;
    super::admission::run(client)?;
    super::ownership::run(client)
}

#[test]
fn configured_cpu_upper_brackets_preserve_physical_roots() -> Result {
    use numr::runtime::cpu::ParallelismConfig;
    let device = CpuDevice::new();
    let original = CpuRuntime::default_client(&device);
    let original_config = original.parallelism();
    let config = ParallelismConfig::new(Some(2), Some(257));
    let client = original.with_parallelism(config);
    run::<CpuRuntime, _>(&client)?;
    assert_eq!(client.parallelism(), config);
    assert_eq!(original.parallelism(), original_config);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn actual_cuda_upper_brackets_preserve_physical_roots() -> Result {
    use solvr::optimize::roots::{CertifiedUpperBracketAlgorithms, CertifiedUpperBracketOptions};
    use std::cell::Cell;
    let client = CudaClient::new(CudaDevice::new(0))?;
    run::<CudaRuntime, _>(&client)?;
    let foreign = Tensor::<CudaRuntime>::empty(&[0], DType::F32, &CudaDevice::new(1))?;
    let called = Cell::new(false);
    let result = client.bracket_upper_certified(
        |_| {
            called.set(true);
            Err(super::fixture::sentinel())
        },
        &foreign,
        &foreign,
        &foreign,
        &CertifiedUpperBracketOptions::default(),
    );
    assert!(result.is_err());
    assert!(!called.get());
    Ok(())
}

#[cfg(feature = "wgpu")]
#[test]
fn actual_wgpu_rejects_uncertified_provider_before_callback() -> Result {
    use solvr::optimize::roots::{CertifiedUpperBracketAlgorithms, CertifiedUpperBracketOptions};
    use std::cell::Cell;
    let device = WgpuDevice::new(0);
    let client = WgpuClient::new(device.clone())?;
    for dtype in [DType::F32, DType::F64] {
        for count in [0, 1] {
            let value = Tensor::<WgpuRuntime>::zeros(&[count], dtype, &device)?;
            let called = Cell::new(false);
            let result = client.bracket_upper_certified(
                |_| {
                    called.set(true);
                    Err(super::fixture::sentinel())
                },
                &value,
                &value,
                &value,
                &CertifiedUpperBracketOptions::default(),
            );
            assert!(result.is_err());
            assert!(!called.get());
        }
    }
    Ok(())
}
