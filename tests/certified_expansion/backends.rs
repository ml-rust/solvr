use super::fixture::Result;
use numr::prelude::*;
use numr::runtime::cpu::ParallelismConfig;

#[test]
fn configured_cpu_expansions_retain_independent_residuals_and_owned_inputs() -> Result {
    let device = CpuDevice::new();
    let base = CpuRuntime::default_client(&device);
    let original = base.parallelism();
    let config = ParallelismConfig::new(Some(2), Some(257));
    let client = base.with_parallelism(config);
    for dtype in [DType::F32, DType::F64] {
        super::arithmetic::references::<CpuRuntime, _>(&client, dtype)?;
        super::arithmetic::cancellation::<CpuRuntime, _>(&client, dtype)?;
        super::arithmetic::product_guards::<CpuRuntime, _>(&client, dtype)?;
        super::admission::run::<CpuRuntime, _>(&client, dtype)?;
        super::ownership::run::<CpuRuntime, _>(&client, dtype)?;
    }
    assert_eq!(client.parallelism(), config);
    assert_eq!(base.parallelism(), original);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn actual_cuda_expansions_retain_independent_residuals_and_owned_inputs() -> Result {
    use super::fixture::rejected;
    use solvr::common::arithmetic::{ArithmeticError, with_interval_arithmetic};

    let client = CudaClient::new(CudaDevice::new(0))?;
    for dtype in [DType::F32, DType::F64] {
        super::arithmetic::references::<CudaRuntime, _>(&client, dtype)?;
        super::arithmetic::cancellation::<CudaRuntime, _>(&client, dtype)?;
        super::arithmetic::product_guards::<CudaRuntime, _>(&client, dtype)?;
        super::admission::run::<CudaRuntime, _>(&client, dtype)?;
        super::ownership::run::<CudaRuntime, _>(&client, dtype)?;
        let foreign = Tensor::<CudaRuntime>::empty(&[0], dtype, &CudaDevice::new(1))?;
        let local = Tensor::<CudaRuntime>::empty(&[0], dtype, client.device())?;
        let error = rejected(with_interval_arithmetic::<CudaRuntime, _, _, _>(
            &client,
            dtype,
            "foreign point device",
            |context| context.expansion_point(&foreign),
        ))?;
        assert!(matches!(error, ArithmeticError::Numr(_)));
        for position in 0..3 {
            let mut components = [&local, &local, &local];
            components[position] = &foreign;
            let error = rejected(with_interval_arithmetic::<CudaRuntime, _, _, _>(
                &client,
                dtype,
                "foreign component device",
                |context| context.expansion_bounds(components[0], components[1], components[2]),
            ))?;
            assert!(matches!(error, ArithmeticError::Numr(_)));
        }
    }
    Ok(())
}

#[cfg(feature = "wgpu")]
#[test]
fn actual_wgpu_expansion_provider_rejects_before_the_closure() -> Result {
    use super::fixture::rejected;
    use solvr::common::arithmetic::{ArithmeticError, with_interval_arithmetic};

    let client = WgpuClient::new(WgpuDevice::new(0))?;
    for dtype in [DType::F32, DType::F64] {
        let mut called = false;
        let error = rejected(with_interval_arithmetic::<WgpuRuntime, _, _, _>(
            &client,
            dtype,
            "undeclared expansion provider",
            |_context| {
                called = true;
                Ok(())
            },
        ))?;
        assert!(!called);
        assert!(
            matches!(error, ArithmeticError::UnsupportedArithmetic { dtype: actual, failed_probes } if actual == dtype && failed_probes == ["provider_contract"])
        );
    }
    Ok(())
}
