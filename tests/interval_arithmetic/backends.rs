use super::support::{Result, failure};
use numr::prelude::*;
use solvr::common::arithmetic::{ArithmeticError, with_interval_arithmetic};

#[cfg(feature = "cuda")]
#[test]
fn actual_cuda_interval_operations_match_exact_references_and_provider_limits() -> Result {
    let client = CudaClient::new(CudaDevice::new(0))?;
    for dtype in [DType::F32, DType::F64] {
        super::arithmetic::elementary::<CudaRuntime, _>(&client, dtype)?;
        super::arithmetic::division::<CudaRuntime, _>(&client, dtype)?;
        super::arithmetic::square_root::<CudaRuntime, _>(&client, dtype)?;
        super::admission::run::<CudaRuntime, _>(&client, dtype)?;
        super::ownership::run::<CudaRuntime, _>(&client, dtype)?;
    }
    let foreign = Tensor::<CudaRuntime>::empty(&[0], DType::F32, &CudaDevice::new(1))?;
    let error = failure(with_interval_arithmetic::<CudaRuntime, _, _, _>(
        &client,
        DType::F32,
        "foreign device",
        |context| context.point(&foreign),
    ));
    assert!(matches!(error, ArithmeticError::Numr(_)));
    Ok(())
}

#[cfg(feature = "wgpu")]
#[test]
fn actual_wgpu_interval_provider_rejects_before_the_closure() -> Result {
    let client = WgpuClient::new(WgpuDevice::new(0))?;
    for dtype in [DType::F32, DType::F64] {
        let mut called = false;
        let error = failure(with_interval_arithmetic::<WgpuRuntime, _, _, _>(
            &client,
            dtype,
            "undeclared provider",
            |_context| {
                called = true;
                Ok(())
            },
        ));
        assert!(!called);
        assert!(
            matches!(error, ArithmeticError::UnsupportedArithmetic { dtype: actual, failed_probes } if actual == dtype && failed_probes == ["provider_contract"])
        );
    }
    Ok(())
}
