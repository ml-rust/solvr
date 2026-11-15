use super::{fixture::Result, suite};
use numr::prelude::*;
#[cfg(feature = "wgpu")]
use solvr::interpolate::{InterpolateError, LinearInterp1dDiagnostics};

#[test]
fn actual_backends_enclose_values_or_reject_backend_defined_arithmetic() -> Result {
    #[cfg(feature = "cuda")]
    {
        let device = CudaDevice::new(0);
        let client = CudaClient::new(device.clone())?;
        client.synchronize();
        for dtype in [DType::F32, DType::F64] {
            suite::run(&client, &device, dtype, false)?;
        }
    }
    #[cfg(feature = "wgpu")]
    {
        let device = WgpuDevice::new(0);
        let client = WgpuClient::new(device.clone())?;
        suite::run(&client, &device, DType::F32, true)?;
        let x = Tensor::<WgpuRuntime>::from_slice(&[0.0f32, 1.0], &[2], &device)?;
        let other_device = WgpuDevice::new(usize::MAX);
        let query = Tensor::<WgpuRuntime>::empty(&[0], DType::F32, &other_device)?;
        assert!(matches!(
            client.interp1d_linear_with_receipts(&x, &x, &query),
            Err(InterpolateError::InvalidParameter { .. })
        ));
        let x = Tensor::<WgpuRuntime>::from_slice(&[0.0f64, 1.0], &[2], &device)?;
        let error = client
            .interp1d_linear_with_receipts(&x, &x, &x)
            .err()
            .ok_or("WebGPU F64 metadata unexpectedly admits receipt arithmetic")?;
        assert!(
            matches!(&error, InterpolateError::InvalidParameter { parameter, .. }
            if parameter == "dtype"),
            "{error}"
        );
        eprintln!("WebGPU F64 raw storage does not establish arithmetic support: {error}");
    }
    Ok(())
}
