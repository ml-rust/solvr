//! Linear interpolation parity on actual CUDA and WebGPU clients.

#[cfg(any(feature = "cuda", feature = "wgpu"))]
#[path = "interp1d/cases.rs"]
mod cases;

#[cfg(feature = "cuda")]
#[test]
fn linear_cuda_matches_configured_cpu_for_f32_and_f64()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    use numr::prelude::*;
    use numr::runtime::cpu::ParallelismConfig;
    let cpu_device = CpuDevice::new();
    let cpu =
        CpuClient::new(cpu_device.clone()).with_parallelism(ParallelismConfig::new(Some(1), None));
    let device = CudaDevice::new(0);
    let client = CudaClient::new(device.clone())?;
    for dtype in [DType::F32, DType::F64] {
        let expected = cases::observations(&cpu, &cpu_device, dtype)?;
        let actual = cases::observations(&client, &device, dtype)?;
        assert_eq!(actual, expected);
    }
    Ok(())
}

#[cfg(feature = "wgpu")]
#[test]
fn linear_wgpu_matches_configured_cpu_and_rejects_device_mismatch()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    use numr::prelude::*;
    use numr::runtime::cpu::ParallelismConfig;
    use solvr::interpolate::InterpolateError;
    use solvr::interpolate::{Interp1dAlgorithms, InterpMethod};
    let cpu_device = CpuDevice::new();
    let cpu =
        CpuClient::new(cpu_device.clone()).with_parallelism(ParallelismConfig::new(Some(1), None));
    let device = WgpuDevice::new(0);
    let client = WgpuClient::new(device.clone())?;
    let expected = cases::observations(&cpu, &cpu_device, DType::F32)?;
    let actual = cases::observations(&client, &device, DType::F32)?;
    assert_eq!(actual, expected);
    let x = Tensor::<WgpuRuntime>::from_slice(&[0.0f32, 1.0], &[2], &device)?;
    let other_device = WgpuDevice::new(usize::MAX);
    let query = Tensor::<WgpuRuntime>::empty(&[0], DType::F32, &other_device)?;
    assert!(matches!(
        client.interp1d(&x, &x, &query, InterpMethod::Linear),
        Err(InterpolateError::InvalidParameter { .. })
    ));
    let f64 = Tensor::<WgpuRuntime>::empty(&[0], DType::F64, &device)?;
    assert!(
        client
            .interp1d(&f64, &f64, &f64, InterpMethod::Linear)
            .is_err()
    );
    Ok(())
}
