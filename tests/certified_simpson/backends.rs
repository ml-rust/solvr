use super::fixture::Result;
use numr::prelude::*;
#[cfg(feature = "wgpu")]
use solvr::integrate::IntegrateError;
use solvr::integrate::{CertifiedSimpsonAlgorithms, CertifiedSimpsonOptions};

#[cfg(feature = "cuda")]
#[test]
fn actual_cuda_integrals_contain_all_frozen_references() -> Result {
    let device = CudaDevice::new(0);
    let client = CudaClient::new(device)?;
    super::suite::run::<CudaRuntime, _>(&client, false)?;
    super::suite::run::<CudaRuntime, _>(&client, true)?;
    super::admission::run::<CudaRuntime, _>(&client)?;
    super::ownership::run::<CudaRuntime, _>(&client)?;
    let foreign = Tensor::<CudaRuntime>::empty(&[0], DType::F32, &CudaDevice::new(1))?;
    assert!(
        client
            .certified_simpson(
                &foreign,
                &foreign,
                CertifiedSimpsonOptions {
                    panels: 1,
                    max_points: 1
                },
                |_| panic!("foreign device reaches callback"),
                |_| panic!("foreign device reaches derivative")
            )
            .is_err()
    );
    Ok(())
}
#[cfg(feature = "wgpu")]
#[test]
fn actual_wgpu_rejects_undeclared_arithmetic_before_callbacks() -> Result {
    let device = WgpuDevice::new(0);
    let client = WgpuClient::new(device.clone())?;
    for values in [vec![], vec![0.0f32], vec![f32::NAN]] {
        let endpoints = Tensor::<WgpuRuntime>::from_slice(&values, &[values.len()], &device)?;
        let error = client
            .certified_simpson(
                &endpoints,
                &endpoints,
                CertifiedSimpsonOptions {
                    panels: 1,
                    max_points: 4,
                },
                |_| panic!("undeclared provider reaches callback"),
                |_| panic!("undeclared provider reaches derivative"),
            )
            .expect_err("undeclared arithmetic accepted");
        assert!(
            matches!(error, IntegrateError::UnsupportedArithmetic { dtype: DType::F32, failed_probes } if failed_probes == ["provider_contract"])
        );
    }
    let empty_f64 = Tensor::<WgpuRuntime>::empty(&[0], DType::F64, &device)?;
    let error = client
        .certified_simpson(
            &empty_f64,
            &empty_f64,
            CertifiedSimpsonOptions {
                panels: 1,
                max_points: 4,
            },
            |_| panic!("undeclared F64 provider reaches callback"),
            |_| panic!("undeclared F64 provider reaches derivative"),
        )
        .expect_err("undeclared F64 arithmetic accepted");
    assert!(
        matches!(error, IntegrateError::UnsupportedArithmetic { dtype: DType::F64, failed_probes } if failed_probes == ["provider_contract"])
    );
    let foreign = Tensor::<WgpuRuntime>::empty(&[0], DType::F32, &WgpuDevice::new(1))?;
    assert!(
        client
            .certified_simpson(
                &foreign,
                &foreign,
                CertifiedSimpsonOptions {
                    panels: 1,
                    max_points: 1
                },
                |_| panic!("foreign device reaches callback"),
                |_| panic!("foreign device reaches derivative")
            )
            .is_err()
    );
    Ok(())
}
