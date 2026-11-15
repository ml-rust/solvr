#[path = "selected_tests_admission.rs"]
mod admission;
#[path = "selected_tests_behavior.rs"]
mod behavior;
#[path = "selected_tests_refresh.rs"]
mod refresh;
#[path = "selected_tests_support.rs"]
mod support;

use support::{DType, Result};

fn run<R, C>(client: &C) -> Result
where
    R: support::Runtime<DType = DType>,
    C: support::Client<R>,
{
    for dtype in [DType::F32, DType::F64] {
        behavior::mixed(client, dtype)?;
        behavior::empty(client, dtype)?;
        behavior::all_owners(client, dtype)?;
        behavior::ceilings(client, dtype)?;
        behavior::floor_history(client, dtype)?;
        admission::run(client, dtype)?;
        refresh::policy(client, dtype)?;
        refresh::exhausted(client, dtype)?;
        refresh::admission(client, dtype)?;
    }
    Ok(())
}
#[test]
fn configured_cpu_selected_mapped_resume_preserves_complete_state() -> Result {
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
    let original = CpuClient::new(CpuDevice::new());
    let before = original.parallelism();
    let config = ParallelismConfig::new(Some(2), Some(257));
    let client = original.with_parallelism(config);
    run::<CpuRuntime, _>(&client)?;
    assert_eq!(client.parallelism(), config);
    assert_eq!(original.parallelism(), before);
    Ok(())
}
#[cfg(feature = "cuda")]
#[test]
fn actual_cuda_selected_mapped_resume_preserves_complete_state() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    let client = CudaClient::new(CudaDevice::new(0))?;
    run::<CudaRuntime, _>(&client)?;
    let foreign = support::Tensor::<CudaRuntime>::empty(&[0], DType::I64, &CudaDevice::new(1))?;
    admission::invalid(&client, DType::F64, &foreign)?;
    Ok(())
}
