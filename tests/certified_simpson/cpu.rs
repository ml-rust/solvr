use super::fixture::Result;
use numr::prelude::*;
use numr::runtime::cpu::ParallelismConfig;

#[test]
fn configured_cpu_integrals_contain_all_frozen_references() -> Result {
    let device = CpuDevice::new();
    let base = CpuRuntime::default_client(&device);
    let original = base.parallelism();
    let config = ParallelismConfig::new(Some(2), Some(257));
    let client = base.with_parallelism(config);
    super::suite::run::<CpuRuntime, _>(&client, false)?;
    assert_eq!(client.parallelism(), config);
    assert_eq!(base.parallelism(), original);
    Ok(())
}
#[test]
fn strided_cpu_endpoints_preserve_reference_integrals_and_storage() -> Result {
    let device = CpuDevice::new();
    super::suite::run::<CpuRuntime, _>(&CpuRuntime::default_client(&device), true)
}
#[test]
fn cpu_admission_preserves_callback_errors_and_rejects_malformed_results() -> Result {
    let device = CpuDevice::new();
    super::admission::run::<CpuRuntime, _>(&CpuRuntime::default_client(&device))
}
#[test]
fn cpu_derivative_callback_cannot_modify_returned_function_values() -> Result {
    let device = CpuDevice::new();
    super::ownership::run::<CpuRuntime, _>(&CpuRuntime::default_client(&device))
}
#[test]
fn cpu_panels_span_value_node_bounds() -> Result {
    let device = CpuDevice::new();
    super::panel_layout::run::<CpuRuntime, _>(&CpuRuntime::default_client(&device))
}
