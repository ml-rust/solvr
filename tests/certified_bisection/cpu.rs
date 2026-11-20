use super::fixture::Result;
use numr::prelude::*;
use numr::runtime::cpu::ParallelismConfig;

#[test]
fn configured_cpu_roots_contain_all_independent_references() -> Result {
    let device = CpuDevice::new();
    let original = CpuRuntime::default_client(&device);
    let original_config = original.parallelism();
    let config = ParallelismConfig::new(Some(2), Some(257));
    let client = original.with_parallelism(config);
    super::suite::run::<CpuRuntime, _>(&client, false)?;
    super::suite::run::<CpuRuntime, _>(&client, true)?;
    super::admission::run::<CpuRuntime, _>(&client)?;
    super::budget::run::<CpuRuntime, _>(&client)?;
    super::ownership::run::<CpuRuntime, _>(&client)?;
    assert_eq!(client.parallelism(), config);
    assert_eq!(original.parallelism(), original_config);
    Ok(())
}

#[test]
fn configured_cpu_slack_secant_keeps_certified_contracts() -> Result {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device)
        .with_parallelism(ParallelismConfig::new(Some(2), Some(257)));
    super::slack_secant::run::<CpuRuntime, _>(&client)
}
