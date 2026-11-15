use super::support::Result;
use numr::prelude::*;
use numr::runtime::cpu::ParallelismConfig;

#[test]
fn configured_cpu_interval_operations_enclose_exact_references() -> Result {
    let device = CpuDevice::new();
    let base = CpuRuntime::default_client(&device);
    let original = base.parallelism();
    let config = ParallelismConfig::new(Some(2), Some(257));
    let client = base.with_parallelism(config);
    for dtype in [DType::F32, DType::F64] {
        super::arithmetic::elementary::<CpuRuntime, _>(&client, dtype)?;
        super::arithmetic::division::<CpuRuntime, _>(&client, dtype)?;
        super::arithmetic::square_root::<CpuRuntime, _>(&client, dtype)?;
    }
    assert_eq!(client.parallelism(), config);
    assert_eq!(base.parallelism(), original);
    Ok(())
}

#[test]
fn cpu_interval_completion_rejects_invalid_arithmetic_and_metadata() -> Result {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    for dtype in [DType::F32, DType::F64] {
        super::admission::run::<CpuRuntime, _>(&client, dtype)?;
    }
    Ok(())
}

#[test]
fn cpu_interval_constructors_own_strided_and_empty_inputs() -> Result {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    for dtype in [DType::F32, DType::F64] {
        super::ownership::run::<CpuRuntime, _>(&client, dtype)?;
    }
    Ok(())
}
