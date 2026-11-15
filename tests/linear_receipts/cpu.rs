use numr::prelude::{CpuClient, CpuDevice};
use numr::runtime::cpu::ParallelismConfig;

pub fn setup() -> (CpuClient, CpuDevice) {
    let device = CpuDevice::new();
    let client =
        CpuClient::new(device.clone()).with_parallelism(ParallelismConfig::new(Some(1), None));
    (client, device)
}
