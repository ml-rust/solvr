//! CPU implementation of OPTICS clustering.

use crate::cluster::impl_generic::optics_impl;
use crate::cluster::traits::optics::{OpticsAlgorithms, OpticsOptions, OpticsResult};
use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl OpticsAlgorithms<CpuRuntime> for CpuClient {
    fn optics(
        &self,
        data: &Tensor<CpuRuntime>,
        options: &OpticsOptions,
    ) -> Result<OpticsResult<CpuRuntime>> {
        optics_impl(self, data, options)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::CpuDevice;

    fn setup() -> (CpuClient, CpuDevice) {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        (client, device)
    }

    #[test]
    fn test_optics_basic() {
        let (client, device) = setup();

        #[rustfmt::skip]
        let data = Tensor::<CpuRuntime>::from_slice(
            &[
                0.0, 0.0,
                0.1, 0.1,
                0.2, 0.0,
                0.0, 0.2,
                10.0, 10.0,
                10.1, 10.1,
                10.2, 10.0,
                10.0, 10.2,
            ],
            &[8, 2],
            &device,
        );

        let options = OpticsOptions {
            min_samples: 2,
            ..Default::default()
        };

        let result = client.optics(&data, &options).unwrap();
        assert_eq!(result.ordering.shape(), &[8]);
        assert_eq!(result.reachability.shape(), &[8]);
        assert_eq!(result.core_distances.shape(), &[8]);
    }
}
