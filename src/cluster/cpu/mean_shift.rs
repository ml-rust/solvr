//! CPU implementation of Mean Shift clustering.

use crate::cluster::impl_generic::mean_shift_impl;
use crate::cluster::traits::mean_shift::{MeanShiftAlgorithms, MeanShiftOptions, MeanShiftResult};
use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl MeanShiftAlgorithms<CpuRuntime> for CpuClient {
    fn mean_shift(
        &self,
        data: &Tensor<CpuRuntime>,
        options: &MeanShiftOptions,
    ) -> Result<MeanShiftResult<CpuRuntime>> {
        mean_shift_impl(self, data, options)
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
    fn test_mean_shift_basic() {
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

        let options = MeanShiftOptions {
            bandwidth: Some(2.0),
            ..Default::default()
        };

        let result = client.mean_shift(&data, &options).unwrap();
        assert_eq!(result.labels.shape(), &[8]);
        // Should find at least 1 cluster center
        assert!(result.cluster_centers.shape()[0] >= 1);
        assert_eq!(result.cluster_centers.shape()[1], 2);
    }
}
