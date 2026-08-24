//! CPU implementation of grey-scale morphology algorithms.

use crate::morphology::impl_generic::{
    black_tophat_impl, grey_closing_impl, grey_dilation_impl, grey_erosion_impl, grey_opening_impl,
    morphological_gradient_impl, white_tophat_impl,
};
use crate::morphology::traits::grey::GreyMorphologyAlgorithms;
use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl GreyMorphologyAlgorithms<CpuRuntime> for CpuClient {
    fn grey_erosion(
        &self,
        input: &Tensor<CpuRuntime>,
        size: &[usize],
    ) -> Result<Tensor<CpuRuntime>> {
        grey_erosion_impl(self, input, size)
    }
    fn grey_dilation(
        &self,
        input: &Tensor<CpuRuntime>,
        size: &[usize],
    ) -> Result<Tensor<CpuRuntime>> {
        grey_dilation_impl(self, input, size)
    }
    fn grey_opening(
        &self,
        input: &Tensor<CpuRuntime>,
        size: &[usize],
    ) -> Result<Tensor<CpuRuntime>> {
        grey_opening_impl(self, input, size)
    }
    fn grey_closing(
        &self,
        input: &Tensor<CpuRuntime>,
        size: &[usize],
    ) -> Result<Tensor<CpuRuntime>> {
        grey_closing_impl(self, input, size)
    }
    fn morphological_gradient(
        &self,
        input: &Tensor<CpuRuntime>,
        size: &[usize],
    ) -> Result<Tensor<CpuRuntime>> {
        morphological_gradient_impl(self, input, size)
    }
    fn white_tophat(
        &self,
        input: &Tensor<CpuRuntime>,
        size: &[usize],
    ) -> Result<Tensor<CpuRuntime>> {
        white_tophat_impl(self, input, size)
    }
    fn black_tophat(
        &self,
        input: &Tensor<CpuRuntime>,
        size: &[usize],
    ) -> Result<Tensor<CpuRuntime>> {
        black_tophat_impl(self, input, size)
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
    fn test_grey_erosion() {
        let (client, device) = setup();
        let data = vec![5.0, 3.0, 7.0, 1.0, 4.0];
        let input = Tensor::from_slice(&data, &[5], &device).unwrap();
        let result = client.grey_erosion(&input, &[3]).unwrap();
        let out: Vec<f64> = result.to_vec();
        // Position 2: min(3, 7, 1) = 1.0
        assert!((out[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_morphological_gradient() {
        let (client, device) = setup();
        let data = vec![0.0, 0.0, 5.0, 5.0, 5.0];
        let input = Tensor::from_slice(&data, &[5], &device).unwrap();
        let result = client.morphological_gradient(&input, &[3]).unwrap();
        let out: Vec<f64> = result.to_vec();
        // At the edge (position 1-2), gradient should be large
        assert!(out[1] > 0.0 || out[2] > 0.0);
    }
}
