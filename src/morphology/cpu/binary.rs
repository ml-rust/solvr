//! CPU implementation of binary morphology algorithms.

use crate::morphology::impl_generic::{
    binary_closing_impl, binary_dilation_impl, binary_erosion_impl, binary_fill_holes_impl,
    binary_opening_impl,
};
use crate::morphology::traits::binary::{BinaryMorphologyAlgorithms, StructuringElement};
use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl BinaryMorphologyAlgorithms<CpuRuntime> for CpuClient {
    fn binary_erosion(
        &self,
        input: &Tensor<CpuRuntime>,
        structure: StructuringElement,
        iterations: usize,
    ) -> Result<Tensor<CpuRuntime>> {
        binary_erosion_impl(self, input, structure, iterations)
    }

    fn binary_dilation(
        &self,
        input: &Tensor<CpuRuntime>,
        structure: StructuringElement,
        iterations: usize,
    ) -> Result<Tensor<CpuRuntime>> {
        binary_dilation_impl(self, input, structure, iterations)
    }

    fn binary_opening(
        &self,
        input: &Tensor<CpuRuntime>,
        structure: StructuringElement,
        iterations: usize,
    ) -> Result<Tensor<CpuRuntime>> {
        binary_opening_impl(self, input, structure, iterations)
    }

    fn binary_closing(
        &self,
        input: &Tensor<CpuRuntime>,
        structure: StructuringElement,
        iterations: usize,
    ) -> Result<Tensor<CpuRuntime>> {
        binary_closing_impl(self, input, structure, iterations)
    }

    fn binary_fill_holes(&self, input: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        binary_fill_holes_impl(self, input)
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
    fn test_binary_erosion_2d() {
        let (client, device) = setup();
        // 5x5 with a 3x3 block of ones in center
        let data = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0,
            1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let input = Tensor::from_slice(&data, &[5, 5], &device).unwrap();
        let result = client
            .binary_erosion(&input, StructuringElement::Cross, 1)
            .unwrap();
        let out: Vec<f64> = result.to_vec();
        // After cross erosion, only center pixel should remain
        assert!((out[12] - 1.0).abs() < 1e-10); // center
        assert!((out[6] - 0.0).abs() < 1e-10); // edge of block
    }

    #[test]
    fn test_binary_dilation_2d() {
        let (client, device) = setup();
        // Single pixel
        let mut data = vec![0.0; 25];
        data[12] = 1.0; // center
        let input = Tensor::from_slice(&data, &[5, 5], &device).unwrap();
        let result = client
            .binary_dilation(&input, StructuringElement::Cross, 1)
            .unwrap();
        let out: Vec<f64> = result.to_vec();
        // Center and 4 neighbors should be 1
        assert!((out[12] - 1.0).abs() < 1e-10);
        assert!((out[7] - 1.0).abs() < 1e-10); // above
        assert!((out[17] - 1.0).abs() < 1e-10); // below
        assert!((out[11] - 1.0).abs() < 1e-10); // left
        assert!((out[13] - 1.0).abs() < 1e-10); // right
    }

    #[test]
    fn test_binary_opening_removes_small() {
        let (client, device) = setup();
        // Large block + isolated pixel
        let data = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0,
            1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let input = Tensor::from_slice(&data, &[5, 5], &device).unwrap();
        let result = client
            .binary_opening(&input, StructuringElement::Cross, 1)
            .unwrap();
        let out: Vec<f64> = result.to_vec();
        // Isolated pixel should be removed
        assert!((out[24] - 0.0).abs() < 1e-10);
        // Center of block should remain
        assert!((out[12] - 1.0).abs() < 1e-10);
    }
}
