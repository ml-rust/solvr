//! CPU implementation of measurement algorithms.

use crate::morphology::impl_generic::{
    center_of_mass_impl, find_objects_impl, label_impl, mean_labels_impl, sum_labels_impl,
};
use crate::morphology::traits::binary::StructuringElement;
use crate::morphology::traits::measurements::{MeasurementAlgorithms, RegionProperties};
use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl MeasurementAlgorithms<CpuRuntime> for CpuClient {
    fn label(
        &self,
        input: &Tensor<CpuRuntime>,
        structure: StructuringElement,
    ) -> Result<(Tensor<CpuRuntime>, usize)> {
        label_impl(self, input, structure)
    }
    fn find_objects(
        &self,
        labels: &Tensor<CpuRuntime>,
        num_labels: usize,
    ) -> Result<Vec<RegionProperties>> {
        find_objects_impl(self, labels, num_labels)
    }
    fn sum_labels(
        &self,
        input: &Tensor<CpuRuntime>,
        labels: &Tensor<CpuRuntime>,
        num_labels: usize,
    ) -> Result<Tensor<CpuRuntime>> {
        sum_labels_impl(self, input, labels, num_labels)
    }
    fn mean_labels(
        &self,
        input: &Tensor<CpuRuntime>,
        labels: &Tensor<CpuRuntime>,
        num_labels: usize,
    ) -> Result<Tensor<CpuRuntime>> {
        mean_labels_impl(self, input, labels, num_labels)
    }
    fn center_of_mass(
        &self,
        input: &Tensor<CpuRuntime>,
        labels: &Tensor<CpuRuntime>,
        num_labels: usize,
    ) -> Result<Tensor<CpuRuntime>> {
        center_of_mass_impl(self, input, labels, num_labels)
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
    fn test_label_two_components() {
        let (client, device) = setup();
        // Two separate blocks
        let data = vec![
            1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0,
        ];
        let input = Tensor::from_slice(&data, &[5, 5], &device).unwrap();
        let (labels, num_labels) = client.label(&input, StructuringElement::Cross).unwrap();
        assert_eq!(num_labels, 2);
        let label_data: Vec<f64> = labels.to_vec();
        // Two blocks should have different labels
        assert!(label_data[0] > 0.0);
        assert!(label_data[23] > 0.0);
        assert_ne!(label_data[0] as i32, label_data[23] as i32);
    }

    #[test]
    fn test_sum_labels() {
        let (client, device) = setup();
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let labels_data = vec![1.0, 1.0, 2.0, 2.0, 2.0];
        let input = Tensor::from_slice(&values, &[5], &device).unwrap();
        let labels = Tensor::from_slice(&labels_data, &[5], &device).unwrap();
        let result = client.sum_labels(&input, &labels, 2).unwrap();
        let out: Vec<f64> = result.to_vec();
        assert!((out[0] - 3.0).abs() < 1e-10); // 1+2
        assert!((out[1] - 12.0).abs() < 1e-10); // 3+4+5
    }
}
