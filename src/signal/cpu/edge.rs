//! CPU implementation of edge detection algorithms.

use crate::signal::impl_generic::{
    gaussian_gradient_magnitude_impl, gaussian_laplace_impl, laplace_impl, prewitt_impl, sobel_impl,
};
use crate::signal::traits::edge::EdgeDetectionAlgorithms;
use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl EdgeDetectionAlgorithms<CpuRuntime> for CpuClient {
    fn sobel(&self, input: &Tensor<CpuRuntime>, axis: usize) -> Result<Tensor<CpuRuntime>> {
        sobel_impl(self, input, axis)
    }

    fn prewitt(&self, input: &Tensor<CpuRuntime>, axis: usize) -> Result<Tensor<CpuRuntime>> {
        prewitt_impl(self, input, axis)
    }

    fn laplace(&self, input: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        laplace_impl(self, input)
    }

    fn gaussian_laplace(
        &self,
        input: &Tensor<CpuRuntime>,
        sigma: f64,
    ) -> Result<Tensor<CpuRuntime>> {
        gaussian_laplace_impl(self, input, sigma)
    }

    fn gaussian_gradient_magnitude(
        &self,
        input: &Tensor<CpuRuntime>,
        sigma: f64,
    ) -> Result<Tensor<CpuRuntime>> {
        gaussian_gradient_magnitude_impl(self, input, sigma)
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
    fn test_sobel_2d_horizontal_edge() {
        let (client, device) = setup();
        // Image with horizontal edge: top half = 0, bottom half = 1
        let mut data = vec![0.0; 30]; // 6x5
        for i in 3..6 {
            for j in 0..5 {
                data[i * 5 + j] = 1.0;
            }
        }
        let input = Tensor::from_slice(&data, &[6, 5], &device).unwrap();

        // Sobel along axis 0 should detect horizontal edge
        let result = client.sobel(&input, 0).unwrap();
        let out: Vec<f64> = result.to_vec();

        // Rows 2-3 (near the edge) should have larger absolute values
        let edge_val = out[2 * 5 + 2].abs();
        let non_edge_val = out[2].abs(); // row 0, col 2
        assert!(edge_val > non_edge_val);
    }

    #[test]
    fn test_laplace_2d() {
        let (client, device) = setup();
        // Smooth quadratic: f(x,y) = x^2 + y^2, laplacian = 4
        let mut data = Vec::new();
        for i in 0..5 {
            for j in 0..5 {
                data.push((i as f64 - 2.0).powi(2) + (j as f64 - 2.0).powi(2));
            }
        }
        let input = Tensor::from_slice(&data, &[5, 5], &device).unwrap();
        let result = client.laplace(&input).unwrap();
        let out: Vec<f64> = result.to_vec();

        // Interior points: laplacian of x^2+y^2 using [1,-2,1] gives 2+2=4
        assert!((out[12] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_gaussian_gradient_magnitude() {
        let (client, device) = setup();
        // Constant image: gradient magnitude should be ~0
        let data = vec![5.0; 25];
        let input = Tensor::from_slice(&data, &[5, 5], &device).unwrap();
        let result = client.gaussian_gradient_magnitude(&input, 1.0).unwrap();
        let out: Vec<f64> = result.to_vec();
        for v in &out {
            assert!(v.abs() < 1e-6);
        }
    }
}
