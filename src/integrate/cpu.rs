//! CPU implementation of integration algorithms.

use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

use crate::integrate::IntegrationAlgorithms;
use crate::integrate::impl_generic::quadrature::{
    cumulative_trapezoid_impl, fixed_quad_impl, simpson_impl, trapezoid_impl,
    trapezoid_uniform_impl,
};

impl IntegrationAlgorithms<CpuRuntime> for CpuClient {
    fn trapezoid(
        &self,
        y: &Tensor<CpuRuntime>,
        x: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        trapezoid_impl(self, y, x)
    }

    fn trapezoid_uniform(&self, y: &Tensor<CpuRuntime>, dx: f64) -> Result<Tensor<CpuRuntime>> {
        trapezoid_uniform_impl(self, y, dx)
    }

    fn cumulative_trapezoid(
        &self,
        y: &Tensor<CpuRuntime>,
        x: Option<&Tensor<CpuRuntime>>,
        dx: f64,
    ) -> Result<Tensor<CpuRuntime>> {
        cumulative_trapezoid_impl(self, y, x, dx)
    }

    fn simpson(
        &self,
        y: &Tensor<CpuRuntime>,
        x: Option<&Tensor<CpuRuntime>>,
        dx: f64,
    ) -> Result<Tensor<CpuRuntime>> {
        simpson_impl(self, y, x, dx)
    }

    fn fixed_quad<F>(&self, f: F, a: f64, b: f64, n: usize) -> Result<Tensor<CpuRuntime>>
    where
        F: Fn(&Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>>,
    {
        fixed_quad_impl(self, f, a, b, n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::CpuDevice;

    fn setup() -> (CpuDevice, CpuClient) {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        (device, client)
    }

    #[test]
    fn test_trapezoid_cpu() {
        let (device, client) = setup();

        // Integrate y = x^2 from 0 to 1
        let n = 101;
        let x_data: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
        let y_data: Vec<f64> = x_data.iter().map(|&xi| xi * xi).collect();

        let x = Tensor::<CpuRuntime>::from_slice(&x_data, &[n], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&y_data, &[n], &device);

        let result = client.trapezoid(&y, &x).unwrap();
        let result_val: Vec<f64> = result.to_vec();

        // Exact value is 1/3
        assert!((result_val[0] - 1.0 / 3.0).abs() < 0.001);
    }

    #[test]
    fn test_simpson_cpu() {
        let (device, client) = setup();

        // Integrate y = x^2 from 0 to 1 (using uniform spacing)
        let n = 101;
        let dx = 1.0 / (n - 1) as f64;
        let y_data: Vec<f64> = (0..n)
            .map(|i| {
                let x = i as f64 * dx;
                x * x
            })
            .collect();

        let y = Tensor::<CpuRuntime>::from_slice(&y_data, &[n], &device);

        let result = client.simpson(&y, None, dx).unwrap();
        let result_val: Vec<f64> = result.to_vec();

        // Exact value is 1/3
        assert!((result_val[0] - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_fixed_quad_cpu() {
        let (device, client) = setup();

        // Integrate sin(x) from 0 to pi
        let result = client
            .fixed_quad(
                |x| {
                    let data: Vec<f64> = x.to_vec();
                    let sin_data: Vec<f64> = data.iter().map(|&xi| xi.sin()).collect();
                    Ok(Tensor::<CpuRuntime>::from_slice(
                        &sin_data,
                        x.shape(),
                        &device,
                    ))
                },
                0.0,
                std::f64::consts::PI,
                10,
            )
            .unwrap();

        let result_val: Vec<f64> = result.to_vec();

        // Exact value is 2.0
        assert!((result_val[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_batch_trapezoid_cpu() {
        let (device, client) = setup();

        // Batch integration: integrate y = x and y = x^2 simultaneously
        let n = 101;
        let x_data: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();

        // First row: y = x (integral = 0.5)
        // Second row: y = x^2 (integral = 1/3)
        let mut y_data = Vec::with_capacity(2 * n);
        for &xi in &x_data {
            y_data.push(xi);
        }
        for &xi in &x_data {
            y_data.push(xi * xi);
        }

        let x = Tensor::<CpuRuntime>::from_slice(&x_data, &[n], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&y_data, &[2, n], &device);

        let result = client.trapezoid(&y, &x).unwrap();
        let result_val: Vec<f64> = result.to_vec();

        assert!((result_val[0] - 0.5).abs() < 0.001);
        assert!((result_val[1] - 1.0 / 3.0).abs() < 0.001);
    }
}
