//! WebGPU implementation of integration algorithms.

use numr::error::Result;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

use crate::integrate::IntegrationAlgorithms;
use crate::integrate::impl_generic::quadrature::{
    cumulative_trapezoid_impl, fixed_quad_impl, simpson_impl, trapezoid_impl,
    trapezoid_uniform_impl,
};

impl IntegrationAlgorithms<WgpuRuntime> for WgpuClient {
    fn trapezoid(
        &self,
        y: &Tensor<WgpuRuntime>,
        x: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        trapezoid_impl(self, y, x)
    }

    fn trapezoid_uniform(&self, y: &Tensor<WgpuRuntime>, dx: f64) -> Result<Tensor<WgpuRuntime>> {
        trapezoid_uniform_impl(self, y, dx)
    }

    fn cumulative_trapezoid(
        &self,
        y: &Tensor<WgpuRuntime>,
        x: Option<&Tensor<WgpuRuntime>>,
        dx: f64,
    ) -> Result<Tensor<WgpuRuntime>> {
        cumulative_trapezoid_impl(self, y, x, dx)
    }

    fn simpson(
        &self,
        y: &Tensor<WgpuRuntime>,
        x: Option<&Tensor<WgpuRuntime>>,
        dx: f64,
    ) -> Result<Tensor<WgpuRuntime>> {
        simpson_impl(self, y, x, dx)
    }

    fn fixed_quad<F>(&self, f: F, a: f64, b: f64, n: usize) -> Result<Tensor<WgpuRuntime>>
    where
        F: Fn(&Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>>,
    {
        fixed_quad_impl(self, f, a, b, n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::wgpu::WgpuDevice;

    fn setup() -> Option<(WgpuDevice, WgpuClient)> {
        let device = WgpuDevice::new(0);
        let client = WgpuClient::new(device.clone()).ok()?;
        Some((device, client))
    }

    #[test]
    fn test_trapezoid_wgpu() {
        let Some((device, client)) = setup() else {
            eprintln!("Skipping WebGPU test: no device");
            return;
        };

        let n = 101;
        let x_data: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
        let y_data: Vec<f64> = x_data.iter().map(|&xi| xi * xi).collect();

        let x = Tensor::<WgpuRuntime>::from_slice(&x_data, &[n], &device);
        let y = Tensor::<WgpuRuntime>::from_slice(&y_data, &[n], &device);

        let result = client.trapezoid(&y, &x).unwrap();
        let result_val: Vec<f64> = result.to_vec();

        assert!((result_val[0] - 1.0 / 3.0).abs() < 0.001);
    }

    #[test]
    fn test_fixed_quad_wgpu() {
        let Some((device, client)) = setup() else {
            eprintln!("Skipping WebGPU test: no device");
            return;
        };

        let result = client
            .fixed_quad(
                |x| {
                    let data: Vec<f64> = x.to_vec();
                    let sin_data: Vec<f64> = data.iter().map(|&xi| xi.sin()).collect();
                    Ok(Tensor::<WgpuRuntime>::from_slice(
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
        assert!((result_val[0] - 2.0).abs() < 1e-10);
    }
}
