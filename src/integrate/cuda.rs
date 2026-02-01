//! CUDA implementation of integration algorithms.

use numr::error::Result;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use crate::integrate::IntegrationAlgorithms;
use crate::integrate::impl_generic::quadrature::{
    cumulative_trapezoid_impl, fixed_quad_impl, simpson_impl, trapezoid_impl,
    trapezoid_uniform_impl,
};

impl IntegrationAlgorithms<CudaRuntime> for CudaClient {
    fn trapezoid(
        &self,
        y: &Tensor<CudaRuntime>,
        x: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        trapezoid_impl(self, y, x)
    }

    fn trapezoid_uniform(&self, y: &Tensor<CudaRuntime>, dx: f64) -> Result<Tensor<CudaRuntime>> {
        trapezoid_uniform_impl(self, y, dx)
    }

    fn cumulative_trapezoid(
        &self,
        y: &Tensor<CudaRuntime>,
        x: Option<&Tensor<CudaRuntime>>,
        dx: f64,
    ) -> Result<Tensor<CudaRuntime>> {
        cumulative_trapezoid_impl(self, y, x, dx)
    }

    fn simpson(
        &self,
        y: &Tensor<CudaRuntime>,
        x: Option<&Tensor<CudaRuntime>>,
        dx: f64,
    ) -> Result<Tensor<CudaRuntime>> {
        simpson_impl(self, y, x, dx)
    }

    fn fixed_quad<F>(&self, f: F, a: f64, b: f64, n: usize) -> Result<Tensor<CudaRuntime>>
    where
        F: Fn(&Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>>,
    {
        fixed_quad_impl(self, f, a, b, n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cuda::CudaDevice;

    fn setup() -> Option<(CudaDevice, CudaClient)> {
        let device = CudaDevice::new(0);
        let client = CudaClient::new(device.clone()).ok()?;
        Some((device, client))
    }

    #[test]
    fn test_trapezoid_cuda() {
        let Some((device, client)) = setup() else {
            eprintln!("Skipping CUDA test: no device");
            return;
        };

        let n = 101;
        let x_data: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
        let y_data: Vec<f64> = x_data.iter().map(|&xi| xi * xi).collect();

        let x = Tensor::<CudaRuntime>::from_slice(&x_data, &[n], &device);
        let y = Tensor::<CudaRuntime>::from_slice(&y_data, &[n], &device);

        let result = client.trapezoid(&y, &x).unwrap();
        let result_val: Vec<f64> = result.to_vec();

        assert!((result_val[0] - 1.0 / 3.0).abs() < 0.001);
    }

    #[test]
    fn test_fixed_quad_cuda() {
        let Some((device, client)) = setup() else {
            eprintln!("Skipping CUDA test: no device");
            return;
        };

        let result = client
            .fixed_quad(
                |x| {
                    let data: Vec<f64> = x.to_vec();
                    let sin_data: Vec<f64> = data.iter().map(|&xi| xi.sin()).collect();
                    Ok(Tensor::<CudaRuntime>::from_slice(
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
