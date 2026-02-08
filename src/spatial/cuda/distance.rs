//! CUDA implementation of distance algorithms.

use crate::spatial::impl_generic::{
    cdist_impl, pdist_impl, squareform_impl, squareform_inverse_impl,
};
use crate::spatial::traits::distance::{DistanceAlgorithms, DistanceMetric};
use numr::error::Result;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl DistanceAlgorithms<CudaRuntime> for CudaClient {
    fn cdist(
        &self,
        x: &Tensor<CudaRuntime>,
        y: &Tensor<CudaRuntime>,
        metric: DistanceMetric,
    ) -> Result<Tensor<CudaRuntime>> {
        cdist_impl(self, x, y, metric)
    }

    fn pdist(
        &self,
        x: &Tensor<CudaRuntime>,
        metric: DistanceMetric,
    ) -> Result<Tensor<CudaRuntime>> {
        pdist_impl(self, x, metric)
    }

    fn squareform(&self, condensed: &Tensor<CudaRuntime>, n: usize) -> Result<Tensor<CudaRuntime>> {
        squareform_impl(self, condensed, n)
    }

    fn squareform_inverse(&self, square: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        squareform_inverse_impl(self, square)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cuda::CudaDevice;

    fn setup() -> Option<(CudaClient, CudaDevice)> {
        let device = CudaDevice::new(0);
        let client = CudaClient::new(device.clone()).ok()?;
        Some((client, device))
    }

    #[test]
    fn test_cdist_cuda() {
        let Some((client, device)) = setup() else {
            eprintln!("Skipping CUDA test: no device available");
            return;
        };

        let x = Tensor::<CudaRuntime>::from_slice(&[0.0, 0.0, 1.0, 0.0], &[2, 2], &device);
        let y = Tensor::<CudaRuntime>::from_slice(&[1.0, 1.0], &[1, 2], &device);

        let result = client.cdist(&x, &y, DistanceMetric::Euclidean).unwrap();
        assert_eq!(result.shape(), &[2, 1]);
    }
}
