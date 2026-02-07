//! CUDA implementation of information theory algorithms.

use crate::stats::impl_generic::{
    differential_entropy_impl, entropy_impl, kl_divergence_impl, mutual_information_impl,
};
use crate::stats::traits::InformationTheoryAlgorithms;
use numr::error::Result;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl InformationTheoryAlgorithms<CudaRuntime> for CudaClient {
    fn entropy(&self, pk: &Tensor<CudaRuntime>, base: Option<f64>) -> Result<Tensor<CudaRuntime>> {
        entropy_impl(self, pk, base)
    }

    fn differential_entropy(
        &self,
        x: &Tensor<CudaRuntime>,
        k: usize,
    ) -> Result<Tensor<CudaRuntime>> {
        differential_entropy_impl(self, x, k)
    }

    fn kl_divergence(
        &self,
        pk: &Tensor<CudaRuntime>,
        qk: &Tensor<CudaRuntime>,
        base: Option<f64>,
    ) -> Result<Tensor<CudaRuntime>> {
        kl_divergence_impl(self, pk, qk, base)
    }

    fn mutual_information(
        &self,
        x: &Tensor<CudaRuntime>,
        y: &Tensor<CudaRuntime>,
        bins: usize,
        base: Option<f64>,
    ) -> Result<Tensor<CudaRuntime>> {
        mutual_information_impl(self, x, y, bins, base)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stats::helpers::extract_scalar;
    use numr::runtime::cuda::CudaDevice;

    fn setup() -> Option<(CudaClient, CudaDevice)> {
        let device = CudaDevice::new(0).ok()?;
        let client = CudaClient::new(device.clone());
        Some((client, device))
    }

    #[test]
    fn test_entropy_cuda() {
        let Some((client, device)) = setup() else {
            eprintln!("Skipping CUDA test: no device available");
            return;
        };

        let pk = Tensor::<CudaRuntime>::from_slice(&[0.25f64, 0.25, 0.25, 0.25], &[4], &device);
        let result = client.entropy(&pk, None).unwrap();
        let val = extract_scalar(&result).unwrap();
        assert!((val - 4.0_f64.ln()).abs() < 1e-10);
    }
}
