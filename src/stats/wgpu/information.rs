//! WebGPU implementation of information theory algorithms.

use crate::stats::impl_generic::{
    differential_entropy_impl, entropy_impl, kl_divergence_impl, mutual_information_impl,
};
use crate::stats::traits::InformationTheoryAlgorithms;
use numr::error::Result;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl InformationTheoryAlgorithms<WgpuRuntime> for WgpuClient {
    fn entropy(&self, pk: &Tensor<WgpuRuntime>, base: Option<f64>) -> Result<Tensor<WgpuRuntime>> {
        entropy_impl(self, pk, base)
    }

    fn differential_entropy(
        &self,
        x: &Tensor<WgpuRuntime>,
        k: usize,
    ) -> Result<Tensor<WgpuRuntime>> {
        differential_entropy_impl(self, x, k)
    }

    fn kl_divergence(
        &self,
        pk: &Tensor<WgpuRuntime>,
        qk: &Tensor<WgpuRuntime>,
        base: Option<f64>,
    ) -> Result<Tensor<WgpuRuntime>> {
        kl_divergence_impl(self, pk, qk, base)
    }

    fn mutual_information(
        &self,
        x: &Tensor<WgpuRuntime>,
        y: &Tensor<WgpuRuntime>,
        bins: usize,
        base: Option<f64>,
    ) -> Result<Tensor<WgpuRuntime>> {
        mutual_information_impl(self, x, y, bins, base)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stats::helpers::extract_scalar;
    use numr::runtime::wgpu::WgpuDevice;

    fn setup() -> Option<(WgpuClient, WgpuDevice)> {
        let device = WgpuDevice::new(0);
        let client = WgpuClient::new(device.clone()).ok()?;
        Some((client, device))
    }

    #[test]
    fn test_entropy_wgpu() {
        let Some((client, device)) = setup() else {
            eprintln!("Skipping WebGPU test: no device available");
            return;
        };

        let pk = Tensor::<WgpuRuntime>::from_slice(&[0.25f32, 0.25, 0.25, 0.25], &[4], &device);
        let result = client.entropy(&pk, None).unwrap();
        let val = extract_scalar(&result).unwrap();
        assert!((val - 4.0_f64.ln()).abs() < 1e-3);
    }
}
