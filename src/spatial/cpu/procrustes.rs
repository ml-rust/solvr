//! CPU implementation of Procrustes analysis.

use crate::spatial::impl_generic::{orthogonal_procrustes_impl, procrustes_impl};
use crate::spatial::traits::procrustes::{ProcrustesAlgorithms, ProcrustesResult};
use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl ProcrustesAlgorithms<CpuRuntime> for CpuClient {
    fn procrustes(
        &self,
        source: &Tensor<CpuRuntime>,
        target: &Tensor<CpuRuntime>,
        scaling: bool,
        reflection: bool,
    ) -> Result<ProcrustesResult<CpuRuntime>> {
        procrustes_impl(self, source, target, scaling, reflection)
    }

    fn orthogonal_procrustes(
        &self,
        a: &Tensor<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
    ) -> Result<(Tensor<CpuRuntime>, f64)> {
        orthogonal_procrustes_impl(self, a, b)
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
    fn test_procrustes_identical() {
        let (client, device) = setup();

        let points =
            Tensor::<CpuRuntime>::from_slice(&[0.0, 0.0, 1.0, 0.0, 0.5, 1.0], &[3, 2], &device);

        let result = client.procrustes(&points, &points, false, false).unwrap();

        // Identical point sets should have near-zero disparity
        assert!(result.disparity < 1e-10);
        assert!((result.scale - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_procrustes_translation() {
        let (client, device) = setup();

        let source =
            Tensor::<CpuRuntime>::from_slice(&[0.0, 0.0, 1.0, 0.0, 0.5, 1.0], &[3, 2], &device);

        // Target is source shifted by (2, 3)
        let target =
            Tensor::<CpuRuntime>::from_slice(&[2.0, 3.0, 3.0, 3.0, 2.5, 4.0], &[3, 2], &device);

        let result = client.procrustes(&source, &target, false, false).unwrap();

        // Should find pure translation
        let translation: Vec<f64> = result.translation.to_vec();
        assert!((translation[0] - 2.0).abs() < 1e-6);
        assert!((translation[1] - 3.0).abs() < 1e-6);
        assert!(result.disparity < 1e-10);
    }

    #[test]
    fn test_procrustes_with_scaling() {
        let (client, device) = setup();

        let source =
            Tensor::<CpuRuntime>::from_slice(&[0.0, 0.0, 1.0, 0.0, 0.5, 1.0], &[3, 2], &device);

        // Target is source scaled by 2
        let target =
            Tensor::<CpuRuntime>::from_slice(&[0.0, 0.0, 2.0, 0.0, 1.0, 2.0], &[3, 2], &device);

        let result = client.procrustes(&source, &target, true, false).unwrap();

        // Should find scale = 2
        assert!((result.scale - 2.0).abs() < 1e-6);
        assert!(result.disparity < 1e-10);
    }

    #[test]
    fn test_orthogonal_procrustes() {
        let (client, device) = setup();

        // Identity test: A @ R should equal B when R = I
        let a = Tensor::<CpuRuntime>::from_slice(
            &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            &[4, 3],
            &device,
        );

        let b = a.clone();

        let (r, residual) = client.orthogonal_procrustes(&a, &b).unwrap();

        // R should be identity (or close to it)
        assert_eq!(r.shape(), &[3, 3]);
        assert!(residual < 1e-10);
    }
}
