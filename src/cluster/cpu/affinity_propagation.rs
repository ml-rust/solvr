//! CPU implementation of Affinity Propagation clustering.

use crate::cluster::impl_generic::affinity_propagation_impl;
use crate::cluster::traits::affinity_propagation::{
    AffinityPropagationAlgorithms, AffinityPropagationOptions, AffinityPropagationResult,
};
use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl AffinityPropagationAlgorithms<CpuRuntime> for CpuClient {
    fn affinity_propagation(
        &self,
        similarities: &Tensor<CpuRuntime>,
        options: &AffinityPropagationOptions,
    ) -> Result<AffinityPropagationResult<CpuRuntime>> {
        affinity_propagation_impl(self, similarities, options)
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
    fn test_affinity_propagation_basic() {
        let (client, device) = setup();

        // Build a negative squared-distance similarity matrix for 6 points
        // Cluster 1: (0,0), (0.1,0.1), (0.2,0)
        // Cluster 2: (10,10), (10.1,10.1), (10.2,10)
        let points: &[(f64, f64)] = &[
            (0.0, 0.0),
            (0.1, 0.1),
            (0.2, 0.0),
            (10.0, 10.0),
            (10.1, 10.1),
            (10.2, 10.0),
        ];
        let n = points.len();
        let mut sim = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let dx = points[i].0 - points[j].0;
                let dy = points[i].1 - points[j].1;
                sim[i * n + j] = -(dx * dx + dy * dy);
            }
        }

        let similarities = Tensor::<CpuRuntime>::from_slice(&sim, &[n, n], &device);

        let options = AffinityPropagationOptions {
            damping: 0.5,
            max_iter: 200,
            convergence_iter: 15,
            preference: None,
        };

        let result = client
            .affinity_propagation(&similarities, &options)
            .unwrap();
        assert_eq!(result.labels.shape(), &[6]);
    }
}
