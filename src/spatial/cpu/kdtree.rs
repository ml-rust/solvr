//! CPU implementation of KDTree algorithms.

use crate::spatial::impl_generic::{
    kdtree_build_impl, kdtree_query_impl, kdtree_query_radius_impl,
};
use crate::spatial::traits::kdtree::{
    KDTree, KDTreeAlgorithms, KDTreeOptions, KNNResult, RadiusResult,
};
use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl KDTreeAlgorithms<CpuRuntime> for CpuClient {
    fn kdtree_build(
        &self,
        points: &Tensor<CpuRuntime>,
        options: KDTreeOptions,
    ) -> Result<KDTree<CpuRuntime>> {
        kdtree_build_impl(self, points, options)
    }

    fn kdtree_query(
        &self,
        tree: &KDTree<CpuRuntime>,
        query: &Tensor<CpuRuntime>,
        k: usize,
    ) -> Result<KNNResult<CpuRuntime>> {
        kdtree_query_impl(self, tree, query, k)
    }

    fn kdtree_query_radius(
        &self,
        tree: &KDTree<CpuRuntime>,
        query: &Tensor<CpuRuntime>,
        radius: f64,
    ) -> Result<RadiusResult<CpuRuntime>> {
        kdtree_query_radius_impl(self, tree, query, radius)
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
    fn test_kdtree_build() {
        let (client, device) = setup();

        let points = Tensor::<CpuRuntime>::from_slice(
            &[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            &[4, 2],
            &device,
        );

        let tree = client
            .kdtree_build(&points, KDTreeOptions::default())
            .unwrap();

        assert_eq!(tree.data.shape(), &[4, 2]);
    }

    #[test]
    fn test_kdtree_query() {
        let (client, device) = setup();

        let points = Tensor::<CpuRuntime>::from_slice(
            &[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            &[4, 2],
            &device,
        );

        let tree = client
            .kdtree_build(&points, KDTreeOptions::default())
            .unwrap();

        // Query at (0.1, 0.1) - closest should be (0, 0) at index 0
        let query = Tensor::<CpuRuntime>::from_slice(&[0.1, 0.1], &[1, 2], &device);

        let result = client.kdtree_query(&tree, &query, 2).unwrap();

        assert_eq!(result.distances.shape(), &[1, 2]);
        assert_eq!(result.indices.shape(), &[1, 2]);

        let indices: Vec<i64> = result.indices.to_vec();
        // Closest point should be (0,0) at index 0
        assert_eq!(indices[0], 0);
    }

    #[test]
    fn test_kdtree_query_radius() {
        let (client, device) = setup();

        let points = Tensor::<CpuRuntime>::from_slice(
            &[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            &[4, 2],
            &device,
        );

        let tree = client
            .kdtree_build(&points, KDTreeOptions::default())
            .unwrap();

        // Query at origin with radius 0.5 - should only find (0,0)
        let query = Tensor::<CpuRuntime>::from_slice(&[0.0, 0.0], &[1, 2], &device);

        let result = client.kdtree_query_radius(&tree, &query, 0.5).unwrap();

        let counts: Vec<i64> = result.counts.to_vec();
        assert_eq!(counts[0], 1);

        let indices: Vec<i64> = result.indices.to_vec();
        assert_eq!(indices[0], 0);
    }
}
