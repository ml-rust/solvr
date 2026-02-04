//! CPU implementation of BallTree algorithms.

use crate::spatial::impl_generic::{
    balltree_build_impl, balltree_query_impl, balltree_query_radius_impl,
};
use crate::spatial::traits::balltree::{BallTree, BallTreeAlgorithms, BallTreeOptions};
use crate::spatial::traits::kdtree::{KNNResult, RadiusResult};
use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl BallTreeAlgorithms<CpuRuntime> for CpuClient {
    fn balltree_build(
        &self,
        points: &Tensor<CpuRuntime>,
        options: BallTreeOptions,
    ) -> Result<BallTree<CpuRuntime>> {
        balltree_build_impl(self, points, options)
    }

    fn balltree_query(
        &self,
        tree: &BallTree<CpuRuntime>,
        query: &Tensor<CpuRuntime>,
        k: usize,
    ) -> Result<KNNResult<CpuRuntime>> {
        balltree_query_impl(self, tree, query, k)
    }

    fn balltree_query_radius(
        &self,
        tree: &BallTree<CpuRuntime>,
        query: &Tensor<CpuRuntime>,
        radius: f64,
    ) -> Result<RadiusResult<CpuRuntime>> {
        balltree_query_radius_impl(self, tree, query, radius)
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
    fn test_balltree_build() {
        let (client, device) = setup();

        let points = Tensor::<CpuRuntime>::from_slice(
            &[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            &[4, 2],
            &device,
        );

        let tree = client
            .balltree_build(&points, BallTreeOptions::default())
            .unwrap();

        assert_eq!(tree.data.shape(), &[4, 2]);
    }

    #[test]
    fn test_balltree_query() {
        let (client, device) = setup();

        let points = Tensor::<CpuRuntime>::from_slice(
            &[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            &[4, 2],
            &device,
        );

        let tree = client
            .balltree_build(&points, BallTreeOptions::default())
            .unwrap();

        let query = Tensor::<CpuRuntime>::from_slice(&[0.1, 0.1], &[1, 2], &device);

        let result = client.balltree_query(&tree, &query, 2).unwrap();

        assert_eq!(result.distances.shape(), &[1, 2]);
        assert_eq!(result.indices.shape(), &[1, 2]);

        let indices: Vec<i64> = result.indices.to_vec();
        assert_eq!(indices[0], 0);
    }
}
