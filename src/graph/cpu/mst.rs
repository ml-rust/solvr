//! CPU implementation of minimum spanning tree algorithms.

use crate::graph::impl_generic::kruskal_impl;
use crate::graph::traits::mst::MSTAlgorithms;
use crate::graph::traits::types::{GraphData, MSTResult};
use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};

impl MSTAlgorithms<CpuRuntime> for CpuClient {
    fn minimum_spanning_tree(
        &self,
        graph: &GraphData<CpuRuntime>,
    ) -> Result<MSTResult<CpuRuntime>> {
        kruskal_impl(self, graph)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::CpuDevice;

    #[test]
    fn test_mst() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());

        // Triangle: 0-1 (1), 1-2 (2), 0-2 (3)
        let graph = GraphData::from_edge_list::<f64>(
            &[0, 1, 0],
            &[1, 2, 2],
            Some(&[1.0, 2.0, 3.0]),
            3,
            false,
            &device,
        )
        .unwrap();

        let result = client.minimum_spanning_tree(&graph).unwrap();
        assert!((result.total_weight - 3.0).abs() < 1e-10); // edges 1+2=3
    }
}
