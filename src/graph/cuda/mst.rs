//! CUDA implementation of MST algorithms.

use crate::graph::impl_generic::kruskal_impl;
use crate::graph::traits::mst::MSTAlgorithms;
use crate::graph::traits::types::{GraphData, MSTResult};
use numr::error::Result;
use numr::runtime::cuda::{CudaClient, CudaRuntime};

impl MSTAlgorithms<CudaRuntime> for CudaClient {
    fn minimum_spanning_tree(
        &self,
        graph: &GraphData<CudaRuntime>,
    ) -> Result<MSTResult<CudaRuntime>> {
        kruskal_impl(self, graph)
    }
}
