//! WebGPU implementation of MST algorithms.

use crate::graph::impl_generic::kruskal_impl;
use crate::graph::traits::mst::MSTAlgorithms;
use crate::graph::traits::types::{GraphData, MSTResult};
use numr::error::Result;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};

impl MSTAlgorithms<WgpuRuntime> for WgpuClient {
    fn minimum_spanning_tree(
        &self,
        graph: &GraphData<WgpuRuntime>,
    ) -> Result<MSTResult<WgpuRuntime>> {
        kruskal_impl(self, graph)
    }
}
