//! WebGPU implementation of connectivity algorithms.

use crate::graph::impl_generic::{connected_components_impl, tarjan_impl};
use crate::graph::traits::connectivity::ConnectivityAlgorithms;
use crate::graph::traits::types::{ComponentResult, GraphData};
use numr::error::Result;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};

impl ConnectivityAlgorithms<WgpuRuntime> for WgpuClient {
    fn connected_components(
        &self,
        graph: &GraphData<WgpuRuntime>,
    ) -> Result<ComponentResult<WgpuRuntime>> {
        connected_components_impl(self, graph)
    }

    fn strongly_connected_components(
        &self,
        graph: &GraphData<WgpuRuntime>,
    ) -> Result<ComponentResult<WgpuRuntime>> {
        tarjan_impl(self, graph)
    }

    fn is_connected(&self, graph: &GraphData<WgpuRuntime>) -> Result<bool> {
        let result = connected_components_impl(self, graph)?;
        Ok(result.num_components == 1)
    }

    fn is_strongly_connected(&self, graph: &GraphData<WgpuRuntime>) -> Result<bool> {
        let result = tarjan_impl(self, graph)?;
        Ok(result.num_components == 1)
    }
}
