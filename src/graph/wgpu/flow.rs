//! WebGPU implementation of network flow algorithms.

use crate::graph::impl_generic::{max_flow_impl, min_cost_flow_impl};
use crate::graph::traits::flow::FlowAlgorithms;
use crate::graph::traits::types::{FlowResult, GraphData, MinCostFlowOptions};
use numr::error::Result;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};

impl FlowAlgorithms<WgpuRuntime> for WgpuClient {
    fn max_flow(
        &self,
        graph: &GraphData<WgpuRuntime>,
        source: usize,
        sink: usize,
    ) -> Result<FlowResult<WgpuRuntime>> {
        max_flow_impl(self, graph, source, sink)
    }

    fn min_cost_flow(
        &self,
        graph: &GraphData<WgpuRuntime>,
        source: usize,
        sink: usize,
        options: &MinCostFlowOptions,
    ) -> Result<FlowResult<WgpuRuntime>> {
        min_cost_flow_impl(self, graph, source, sink, options)
    }
}
