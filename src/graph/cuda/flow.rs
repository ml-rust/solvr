//! CUDA implementation of network flow algorithms.

use crate::graph::impl_generic::{max_flow_impl, min_cost_flow_impl};
use crate::graph::traits::flow::FlowAlgorithms;
use crate::graph::traits::types::{FlowResult, GraphData, MinCostFlowOptions};
use numr::error::Result;
use numr::runtime::cuda::{CudaClient, CudaRuntime};

impl FlowAlgorithms<CudaRuntime> for CudaClient {
    fn max_flow(
        &self,
        graph: &GraphData<CudaRuntime>,
        source: usize,
        sink: usize,
    ) -> Result<FlowResult<CudaRuntime>> {
        max_flow_impl(self, graph, source, sink)
    }

    fn min_cost_flow(
        &self,
        graph: &GraphData<CudaRuntime>,
        source: usize,
        sink: usize,
        options: &MinCostFlowOptions,
    ) -> Result<FlowResult<CudaRuntime>> {
        min_cost_flow_impl(self, graph, source, sink, options)
    }
}
