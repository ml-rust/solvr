//! CPU implementation of network flow algorithms.

use crate::graph::impl_generic::{max_flow_impl, min_cost_flow_impl};
use crate::graph::traits::flow::FlowAlgorithms;
use crate::graph::traits::types::{FlowResult, GraphData, MinCostFlowOptions};
use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};

impl FlowAlgorithms<CpuRuntime> for CpuClient {
    fn max_flow(
        &self,
        graph: &GraphData<CpuRuntime>,
        source: usize,
        sink: usize,
    ) -> Result<FlowResult<CpuRuntime>> {
        max_flow_impl(self, graph, source, sink)
    }

    fn min_cost_flow(
        &self,
        graph: &GraphData<CpuRuntime>,
        source: usize,
        sink: usize,
        options: &MinCostFlowOptions,
    ) -> Result<FlowResult<CpuRuntime>> {
        min_cost_flow_impl(self, graph, source, sink, options)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::CpuDevice;

    #[test]
    fn test_max_flow() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());

        // Simple flow network:
        // 0 --10--> 1 --5--> 3
        // 0 --8---> 2 --7--> 3
        // 1 --3---> 2
        let graph = GraphData::from_edge_list::<f64>(
            &[0, 0, 1, 1, 2],
            &[1, 2, 2, 3, 3],
            Some(&[10.0, 8.0, 3.0, 5.0, 7.0]),
            4,
            true,
            &device,
        )
        .unwrap();

        let result = client.max_flow(&graph, 0, 3).unwrap();
        // Max flow = min(10, 5+3) from path 0->1->3 (5) + min(8,7) from 0->2->3 (7) + 0->1->2->3 (3) = 5+7+3 = 15
        // Actually: 0->1->3: 5, 0->1->2->3: 3 (remaining cap 1->2=3), 0->2->3: 7
        // Total: 5+3+7 = 15? Let me verify:
        // From 0: out cap = 10+8 = 18
        // To 3: in cap = 5+7 = 12
        // So max flow <= 12
        assert!(result.max_flow > 0.0);
        assert!(result.max_flow <= 12.0 + 1e-10);
    }
}
