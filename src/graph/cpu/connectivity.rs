//! CPU implementation of connectivity algorithms.

use crate::graph::impl_generic::{connected_components_impl, tarjan_impl};
use crate::graph::traits::connectivity::ConnectivityAlgorithms;
use crate::graph::traits::types::{ComponentResult, GraphData};
use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};

impl ConnectivityAlgorithms<CpuRuntime> for CpuClient {
    fn connected_components(
        &self,
        graph: &GraphData<CpuRuntime>,
    ) -> Result<ComponentResult<CpuRuntime>> {
        connected_components_impl(self, graph)
    }

    fn strongly_connected_components(
        &self,
        graph: &GraphData<CpuRuntime>,
    ) -> Result<ComponentResult<CpuRuntime>> {
        tarjan_impl(self, graph)
    }

    fn is_connected(&self, graph: &GraphData<CpuRuntime>) -> Result<bool> {
        let result = connected_components_impl(self, graph)?;
        Ok(result.num_components == 1)
    }

    fn is_strongly_connected(&self, graph: &GraphData<CpuRuntime>) -> Result<bool> {
        let result = tarjan_impl(self, graph)?;
        Ok(result.num_components == 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::CpuDevice;

    #[test]
    fn test_connected_components() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());

        // Two components: {0,1} and {2,3}
        let graph =
            GraphData::from_edge_list::<f64>(&[0, 2], &[1, 3], None, 4, false, &device).unwrap();

        let result = client.connected_components(&graph).unwrap();
        assert_eq!(result.num_components, 2);
    }

    #[test]
    fn test_is_connected() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());

        let graph =
            GraphData::from_edge_list::<f64>(&[0, 1], &[1, 2], None, 3, false, &device).unwrap();

        assert!(client.is_connected(&graph).unwrap());
    }
}
