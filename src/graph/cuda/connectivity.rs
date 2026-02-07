//! CUDA implementation of connectivity algorithms.

use crate::graph::impl_generic::{connected_components_impl, tarjan_impl};
use crate::graph::traits::connectivity::ConnectivityAlgorithms;
use crate::graph::traits::types::{ComponentResult, GraphData};
use numr::error::Result;
use numr::runtime::cuda::{CudaClient, CudaRuntime};

impl ConnectivityAlgorithms<CudaRuntime> for CudaClient {
    fn connected_components(
        &self,
        graph: &GraphData<CudaRuntime>,
    ) -> Result<ComponentResult<CudaRuntime>> {
        connected_components_impl(self, graph)
    }

    fn strongly_connected_components(
        &self,
        graph: &GraphData<CudaRuntime>,
    ) -> Result<ComponentResult<CudaRuntime>> {
        tarjan_impl(self, graph)
    }

    fn is_connected(&self, graph: &GraphData<CudaRuntime>) -> Result<bool> {
        let result = connected_components_impl(self, graph)?;
        Ok(result.num_components == 1)
    }

    fn is_strongly_connected(&self, graph: &GraphData<CudaRuntime>) -> Result<bool> {
        let result = tarjan_impl(self, graph)?;
        Ok(result.num_components == 1)
    }
}
