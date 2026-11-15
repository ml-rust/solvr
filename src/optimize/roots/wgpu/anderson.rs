//! WebGPU implementation of Anderson mixing.

use crate::optimize::roots::RootTensorResult;
use crate::optimize::roots::impl_generic::{anderson_impl, anderson_with_convergence_impl};
use crate::optimize::roots::traits::anderson::{
    AndersonAlgorithms, AndersonConvergence, AndersonOptions,
};
use numr::error::Result;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl AndersonAlgorithms<WgpuRuntime> for WgpuClient {
    fn anderson<G>(
        &self,
        g: G,
        x0: &Tensor<WgpuRuntime>,
        options: &AndersonOptions,
    ) -> Result<RootTensorResult<WgpuRuntime>>
    where
        G: Fn(&Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>>,
    {
        let result = anderson_impl(self, g, x0, options).map_err(|e| {
            numr::error::Error::backend_limitation("wgpu", "anderson", e.to_string())
        })?;
        Ok(result)
    }

    fn anderson_with_convergence<G, K>(
        &self,
        g: G,
        x0: &Tensor<WgpuRuntime>,
        options: &AndersonOptions,
        convergence: K,
    ) -> Result<RootTensorResult<WgpuRuntime>>
    where
        G: Fn(&Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>>,
        K: Fn(
            &Tensor<WgpuRuntime>,
            &Tensor<WgpuRuntime>,
            &Tensor<WgpuRuntime>,
        ) -> Result<AndersonConvergence>,
    {
        anderson_with_convergence_impl(self, g, x0, options, convergence).map_err(|e| {
            numr::error::Error::backend_limitation(
                "wgpu",
                "anderson_with_convergence",
                e.to_string(),
            )
        })
    }
}
