//! WebGPU implementation of SHGO.

use numr::error::Result;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

use crate::optimize::global::GlobalOptions;
use crate::optimize::global::impl_generic::shgo::shgo_impl;
use crate::optimize::global::traits::ShgoAlgorithms;
use crate::optimize::global::traits::shgo::ShgoResult;

impl ShgoAlgorithms<WgpuRuntime> for WgpuClient {
    fn shgo<F>(
        &self,
        f: F,
        lower_bounds: &Tensor<WgpuRuntime>,
        upper_bounds: &Tensor<WgpuRuntime>,
        options: &GlobalOptions,
    ) -> Result<ShgoResult<WgpuRuntime>>
    where
        F: Fn(&Tensor<WgpuRuntime>) -> Result<f64>,
    {
        let result = shgo_impl(self, f, lower_bounds, upper_bounds, options)
            .map_err(|e| numr::error::Error::backend_limitation("wgpu", "shgo", e.to_string()))?;
        Ok(ShgoResult {
            x: result.x,
            fun: result.fun,
            local_minima: result.local_minima,
            nfev: result.nfev,
            converged: result.converged,
        })
    }
}
