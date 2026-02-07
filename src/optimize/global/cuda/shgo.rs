//! CUDA implementation of SHGO.

use numr::error::Result;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use crate::optimize::global::GlobalOptions;
use crate::optimize::global::impl_generic::shgo::shgo_impl;
use crate::optimize::global::traits::ShgoAlgorithms;
use crate::optimize::global::traits::shgo::ShgoResult;

impl ShgoAlgorithms<CudaRuntime> for CudaClient {
    fn shgo<F>(
        &self,
        f: F,
        lower_bounds: &Tensor<CudaRuntime>,
        upper_bounds: &Tensor<CudaRuntime>,
        options: &GlobalOptions,
    ) -> Result<ShgoResult<CudaRuntime>>
    where
        F: Fn(&Tensor<CudaRuntime>) -> Result<f64>,
    {
        let result = shgo_impl(self, f, lower_bounds, upper_bounds, options)
            .map_err(|e| numr::error::Error::backend_limitation("cuda", "shgo", e.to_string()))?;
        Ok(ShgoResult {
            x: result.x,
            fun: result.fun,
            local_minima: result.local_minima,
            nfev: result.nfev,
            converged: result.converged,
        })
    }
}
