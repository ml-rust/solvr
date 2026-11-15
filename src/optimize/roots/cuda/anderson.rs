//! CUDA implementation of Anderson mixing.

use crate::optimize::roots::RootTensorResult;
use crate::optimize::roots::impl_generic::{anderson_impl, anderson_with_convergence_impl};
use crate::optimize::roots::traits::anderson::{
    AndersonAlgorithms, AndersonConvergence, AndersonOptions,
};
use numr::error::Result;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl AndersonAlgorithms<CudaRuntime> for CudaClient {
    fn anderson<G>(
        &self,
        g: G,
        x0: &Tensor<CudaRuntime>,
        options: &AndersonOptions,
    ) -> Result<RootTensorResult<CudaRuntime>>
    where
        G: Fn(&Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>>,
    {
        let result = anderson_impl(self, g, x0, options).map_err(|e| {
            numr::error::Error::backend_limitation("cuda", "anderson", e.to_string())
        })?;
        Ok(result)
    }

    fn anderson_with_convergence<G, K>(
        &self,
        g: G,
        x0: &Tensor<CudaRuntime>,
        options: &AndersonOptions,
        convergence: K,
    ) -> Result<RootTensorResult<CudaRuntime>>
    where
        G: Fn(&Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>>,
        K: Fn(
            &Tensor<CudaRuntime>,
            &Tensor<CudaRuntime>,
            &Tensor<CudaRuntime>,
        ) -> Result<AndersonConvergence>,
    {
        anderson_with_convergence_impl(self, g, x0, options, convergence).map_err(|e| {
            numr::error::Error::backend_limitation(
                "cuda",
                "anderson_with_convergence",
                e.to_string(),
            )
        })
    }
}
