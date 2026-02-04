//! CUDA implementation of Procrustes analysis.

use crate::spatial::impl_generic::{orthogonal_procrustes_impl, procrustes_impl};
use crate::spatial::traits::procrustes::{ProcrustesAlgorithms, ProcrustesResult};
use numr::error::Result;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl ProcrustesAlgorithms<CudaRuntime> for CudaClient {
    fn procrustes(
        &self,
        source: &Tensor<CudaRuntime>,
        target: &Tensor<CudaRuntime>,
        scaling: bool,
        reflection: bool,
    ) -> Result<ProcrustesResult<CudaRuntime>> {
        procrustes_impl(self, source, target, scaling, reflection)
    }

    fn orthogonal_procrustes(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
    ) -> Result<(Tensor<CudaRuntime>, f64)> {
        orthogonal_procrustes_impl(self, a, b)
    }
}
