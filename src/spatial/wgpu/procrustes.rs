//! WebGPU implementation of Procrustes analysis.

use crate::spatial::impl_generic::{orthogonal_procrustes_impl, procrustes_impl};
use crate::spatial::traits::procrustes::{ProcrustesAlgorithms, ProcrustesResult};
use numr::error::Result;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl ProcrustesAlgorithms<WgpuRuntime> for WgpuClient {
    fn procrustes(
        &self,
        source: &Tensor<WgpuRuntime>,
        target: &Tensor<WgpuRuntime>,
        scaling: bool,
        reflection: bool,
    ) -> Result<ProcrustesResult<WgpuRuntime>> {
        procrustes_impl(self, source, target, scaling, reflection)
    }

    fn orthogonal_procrustes(
        &self,
        a: &Tensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
    ) -> Result<(Tensor<WgpuRuntime>, f64)> {
        orthogonal_procrustes_impl(self, a, b)
    }
}
