use crate::interpolate::error::InterpolateResult;
use crate::interpolate::impl_generic::rbf::{rbf_evaluate_impl, rbf_fit_impl};
use crate::interpolate::traits::rbf::{RbfAlgorithms, RbfKernel, RbfModel};
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl RbfAlgorithms<WgpuRuntime> for WgpuClient {
    fn rbf_fit(
        &self,
        points: &Tensor<WgpuRuntime>,
        values: &Tensor<WgpuRuntime>,
        kernel: RbfKernel,
        epsilon: Option<f64>,
        smoothing: f64,
    ) -> InterpolateResult<RbfModel<WgpuRuntime>> {
        rbf_fit_impl(self, points, values, kernel, epsilon, smoothing)
    }

    fn rbf_evaluate(
        &self,
        model: &RbfModel<WgpuRuntime>,
        query: &Tensor<WgpuRuntime>,
    ) -> InterpolateResult<Tensor<WgpuRuntime>> {
        rbf_evaluate_impl(self, model, query)
    }
}
