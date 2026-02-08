use crate::interpolate::error::InterpolateResult;
use crate::interpolate::impl_generic::rbf::{rbf_evaluate_impl, rbf_fit_impl};
use crate::interpolate::traits::rbf::{RbfAlgorithms, RbfKernel, RbfModel};
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl RbfAlgorithms<CpuRuntime> for CpuClient {
    fn rbf_fit(
        &self,
        points: &Tensor<CpuRuntime>,
        values: &Tensor<CpuRuntime>,
        kernel: RbfKernel,
        epsilon: Option<f64>,
        smoothing: f64,
    ) -> InterpolateResult<RbfModel<CpuRuntime>> {
        rbf_fit_impl(self, points, values, kernel, epsilon, smoothing)
    }

    fn rbf_evaluate(
        &self,
        model: &RbfModel<CpuRuntime>,
        query: &Tensor<CpuRuntime>,
    ) -> InterpolateResult<Tensor<CpuRuntime>> {
        rbf_evaluate_impl(self, model, query)
    }
}
