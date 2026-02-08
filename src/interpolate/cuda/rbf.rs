use crate::interpolate::error::InterpolateResult;
use crate::interpolate::impl_generic::rbf::{rbf_evaluate_impl, rbf_fit_impl};
use crate::interpolate::traits::rbf::{RbfAlgorithms, RbfKernel, RbfModel};
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl RbfAlgorithms<CudaRuntime> for CudaClient {
    fn rbf_fit(
        &self,
        points: &Tensor<CudaRuntime>,
        values: &Tensor<CudaRuntime>,
        kernel: RbfKernel,
        epsilon: Option<f64>,
        smoothing: f64,
    ) -> InterpolateResult<RbfModel<CudaRuntime>> {
        rbf_fit_impl(self, points, values, kernel, epsilon, smoothing)
    }

    fn rbf_evaluate(
        &self,
        model: &RbfModel<CudaRuntime>,
        query: &Tensor<CudaRuntime>,
    ) -> InterpolateResult<Tensor<CudaRuntime>> {
        rbf_evaluate_impl(self, model, query)
    }
}
