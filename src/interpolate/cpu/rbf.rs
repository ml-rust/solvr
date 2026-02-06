use crate::interpolate::error::InterpolateResult;
use crate::interpolate::impl_generic::rbf::{rbf_evaluate_impl, rbf_fit_impl};
use crate::interpolate::traits::rbf::{RbfAlgorithms, RbfKernel, RbfModel};
use numr::algorithm::linalg::LinearAlgebraAlgorithms;
use numr::ops::{CompareOps, MatmulOps, ScalarOps, ShapeOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

impl<
    R: Runtime,
    C: TensorOps<R>
        + ScalarOps<R>
        + CompareOps<R>
        + MatmulOps<R>
        + ShapeOps<R>
        + LinearAlgebraAlgorithms<R>
        + RuntimeClient<R>,
> RbfAlgorithms<R> for C
{
    fn rbf_fit(
        &self,
        points: &Tensor<R>,
        values: &Tensor<R>,
        kernel: RbfKernel,
        epsilon: Option<f64>,
        smoothing: f64,
    ) -> InterpolateResult<RbfModel<R>> {
        rbf_fit_impl(self, points, values, kernel, epsilon, smoothing)
    }

    fn rbf_evaluate(&self, model: &RbfModel<R>, query: &Tensor<R>) -> InterpolateResult<Tensor<R>> {
        rbf_evaluate_impl(self, model, query)
    }
}
