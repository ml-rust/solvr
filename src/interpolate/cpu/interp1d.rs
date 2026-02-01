use crate::interpolate::error::InterpolateResult;
use crate::interpolate::impl_generic::interp1d::interp1d_evaluate;
use crate::interpolate::traits::interp1d::{Interp1dAlgorithms, InterpMethod};
use numr::ops::{CompareOps, ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

impl<R: Runtime, C: TensorOps<R> + ScalarOps<R> + CompareOps<R> + RuntimeClient<R>>
    Interp1dAlgorithms<R> for C
{
    fn interp1d(
        &self,
        x: &Tensor<R>,
        y: &Tensor<R>,
        x_new: &Tensor<R>,
        method: InterpMethod,
    ) -> InterpolateResult<Tensor<R>> {
        interp1d_evaluate(self, x, y, x_new, method)
    }
}
