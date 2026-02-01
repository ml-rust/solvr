use crate::interpolate::error::InterpolateResult;
use crate::interpolate::impl_generic::akima::akima_slopes;
use crate::interpolate::traits::akima::AkimaAlgorithms;
use numr::ops::{CompareOps, ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

impl<R: Runtime, C: TensorOps<R> + ScalarOps<R> + CompareOps<R> + RuntimeClient<R>>
    AkimaAlgorithms<R> for C
{
    fn akima_slopes(&self, x: &Tensor<R>, y: &Tensor<R>) -> InterpolateResult<Tensor<R>> {
        akima_slopes(self, x, y)
    }
}
