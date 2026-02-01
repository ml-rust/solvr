//! N-dimensional grid interpolation for WebGPU runtime (delegates to generic implementation)

use crate::interpolate::error::InterpolateResult;
use crate::interpolate::impl_generic::interpnd::interpnd_evaluate;
use crate::interpolate::traits::interpnd::{ExtrapolateMode, InterpNdAlgorithms, InterpNdMethod};
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

impl<R: Runtime, C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>> InterpNdAlgorithms<R> for C {
    fn interpnd(
        &self,
        points: &[&Tensor<R>],
        values: &Tensor<R>,
        xi: &Tensor<R>,
        method: InterpNdMethod,
        extrapolate: ExtrapolateMode,
    ) -> InterpolateResult<Tensor<R>> {
        interpnd_evaluate(self, points, values, xi, method, extrapolate)
    }
}
