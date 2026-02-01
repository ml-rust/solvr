use crate::interpolate::error::InterpolateResult;
use crate::interpolate::impl_generic::cubic_spline::cubic_spline_coefficients;
use crate::interpolate::traits::cubic_spline::{CubicSplineAlgorithms, SplineBoundary};
use numr::ops::ScalarOps;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

impl<R: Runtime, C: ScalarOps<R> + RuntimeClient<R>> CubicSplineAlgorithms<R> for C {
    fn cubic_spline_coefficients(
        &self,
        x: &Tensor<R>,
        y: &Tensor<R>,
        boundary: &SplineBoundary,
    ) -> InterpolateResult<(Tensor<R>, Tensor<R>, Tensor<R>, Tensor<R>)> {
        cubic_spline_coefficients(self, x, y, boundary)
    }
}
