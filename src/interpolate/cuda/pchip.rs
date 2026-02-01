//! PCHIP interpolation for CUDA runtime (delegates to generic implementation)

use crate::interpolate::error::InterpolateResult;
use crate::interpolate::impl_generic::pchip::pchip_slopes;
use crate::interpolate::traits::pchip::PchipAlgorithms;
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

impl<R: Runtime, C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>> PchipAlgorithms<R> for C {
    fn pchip_slopes(&self, x: &Tensor<R>, y: &Tensor<R>) -> InterpolateResult<Tensor<R>> {
        pchip_slopes(self, x, y)
    }
}
