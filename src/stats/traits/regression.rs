//! Regression analysis algorithms.

use crate::stats::LinregressResult;
use numr::error::Result;
use numr::ops::TensorOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Regression analysis algorithms for tensors.
///
/// Provides methods for fitting regression models to tensor data.
pub trait RegressionAlgorithms<R: Runtime>: TensorOps<R> {
    /// Simple linear regression.
    ///
    /// Fits y = slope * x + intercept using ordinary least squares.
    fn linregress(&self, x: &Tensor<R>, y: &Tensor<R>) -> Result<LinregressResult>;
}
