//! Basin-hopping algorithm trait.

use super::GlobalOptions;
use numr::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Tensor-based result from basin-hopping.
#[derive(Debug, Clone)]
pub struct BasinHoppingResult<R: Runtime> {
    pub x: Tensor<R>,
    pub fun: f64,
    pub iterations: usize,
    pub nfev: usize,
    pub converged: bool,
}

/// Basin-hopping algorithm trait.
pub trait BasinHoppingAlgorithms<R: Runtime> {
    /// Basin-hopping global optimizer.
    ///
    /// Combines local minimization with random perturbations to escape local minima.
    fn basinhopping<F>(
        &self,
        f: F,
        x0: &Tensor<R>,
        lower_bounds: &Tensor<R>,
        upper_bounds: &Tensor<R>,
        options: &GlobalOptions,
    ) -> Result<BasinHoppingResult<R>>
    where
        F: Fn(&Tensor<R>) -> Result<f64>;
}
