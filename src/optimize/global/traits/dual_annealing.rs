//! Dual annealing algorithm trait.

use super::GlobalOptions;
use numr::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Tensor-based result from dual annealing.
#[derive(Debug, Clone)]
pub struct DualAnnealingResult<R: Runtime> {
    pub x: Tensor<R>,
    pub fun: f64,
    pub iterations: usize,
    pub nfev: usize,
    pub converged: bool,
}

/// Dual annealing algorithm trait.
pub trait DualAnnealingAlgorithms<R: Runtime> {
    /// Dual annealing global optimizer.
    ///
    /// Combines simulated annealing with local search for smooth functions.
    fn dual_annealing<F>(
        &self,
        f: F,
        lower_bounds: &Tensor<R>,
        upper_bounds: &Tensor<R>,
        options: &GlobalOptions,
    ) -> Result<DualAnnealingResult<R>>
    where
        F: Fn(&Tensor<R>) -> Result<f64>;
}
