//! Simulated annealing algorithm trait.

use super::GlobalOptions;
use numr::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Tensor-based result from simulated annealing.
#[derive(Debug, Clone)]
pub struct SimulatedAnnealingResult<R: Runtime> {
    pub x: Tensor<R>,
    pub fun: f64,
    pub iterations: usize,
    pub nfev: usize,
    pub converged: bool,
}

/// Simulated annealing algorithm trait.
pub trait SimulatedAnnealingAlgorithms<R: Runtime> {
    /// Simulated annealing global optimizer.
    ///
    /// Uses probabilistic acceptance of worse solutions to escape local minima.
    fn simulated_annealing<F>(
        &self,
        f: F,
        lower_bounds: &Tensor<R>,
        upper_bounds: &Tensor<R>,
        options: &GlobalOptions,
    ) -> Result<SimulatedAnnealingResult<R>>
    where
        F: Fn(&Tensor<R>) -> Result<f64>;
}
