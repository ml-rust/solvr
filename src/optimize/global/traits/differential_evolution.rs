//! Differential evolution algorithm trait.

use super::GlobalOptions;
use numr::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Tensor-based result from differential evolution.
#[derive(Debug, Clone)]
pub struct DifferentialEvolutionResult<R: Runtime> {
    pub x: Tensor<R>,
    pub fun: f64,
    pub iterations: usize,
    pub nfev: usize,
    pub converged: bool,
}

/// Differential evolution algorithm trait.
pub trait DifferentialEvolutionAlgorithms<R: Runtime> {
    /// Differential Evolution global optimizer.
    ///
    /// Population-based optimizer using mutation, crossover, and selection.
    fn differential_evolution<F>(
        &self,
        f: F,
        lower_bounds: &Tensor<R>,
        upper_bounds: &Tensor<R>,
        options: &GlobalOptions,
    ) -> Result<DifferentialEvolutionResult<R>>
    where
        F: Fn(&Tensor<R>) -> Result<f64>;
}
