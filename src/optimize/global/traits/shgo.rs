//! SHGO (Simplicial Homology Global Optimization) trait.

use super::GlobalOptions;
use numr::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Result from SHGO optimization.
#[derive(Debug, Clone)]
pub struct ShgoResult<R: Runtime> {
    /// Global minimum point.
    pub x: Tensor<R>,
    /// Global minimum value.
    pub fun: f64,
    /// All local minima found (x, fun).
    pub local_minima: Vec<(Tensor<R>, f64)>,
    /// Number of function evaluations.
    pub nfev: usize,
    /// Whether converged.
    pub converged: bool,
}

/// SHGO algorithm trait.
pub trait ShgoAlgorithms<R: Runtime> {
    /// Simplicial Homology Global Optimization.
    ///
    /// Finds the global minimum by sampling the domain, building a simplicial
    /// complex, and running local minimizers at promising points.
    fn shgo<F>(
        &self,
        f: F,
        lower_bounds: &Tensor<R>,
        upper_bounds: &Tensor<R>,
        options: &GlobalOptions,
    ) -> Result<ShgoResult<R>>
    where
        F: Fn(&Tensor<R>) -> Result<f64>;
}
