//! Options for global optimization algorithms.

/// Options for global optimization.
#[derive(Debug, Clone)]
pub struct GlobalOptions {
    /// Maximum number of iterations/generations
    pub max_iter: usize,
    /// Tolerance for convergence
    pub tol: f64,
    /// Random seed (None for random)
    pub seed: Option<u64>,
}

impl Default for GlobalOptions {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            tol: 1e-8,
            seed: None,
        }
    }
}
