//! Global optimization algorithms.
//!
//! This module provides methods for finding global minima of functions,
//! avoiding local minima traps that affect local optimization methods.

mod basin_hopping;
mod differential_evolution;
mod simulated_annealing;

pub use basin_hopping::{basinhopping, dual_annealing};
pub use differential_evolution::differential_evolution;
pub use simulated_annealing::simulated_annealing;

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

/// Result from a global optimization method.
#[derive(Debug, Clone)]
pub struct GlobalResult {
    /// The global minimum point found
    pub x: Vec<f64>,
    /// Function value at minimum
    pub fun: f64,
    /// Number of iterations/generations
    pub iterations: usize,
    /// Number of function evaluations
    pub nfev: usize,
    /// Whether the method converged
    pub converged: bool,
}

/// Simple linear congruential generator for reproducible randomness.
pub(crate) struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        // LCG parameters from Numerical Recipes
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.state
    }

    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    pub fn next_range(&mut self, min: f64, max: f64) -> f64 {
        min + self.next_f64() * (max - min)
    }

    pub fn next_int(&mut self, max: usize) -> usize {
        (self.next_u64() as usize) % max
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sphere(x: &[f64]) -> f64 {
        x.iter().map(|&xi| xi * xi).sum()
    }

    #[test]
    fn test_compare_methods() {
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let opts = GlobalOptions {
            max_iter: 500,
            seed: Some(42),
            ..Default::default()
        };

        let de_result = differential_evolution(&sphere, &bounds, &opts).expect("DE failed");
        let sa_result = simulated_annealing(&sphere, &bounds, &opts).expect("SA failed");
        let da_result = dual_annealing(&sphere, &bounds, &opts).expect("DA failed");

        assert!(de_result.fun < 1e-4);
        assert!(sa_result.fun < 0.5);
        assert!(da_result.fun < 1e-4);
    }
}
