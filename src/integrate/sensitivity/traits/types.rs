//! Types for adjoint sensitivity analysis.
//!
//! These types configure and store results from computing parameter gradients
//! via backward integration of the adjoint ODE.

use numr::runtime::Runtime;
use numr::tensor::Tensor;

use crate::integrate::ODEMethod;

/// Strategy for checkpoint placement during forward integration.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum CheckpointStrategy {
    /// Checkpoints placed at uniform time intervals.
    #[default]
    Uniform,

    /// Checkpoints placed at logarithmically spaced intervals.
    /// More checkpoints near t=0, fewer near t=T.
    Logarithmic,

    /// Checkpoints placed adaptively based on solution variation.
    Adaptive,
}

/// Options for adjoint sensitivity analysis.
#[derive(Debug, Clone)]
pub struct SensitivityOptions {
    /// Number of checkpoints for forward pass (default: 10).
    ///
    /// More checkpoints = less recomputation but more memory.
    /// Fewer checkpoints = more recomputation but less memory.
    pub n_checkpoints: usize,

    /// Strategy for checkpoint placement (default: Uniform).
    pub checkpoint_strategy: CheckpointStrategy,

    /// Relative tolerance for adjoint integration (default: 1e-6).
    pub adjoint_rtol: f64,

    /// Absolute tolerance for adjoint integration (default: 1e-8).
    pub adjoint_atol: f64,

    /// Method for adjoint integration (default: RK45).
    ///
    /// For stiff ODEs, consider using BDF for the adjoint as well.
    pub adjoint_method: ODEMethod,

    /// Maximum steps for adjoint integration (default: 10000).
    pub adjoint_max_steps: usize,
}

impl Default for SensitivityOptions {
    fn default() -> Self {
        Self {
            n_checkpoints: 10,
            checkpoint_strategy: CheckpointStrategy::Uniform,
            adjoint_rtol: 1e-6,
            adjoint_atol: 1e-8,
            adjoint_method: ODEMethod::RK45,
            adjoint_max_steps: 10000,
        }
    }
}

impl SensitivityOptions {
    /// Set the number of checkpoints.
    pub fn with_checkpoints(mut self, n: usize) -> Self {
        self.n_checkpoints = n;
        self
    }

    /// Set the checkpoint strategy.
    pub fn with_strategy(mut self, strategy: CheckpointStrategy) -> Self {
        self.checkpoint_strategy = strategy;
        self
    }

    /// Set adjoint integration tolerances.
    pub fn with_adjoint_tolerances(mut self, rtol: f64, atol: f64) -> Self {
        self.adjoint_rtol = rtol;
        self.adjoint_atol = atol;
        self
    }

    /// Set adjoint integration method.
    pub fn with_adjoint_method(mut self, method: ODEMethod) -> Self {
        self.adjoint_method = method;
        self
    }
}

/// Result of adjoint sensitivity analysis.
#[derive(Debug, Clone)]
pub struct SensitivityResult<R: Runtime> {
    /// Gradient of the cost with respect to parameters: ∂J/∂p [n_params].
    pub gradient: Tensor<R>,

    /// Cost function value J = g(y(T)).
    pub cost: f64,

    /// Final state y(T).
    pub y_final: Tensor<R>,

    /// Number of function evaluations during forward pass.
    pub nfev_forward: usize,

    /// Number of function evaluations during adjoint pass.
    pub nfev_adjoint: usize,

    /// Number of checkpoints used.
    pub n_checkpoints: usize,
}

impl<R: Runtime> SensitivityResult<R> {
    /// Get the gradient as a Vec<f64>.
    pub fn gradient_vec(&self) -> Vec<f64> {
        self.gradient.to_vec()
    }

    /// Get the final state as a Vec<f64>.
    pub fn y_final_vec(&self) -> Vec<f64> {
        self.y_final.to_vec()
    }
}

/// A checkpoint storing state at a specific time for adjoint computation.
#[derive(Debug, Clone)]
pub struct Checkpoint<R: Runtime> {
    /// Time at this checkpoint.
    pub t: f64,

    /// State at this checkpoint.
    pub y: Tensor<R>,
}

impl<R: Runtime> Checkpoint<R> {
    /// Create a new checkpoint.
    pub fn new(t: f64, y: Tensor<R>) -> Self {
        Self { t, y }
    }
}
