//! Mean Shift clustering trait.

use numr::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Options for Mean Shift.
#[derive(Debug, Clone)]
pub struct MeanShiftOptions {
    /// Kernel bandwidth. None = auto-estimate.
    pub bandwidth: Option<f64>,
    /// Maximum iterations.
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tol: f64,
    /// Use bin seeding for initial seeds.
    pub bin_seeding: bool,
}

impl Default for MeanShiftOptions {
    fn default() -> Self {
        Self {
            bandwidth: None,
            max_iter: 300,
            tol: 1e-3,
            bin_seeding: false,
        }
    }
}

/// Result of Mean Shift clustering.
#[derive(Debug, Clone)]
pub struct MeanShiftResult<R: Runtime> {
    /// Cluster labels [n] I64.
    pub labels: Tensor<R>,
    /// Cluster centers [k, d].
    pub cluster_centers: Tensor<R>,
    /// Number of iterations run.
    pub n_iter: usize,
}

/// Mean Shift clustering algorithms.
pub trait MeanShiftAlgorithms<R: Runtime> {
    /// Run Mean Shift on data [n, d].
    fn mean_shift(
        &self,
        data: &Tensor<R>,
        options: &MeanShiftOptions,
    ) -> Result<MeanShiftResult<R>>;
}
