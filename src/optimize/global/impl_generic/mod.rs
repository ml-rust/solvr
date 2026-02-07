//! Generic implementations of global optimization algorithms.
//!
//! All algorithms use tensor operations and are generic over `R: Runtime`.

pub mod basinhopping;
pub mod differential_evolution;
pub mod dual_annealing;
pub mod shgo;
pub mod simulated_annealing;

pub use basinhopping::basinhopping_impl;
pub use differential_evolution::differential_evolution_impl;
pub use dual_annealing::dual_annealing_impl;
pub use shgo::shgo_impl;
pub use simulated_annealing::simulated_annealing_impl;

use numr::ops::{CompareOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use crate::optimize::error::{OptimizeError, OptimizeResult};

/// Validate that lower < upper for all dimensions using tensor ops.
pub(crate) fn validate_bounds<R, C>(
    client: &C,
    lower: &Tensor<R>,
    upper: &Tensor<R>,
) -> OptimizeResult<()>
where
    R: Runtime,
    C: TensorOps<R> + CompareOps<R> + RuntimeClient<R>,
{
    let violations = client
        .ge(lower, upper)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("bounds check - {}", e),
        })?;

    let violation_sum =
        client
            .sum(&violations, &[0], false)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("sum violations - {}", e),
            })?;

    let sum_val: Vec<f64> = violation_sum.to_vec();
    if sum_val[0] > 0.0 {
        return Err(OptimizeError::InvalidInput {
            context: "lower bounds must be less than upper bounds".to_string(),
        });
    }

    Ok(())
}

/// Clamp tensor to bounds: max(lower, min(upper, x))
pub(crate) fn clamp_to_bounds<R, C>(
    client: &C,
    x: &Tensor<R>,
    lower: &Tensor<R>,
    upper: &Tensor<R>,
) -> OptimizeResult<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R>,
{
    let clamped_upper = client
        .minimum(x, upper)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("clamp: min with upper - {}", e),
        })?;
    client
        .maximum(&clamped_upper, lower)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("clamp: max with lower - {}", e),
        })
}
