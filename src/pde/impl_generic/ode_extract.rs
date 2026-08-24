//! Shared ODE result extraction for time-dependent PDE solvers.
use crate::DType;

use numr::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use crate::pde::types::TimeResult;

/// Extract per-timestep solutions from a stacked ODE result tensor.
///
/// The ODE solver returns `y` with shape `[n_steps * stride]` where each timestep
/// occupies `stride` elements. The first `n_vars` elements of each stride are
/// extracted and reshaped to `sol_shape`.
///
/// - `stride`: total variables per timestep (may differ from `n_vars` for wave equation)
/// - `n_vars`: number of variables to extract per step
/// - `sol_shape`: shape to reshape each extracted solution into
pub fn extract_ode_solutions<R: Runtime<DType = DType>>(
    ode_t: Tensor<R>,
    ode_y: &Tensor<R>,
    n_vars: usize,
    stride: usize,
    sol_shape: &[usize],
    device: &R::Device,
) -> Result<TimeResult<R>> {
    let y_data: Vec<f64> = ode_y.to_vec();
    let t_data: Vec<f64> = ode_t.to_vec();
    let n_steps = t_data.len();

    let mut solutions = Vec::with_capacity(n_steps);
    for step in 0..n_steps {
        let start = step * stride;
        let end = start + n_vars;
        if end <= y_data.len() {
            let sol = Tensor::<R>::from_slice(&y_data[start..end], sol_shape, device)?;
            solutions.push(sol);
        }
    }

    Ok(TimeResult {
        times: ode_t,
        solutions,
    })
}
