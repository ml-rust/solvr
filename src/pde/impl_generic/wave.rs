//! 2D wave equation solver: d^2u/dt^2 = c^2 * nabla^2 u + source.
//!
//! Converts to first-order system [u, v] where v = du/dt:
//!   du/dt = v
//!   dv/dt = c^2 * nabla^2 u + source
use crate::DType;

use numr::error::Result;
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::sparse::{SparseOps, SparseTensor};
use numr::tensor::Tensor;

use crate::integrate::{IntegrationAlgorithms, ODEOptions, ODEResultTensor};
use crate::pde::error::{PdeError, PdeResult};
use crate::pde::types::{BoundarySpec, FdmOptions, Grid2D, TimeDependentOptions, TimeResult};

use super::ode_extract::extract_ode_solutions;
use super::stencil::assemble_laplacian_2d;

/// Solve the 2D wave equation by converting to first-order system.
#[allow(clippy::too_many_arguments)]
pub fn wave_impl<R, C>(
    client: &C,
    u0: &Tensor<R>,
    v0: &Tensor<R>,
    c: f64,
    source: Option<&Tensor<R>>,
    grid: &Grid2D,
    _boundary: &[BoundarySpec<R>],
    time_opts: &TimeDependentOptions,
    _options: &FdmOptions,
) -> PdeResult<TimeResult<R>>
where
    R: Runtime<DType = DType>,
    R::Client: SparseOps<R>,
    C: TensorOps<R> + ScalarOps<R> + IntegrationAlgorithms<R> + RuntimeClient<R>,
{
    let nx = grid.nx;
    let ny = grid.ny;
    let n = nx * ny;
    let device = client.device();

    if nx < 3 || ny < 3 {
        return Err(PdeError::InvalidGrid {
            context: format!("Grid must be at least 3x3, got {}x{}", nx, ny),
        });
    }

    let laplacian = assemble_laplacian_2d::<R>(grid, device).map_err(PdeError::from)?;
    let sparse_l = SparseTensor::Csr(laplacian);

    let u0_flat = u0.reshape(&[n])?;
    let v0_flat = v0.reshape(&[n])?;

    // Stack initial state: [u; v] of length 2*n
    let state0 = client.cat(&[&u0_flat, &v0_flat], 0)?;

    let c2 = c * c;

    let source_flat = match source {
        Some(s) => Some(s.reshape(&[n])?),
        None => None,
    };

    // RHS: f(t, [u;v]) = [v; c^2 * L*u + source]
    let rhs = |_t: &Tensor<R>, state: &Tensor<R>| -> Result<Tensor<R>> {
        let u_part = state.narrow(0, 0, n)?;
        let v_part = state.narrow(0, n, n)?;

        let lu = sparse_l.spmv(&u_part)?;
        let c2_lu = client.mul_scalar(&lu, c2)?;
        let dvdt = match &source_flat {
            Some(s) => client.add(&c2_lu, s)?,
            None => c2_lu,
        };

        client.cat(&[&v_part, &dvdt], 0)
    };

    let ode_opts = ODEOptions {
        rtol: 1e-8,
        atol: 1e-10,
        max_steps: 100000,
        ..Default::default()
    };

    let ode_result: ODEResultTensor<R> = client
        .solve_ivp(rhs, time_opts.t_span, &state0, &ode_opts)
        .map_err(PdeError::from)?;

    if !ode_result.success {
        return Err(PdeError::DidNotConverge {
            iterations: ode_result.nfev,
            tolerance: ode_opts.rtol,
            context: ode_result
                .message
                .unwrap_or_else(|| "Wave equation failed".to_string()),
        });
    }

    // Extract u (first n components) at each timestep; stride is 2*n
    Ok(extract_ode_solutions(
        ode_result.t,
        &ode_result.y,
        n,
        2 * n,
        &[nx, ny],
        device,
    )?)
}
