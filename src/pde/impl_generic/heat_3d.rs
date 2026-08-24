//! 3D heat equation solver: du/dt = alpha * nabla^2 u + source.
use crate::DType;

use numr::error::Result;
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::sparse::{SparseOps, SparseTensor};
use numr::tensor::Tensor;

use crate::integrate::{IntegrationAlgorithms, ODEOptions, ODEResultTensor};
use crate::pde::error::{PdeError, PdeResult};
use crate::pde::types::{BoundarySpec, FdmOptions, Grid3D, TimeDependentOptions, TimeResult};

use super::ode_extract::extract_ode_solutions;
use super::stencil::assemble_laplacian_3d;

/// Solve the 3D heat equation using method of lines.
#[allow(clippy::too_many_arguments)]
pub fn heat_3d_impl<R, C>(
    client: &C,
    u0: &Tensor<R>,
    alpha: f64,
    source: Option<&Tensor<R>>,
    grid: &Grid3D,
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
    let nz = grid.nz;
    let n = nx * ny * nz;
    let device = client.device();

    if nx < 3 || ny < 3 || nz < 3 {
        return Err(PdeError::InvalidGrid {
            context: format!("Grid must be at least 3x3x3, got {}x{}x{}", nx, ny, nz),
        });
    }

    let laplacian = assemble_laplacian_3d::<R>(grid, device).map_err(PdeError::from)?;
    let sparse_l = SparseTensor::Csr(laplacian);

    let u0_flat = u0.reshape(&[n])?;

    let source_flat = match source {
        Some(s) => Some(s.reshape(&[n])?),
        None => None,
    };

    let rhs = |_t: &Tensor<R>, u: &Tensor<R>| -> Result<Tensor<R>> {
        let lu = sparse_l.spmv(u)?;
        let alpha_lu = client.mul_scalar(&lu, alpha)?;
        match &source_flat {
            Some(s) => client.add(&alpha_lu, s),
            None => Ok(alpha_lu),
        }
    };

    let ode_opts = ODEOptions {
        rtol: 1e-6,
        atol: 1e-9,
        max_steps: 50000,
        ..Default::default()
    };

    let ode_result: ODEResultTensor<R> = client
        .solve_ivp(rhs, time_opts.t_span, &u0_flat, &ode_opts)
        .map_err(PdeError::from)?;

    if !ode_result.success {
        return Err(PdeError::DidNotConverge {
            iterations: ode_result.nfev,
            tolerance: ode_opts.rtol,
            context: ode_result
                .message
                .unwrap_or_else(|| "3D heat equation failed".to_string()),
        });
    }

    Ok(extract_ode_solutions(
        ode_result.t,
        &ode_result.y,
        n,
        n,
        &[nx, ny, nz],
        device,
    )?)
}
