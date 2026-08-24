//! 1D Chebyshev spectral method: solve -u'' + q(x)*u = f(x) on [-1,1].
//!
//! Builds Chebyshev differentiation matrix D, solves (D^2 - diag(q))u = f
//! as a dense linear system.
use crate::DType;

use numr::ops::{LinalgOps, ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use crate::pde::error::{PdeError, PdeResult};
use crate::pde::types::SpectralResult;

use super::boundary::extract_dirichlet_1d_bcs;
use super::chebyshev::{chebyshev_diff_matrix, chebyshev_nodes};

/// Solve -u'' + q(x)*u = f(x) on [-1,1] with Chebyshev collocation.
pub fn spectral_1d_impl<R, C>(
    client: &C,
    f_rhs: &Tensor<R>,
    q: Option<&Tensor<R>>,
    n: usize,
    boundary: &[crate::pde::types::BoundarySpec<R>],
) -> PdeResult<SpectralResult<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + LinalgOps<R> + RuntimeClient<R>,
{
    if n < 2 {
        return Err(PdeError::InvalidParameter {
            parameter: "n".to_string(),
            message: "Need at least 2 Chebyshev points".to_string(),
        });
    }

    let device = client.device();
    let np1 = n + 1; // N+1 points

    // Chebyshev nodes
    let x_data = chebyshev_nodes(n);

    // Build Chebyshev differentiation matrix D [np1 x np1]
    let d_data = chebyshev_diff_matrix(n);

    // D^2
    let d_tensor = Tensor::<R>::from_slice(&d_data, &[np1, np1], device)?;
    let d2 = client
        .matmul(&d_tensor, &d_tensor)
        .map_err(PdeError::from)?;

    // System matrix: A = -D^2 + diag(q)
    let neg_d2 = client.mul_scalar(&d2, -1.0).map_err(PdeError::from)?;

    let system_matrix = if let Some(q_tensor) = q {
        let q_data: Vec<f64> = q_tensor.to_vec();
        let mut diag_q = vec![0.0; np1 * np1];
        for i in 0..np1.min(q_data.len()) {
            diag_q[i * np1 + i] = q_data[i];
        }
        let diag_q_tensor = Tensor::<R>::from_slice(&diag_q, &[np1, np1], device)?;
        client
            .add(&neg_d2, &diag_q_tensor)
            .map_err(PdeError::from)?
    } else {
        neg_d2
    };

    // RHS vector
    let f_data: Vec<f64> = f_rhs.to_vec();
    let mut rhs = vec![0.0; np1];
    let copy_len = np1.min(f_data.len());
    rhs[..copy_len].copy_from_slice(&f_data[..copy_len]);

    // Apply boundary conditions
    let (bc_left, bc_right) = extract_dirichlet_1d_bcs(boundary, "spectral 1D")?;

    // Replace first and last rows of system matrix with identity rows
    let mut sys_data: Vec<f64> = system_matrix.to_vec();

    // Row 0 (x=1, left in Chebyshev ordering): identity row
    for (j, val) in sys_data.iter_mut().enumerate().take(np1) {
        *val = if j == 0 { 1.0 } else { 0.0 };
    }
    rhs[0] = bc_left;

    // Row N (x=-1, right in Chebyshev ordering): identity row
    for j in 0..np1 {
        sys_data[n * np1 + j] = if j == n { 1.0 } else { 0.0 };
    }
    rhs[n] = bc_right;

    // Solve dense system
    let a_tensor = Tensor::<R>::from_slice(&sys_data, &[np1, np1], device)?;
    let b_tensor = Tensor::<R>::from_slice(&rhs, &[np1, 1], device)?;

    let solution = client.solve(&a_tensor, &b_tensor).map_err(PdeError::from)?;
    let solution_flat = solution.reshape(&[np1])?;

    let nodes = Tensor::<R>::from_slice(&x_data, &[np1], device)?;

    Ok(SpectralResult {
        solution: solution_flat,
        nodes,
    })
}
