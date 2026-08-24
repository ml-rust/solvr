//! 1D finite element method: -u'' = f(x) on `[a,b]` with Dirichlet BCs.
//!
//! Uses linear hat basis functions on an arbitrary node set.
//! Assembles tridiagonal stiffness matrix, solves with dense solver.
use crate::DType;

use numr::ops::{LinalgOps, ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use crate::pde::error::{PdeError, PdeResult};
use crate::pde::types::{BoundarySpec, FdmOptions, FemResult};

use super::boundary::extract_dirichlet_1d_bcs;

/// Solve 1D BVP: -u'' = f(x) using linear FEM.
///
/// The node set x_nodes defines the element boundaries.
/// f_rhs contains f(x) evaluated at each node.
pub fn fem_1d_impl<R, C>(
    client: &C,
    f_rhs: &Tensor<R>,
    x_nodes: &Tensor<R>,
    boundary: &[BoundarySpec<R>],
    _options: &FdmOptions,
) -> PdeResult<FemResult<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + LinalgOps<R> + RuntimeClient<R>,
{
    let n = x_nodes.shape()[0];
    let device = client.device();

    if n < 3 {
        return Err(PdeError::InvalidGrid {
            context: format!("Need at least 3 nodes for 1D FEM, got {}", n),
        });
    }

    let x_data: Vec<f64> = x_nodes.to_vec();
    let f_data: Vec<f64> = f_rhs.to_vec();

    // Extract boundary values
    let (bc_left, bc_right) = extract_dirichlet_1d_bcs(boundary, "1D FEM")?;

    // Assemble tridiagonal stiffness matrix K and load vector F
    // For linear elements on [x_i, x_{i+1}] with length h_i:
    //   K_local = (1/h_i) * [[1, -1], [-1, 1]]
    //   F_local = (h_i/2) * [f_i, f_{i+1}]  (lumped load)
    let n_interior = n - 2;

    // Build dense matrix for the interior system (small enough for direct solve)
    let mut k_data = vec![0.0; n_interior * n_interior];
    let mut f_vec = vec![0.0; n_interior];

    for e in 0..(n - 1) {
        let h = x_data[e + 1] - x_data[e];
        if h <= 0.0 {
            return Err(PdeError::InvalidGrid {
                context: format!("Non-positive element length at element {}", e),
            });
        }

        let ke = 1.0 / h;

        // Map local nodes to global interior indices (shifted by -1 for BC elimination)
        let global = [e as i64 - 1, e as i64]; // interior index = global - 1

        // Load: lumped quadrature
        let fe = [h * f_data[e] / 2.0, h * f_data[e + 1] / 2.0];

        let local_k = [[ke, -ke], [-ke, ke]];

        for (li, &gi) in global.iter().enumerate() {
            if gi < 0 || gi >= n_interior as i64 {
                // Boundary node: move to RHS
                if gi < 0 {
                    // Left boundary node (global 0)
                    let other = global[1 - li];
                    if other >= 0 && (other as usize) < n_interior {
                        f_vec[other as usize] -= local_k[1 - li][li] * bc_left;
                        f_vec[other as usize] += fe[1 - li];
                    }
                } else {
                    // Right boundary node (global n-1)
                    let other = global[1 - li];
                    if other >= 0 && (other as usize) < n_interior {
                        f_vec[other as usize] -= local_k[1 - li][li] * bc_right;
                        f_vec[other as usize] += fe[1 - li];
                    }
                }
                continue;
            }

            let gi_usize = gi as usize;
            f_vec[gi_usize] += fe[li];

            for (lj, &gj) in global.iter().enumerate() {
                if gj >= 0 && (gj as usize) < n_interior {
                    k_data[gi_usize * n_interior + gj as usize] += local_k[li][lj];
                }
            }
        }
    }

    // Solve K * u_interior = F
    let k_tensor = Tensor::<R>::from_slice(&k_data, &[n_interior, n_interior], device)?;
    let f_tensor = Tensor::<R>::from_slice(&f_vec, &[n_interior, 1], device)?;

    let u_interior = client.solve(&k_tensor, &f_tensor).map_err(PdeError::from)?;
    let u_int_data: Vec<f64> = u_interior.to_vec();

    // Assemble full solution with boundary values
    let mut full_solution = vec![0.0; n];
    full_solution[0] = bc_left;
    full_solution[1..(n_interior + 1)].copy_from_slice(&u_int_data[..n_interior]);
    full_solution[n - 1] = bc_right;

    let solution = Tensor::<R>::from_slice(&full_solution, &[n], device)?;

    Ok(FemResult {
        solution,
        nodes: x_nodes.clone(),
        iterations: 1, // Direct solve
        residual_norm: 0.0,
    })
}
