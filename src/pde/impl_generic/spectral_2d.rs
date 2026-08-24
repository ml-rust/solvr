//! 2D Chebyshev spectral method: solve -nabla^2 u = f on [-1,1]^2.
//!
//! Uses tensor product of 1D Chebyshev differentiation matrices:
//! D_xx = D^2 kron I, D_yy = I kron D^2.
use crate::DType;

use numr::ops::{LinalgOps, ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use crate::pde::error::{PdeError, PdeResult};
use crate::pde::types::{BoundaryCondition, BoundarySpec, SpectralResult};

use super::chebyshev::{chebyshev_diff_matrix, chebyshev_nodes};

/// Solve -nabla^2 u = f on [-1,1]^2 with Chebyshev collocation.
pub fn spectral_2d_impl<R, C>(
    client: &C,
    f_rhs: &Tensor<R>,
    nx: usize,
    ny: usize,
    boundary: &[BoundarySpec<R>],
) -> PdeResult<SpectralResult<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + LinalgOps<R> + RuntimeClient<R>,
{
    if nx < 2 || ny < 2 {
        return Err(PdeError::InvalidParameter {
            parameter: "nx/ny".to_string(),
            message: "Need at least 2 points in each direction".to_string(),
        });
    }

    let device = client.device();
    let npx = nx + 1;
    let npy = ny + 1;
    let total = npx * npy;

    // Build 1D Chebyshev nodes
    let x_nodes = chebyshev_nodes(nx);
    let y_nodes = chebyshev_nodes(ny);

    // Build 1D differentiation matrices
    let dx_data = chebyshev_diff_matrix(nx);
    let dy_data = chebyshev_diff_matrix(ny);

    let dx = Tensor::<R>::from_slice(&dx_data, &[npx, npx], device)?;
    let dy = Tensor::<R>::from_slice(&dy_data, &[npy, npy], device)?;

    // D^2 in each direction
    let dx2 = client.matmul(&dx, &dx).map_err(PdeError::from)?;
    let dy2 = client.matmul(&dy, &dy).map_err(PdeError::from)?;

    // Identity matrices
    let ix = client
        .eye(npx, None, numr::dtype::DType::F64)
        .map_err(PdeError::from)?;
    let iy = client
        .eye(npy, None, numr::dtype::DType::F64)
        .map_err(PdeError::from)?;

    // 2D Laplacian: D_xx = D²_x ⊗ I_y, D_yy = I_x ⊗ D²_y
    let d_xx = client.kron(&dx2, &iy).map_err(PdeError::from)?;
    let d_yy = client.kron(&ix, &dy2).map_err(PdeError::from)?;

    // -Laplacian = -(D_xx + D_yy)
    let laplacian = client.add(&d_xx, &d_yy).map_err(PdeError::from)?;
    let neg_laplacian = client
        .mul_scalar(&laplacian, -1.0)
        .map_err(PdeError::from)?;

    // Get mutable system data
    let mut sys_data: Vec<f64> = neg_laplacian.to_vec();
    let f_data: Vec<f64> = f_rhs.to_vec();
    let mut rhs = vec![0.0; total];
    rhs[..total.min(f_data.len())].copy_from_slice(&f_data[..total.min(f_data.len())]);

    // Identify boundary nodes and apply Dirichlet BCs
    let bc_val = extract_spectral_2d_bc_value(boundary)?;

    for i in 0..npx {
        for j in 0..npy {
            let idx = i * npy + j;
            let is_boundary = i == 0 || i == nx || j == 0 || j == ny;

            if is_boundary {
                // Replace row with identity
                for k in 0..total {
                    sys_data[idx * total + k] = if k == idx { 1.0 } else { 0.0 };
                }
                rhs[idx] = bc_val;
            }
        }
    }

    // Solve dense system
    let a_tensor = Tensor::<R>::from_slice(&sys_data, &[total, total], device)?;
    let b_tensor = Tensor::<R>::from_slice(&rhs, &[total, 1], device)?;

    let solution = client.solve(&a_tensor, &b_tensor).map_err(PdeError::from)?;
    let solution_2d = solution.reshape(&[npx, npy])?;

    // Build 2D node coordinates: shape [total, 2]
    let mut coords = Vec::with_capacity(total * 2);
    for &xi in &x_nodes {
        for &yj in &y_nodes {
            coords.push(xi);
            coords.push(yj);
        }
    }
    let nodes = Tensor::<R>::from_slice(&coords, &[total, 2], device)?;

    Ok(SpectralResult {
        solution: solution_2d,
        nodes,
    })
}

fn extract_spectral_2d_bc_value<R: Runtime<DType = DType>>(
    boundary: &[BoundarySpec<R>],
) -> PdeResult<f64> {
    if let Some(spec) = boundary.first() {
        match &spec.condition {
            BoundaryCondition::Dirichlet(vals) => {
                let v: Vec<f64> = vals.to_vec();
                Ok(if v.is_empty() { 0.0 } else { v[0] })
            }
            _ => Err(PdeError::InvalidBoundary {
                context: "Only Dirichlet BCs supported for spectral 2D".to_string(),
            }),
        }
    } else {
        Ok(0.0) // Default: homogeneous Dirichlet
    }
}
