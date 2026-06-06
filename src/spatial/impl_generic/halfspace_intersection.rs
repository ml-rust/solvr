//! Generic halfspace intersection implementation.
//!
//! Algorithm: dual transform → convex hull → solve linear systems for vertices.

use crate::spatial::impl_generic::convex_hull::convex_hull_impl;
use crate::spatial::traits::halfspace_intersection::HalfspaceIntersection;
use crate::spatial::validate_points_dtype;
use numr::dtype::DType;
use numr::error::{Error, Result};
use numr::ops::{
    CompareOps, IndexingOps, LinalgOps, ReduceOps, ScalarOps, SortingOps, TensorOps,
    TypeConversionOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use std::collections::HashSet;

/// Compute halfspace intersection via dual convex hull.
pub fn halfspace_intersection_impl<R, C>(
    client: &C,
    halfspaces: &Tensor<R>,
    interior_point: &Tensor<R>,
) -> Result<HalfspaceIntersection<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R>
        + ScalarOps<R>
        + ReduceOps<R>
        + CompareOps<R>
        + IndexingOps<R>
        + SortingOps<R>
        + TypeConversionOps<R>
        + LinalgOps<R>
        + RuntimeClient<R>,
{
    validate_points_dtype(halfspaces.dtype(), "halfspace_intersection")?;

    let shape = halfspaces.shape();
    if shape.len() != 2 {
        return Err(Error::InvalidArgument {
            arg: "halfspaces",
            reason: "Expected 2D tensor [m, d+1]".to_string(),
        });
    }
    let m = shape[0];
    let d_plus_1 = shape[1];
    if d_plus_1 < 2 {
        return Err(Error::InvalidArgument {
            arg: "halfspaces",
            reason: "Need at least 2 columns (d >= 1)".to_string(),
        });
    }
    let d = d_plus_1 - 1;

    let ip_shape = interior_point.shape();
    if ip_shape.len() != 1 || ip_shape[0] != d {
        return Err(Error::InvalidArgument {
            arg: "interior_point",
            reason: format!("Expected shape [{}], got {:?}", d, ip_shape),
        });
    }

    let device = halfspaces.device();
    let dtype = halfspaces.dtype();

    // Extract normals [m, d] and offsets [m] from halfspaces [m, d+1]
    let normals = halfspaces.narrow(1, 0, d)?.contiguous()?; // [m, d]
    let offsets = halfspaces.narrow(1, d, 1)?.contiguous()?.reshape(&[m])?; // [m]

    // Verify interior_point is strictly inside all halfspaces: n·x + b < 0
    // vals = normals @ ip + offsets  (all on device)
    let ip_col = interior_point.reshape(&[d, 1])?; // [d, 1]
    let n_dot_ip = client.matmul(&normals, &ip_col)?.reshape(&[m])?; // [m]
    let vals = client.add(&n_dot_ip, &offsets)?; // [m]

    // Check all vals < 0 (single scalar transfer for error reporting)
    let max_val: f64 = client.max(&vals, &[0], false)?.item::<f64>()?;
    if max_val >= 0.0 {
        // Find which halfspace is violated for error message
        let zero = Tensor::<R>::zeros(&[], dtype, device);
        let violated_raw = client.ge(&vals, &zero)?;
        let violated = client.cast(&violated_raw, DType::U8)?;
        let violated_data: Vec<u8> = violated.to_vec();
        let idx = violated_data.iter().position(|&v| v > 0).unwrap_or(0);
        let val_at_idx: f64 = vals.narrow(0, idx, 1)?.item::<f64>()?;
        return Err(Error::InvalidArgument {
            arg: "interior_point",
            reason: format!(
                "Interior point violates halfspace {} (n·x + b = {:.6} >= 0)",
                idx, val_at_idx
            ),
        });
    }

    // Shifted offsets: shifted_b = normals @ ip + offsets (= vals, already computed)
    // Dual transform: dual_point_i = -normal_i / shifted_b_i
    let shifted_b = vals; // [m], all negative
    let shifted_b_col = shifted_b.reshape(&[m, 1])?; // [m, 1]
    let shifted_b_broadcast = shifted_b_col.broadcast_to(&[m, d])?.contiguous()?; // [m, d]
    let neg_normals = client.mul_scalar(&normals, -1.0)?; // [m, d]
    let dual_points = client.div(&neg_normals, &shifted_b_broadcast)?; // [m, d]

    // Compute convex hull of dual points
    let hull = convex_hull_impl(client, &dual_points)?;

    // Each hull facet corresponds to a primal vertex.
    // For a d-dimensional hull, each facet has d vertex indices.
    // The primal vertex is the intersection of the d corresponding halfplanes.
    // This topological traversal + small d×d solves requires CPU-side logic
    // (same pattern as voronoi_from_delaunay_impl).
    let simplices_data: Vec<i64> = hull.simplices.to_vec();
    let hs_data: Vec<f64> = halfspaces.to_vec();
    let n_simplices = hull.simplices.shape()[0];
    let simplex_dim = hull.simplices.shape()[1];

    // Collect unique primal vertices by solving linear systems
    let mut primal_vertices: Vec<f64> = Vec::new();
    let mut seen: HashSet<Vec<i64>> = HashSet::new();

    for s in 0..n_simplices {
        // Get the d halfspace indices for this facet
        let mut hs_indices: Vec<i64> = (0..simplex_dim)
            .map(|k| simplices_data[s * simplex_dim + k])
            .collect();
        hs_indices.sort();

        if seen.contains(&hs_indices) {
            continue;
        }
        seen.insert(hs_indices.clone());

        // Build d×d system: for each halfspace i in this facet,
        // n_i · x + b_i = 0 (active constraint)
        let mut a_data: Vec<f64> = Vec::with_capacity(d * d);
        let mut b_data: Vec<f64> = Vec::with_capacity(d);

        for &hi in &hs_indices {
            let hi = hi as usize;
            for j in 0..d {
                a_data.push(hs_data[hi * d_plus_1 + j]);
            }
            b_data.push(-hs_data[hi * d_plus_1 + d]);
        }

        let a_tensor = Tensor::<R>::from_slice(&a_data, &[d, d], device);
        let b_tensor = Tensor::<R>::from_slice(&b_data, &[d, 1], device);

        match LinalgOps::solve(client, &a_tensor, &b_tensor) {
            Ok(x) => {
                let x_data: Vec<f64> = x.to_vec::<f64>();
                primal_vertices.extend_from_slice(&x_data[..d]);
            }
            Err(_) => {
                // Singular system, skip this facet
                continue;
            }
        }
    }

    let n_vertices = primal_vertices.len() / d;

    if n_vertices == 0 {
        return Err(Error::InvalidArgument {
            arg: "halfspaces",
            reason: "No valid intersection vertices found".to_string(),
        });
    }

    let intersections = Tensor::<R>::from_slice(&primal_vertices, &[n_vertices, d], device);

    Ok(HalfspaceIntersection {
        halfspaces: halfspaces.clone(),
        intersections,
        dual_points,
        interior_point: interior_point.clone(),
    })
}
