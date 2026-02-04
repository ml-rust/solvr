//! Generic Voronoi diagram implementation.
//!
//! Computes Voronoi as dual of Delaunay triangulation.

use crate::spatial::impl_generic::delaunay::delaunay_impl;
use crate::spatial::traits::delaunay::Delaunay;
use crate::spatial::traits::voronoi::Voronoi;
use crate::spatial::{validate_points_2d, validate_points_dtype};
use numr::error::{Error, Result};
use numr::ops::{DistanceOps, IndexingOps, ReduceOps, ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use std::collections::HashSet;

/// Compute Voronoi diagram.
pub fn voronoi_impl<R, C>(client: &C, points: &Tensor<R>) -> Result<Voronoi<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + ReduceOps<R> + RuntimeClient<R>,
{
    validate_points_dtype(points.dtype(), "voronoi")?;
    validate_points_2d(points.shape(), "voronoi")?;

    let tri = delaunay_impl(client, points)?;
    voronoi_from_delaunay_impl(client, &tri)
}

/// Compute Voronoi diagram from existing Delaunay triangulation.
pub fn voronoi_from_delaunay_impl<R, C>(_client: &C, tri: &Delaunay<R>) -> Result<Voronoi<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
    let d = tri.points.shape()[1];
    if d != 2 {
        return Err(Error::InvalidArgument {
            arg: "tri",
            reason: format!("Voronoi currently only supports 2D, got {}D", d),
        });
    }

    let points_data: Vec<f64> = tri.points.to_vec();
    let simplices_data: Vec<i64> = tri.simplices.to_vec();
    let neighbors_data: Vec<i64> = tri.neighbors.to_vec();
    let n = tri.points.shape()[0];
    let n_simplices = tri.simplices.shape()[0];
    let device = tri.points.device();

    // Compute circumcenters (Voronoi vertices)
    let mut vertices: Vec<f64> = Vec::with_capacity(n_simplices * 2);
    for s in 0..n_simplices {
        let v0 = simplices_data[s * 3] as usize;
        let v1 = simplices_data[s * 3 + 1] as usize;
        let v2 = simplices_data[s * 3 + 2] as usize;

        let (cx, cy) = circumcenter(&points_data, v0, v1, v2);
        vertices.push(cx);
        vertices.push(cy);
    }

    // Build ridges (Voronoi edges) and their generator points
    let mut ridge_vertices_data: Vec<i64> = Vec::new();
    let mut ridge_points_data: Vec<i64> = Vec::new();
    let mut seen_ridges: HashSet<(usize, usize)> = HashSet::new();

    for s in 0..n_simplices {
        for k in 0..3 {
            let neighbor = neighbors_data[s * 3 + k];
            if neighbor != -1 {
                let ns = neighbor as usize;
                if ns > s {
                    // Get the two points sharing this edge
                    let e_v1 = simplices_data[s * 3 + k] as usize;
                    let e_v2 = simplices_data[s * 3 + ((k + 1) % 3)] as usize;

                    let ridge = if e_v1 < e_v2 {
                        (e_v1, e_v2)
                    } else {
                        (e_v2, e_v1)
                    };

                    if !seen_ridges.contains(&ridge) {
                        seen_ridges.insert(ridge);
                        ridge_vertices_data.push(s as i64);
                        ridge_vertices_data.push(ns as i64);
                        ridge_points_data.push(ridge.0 as i64);
                        ridge_points_data.push(ridge.1 as i64);
                    }
                }
            }
        }
    }

    let n_ridges = ridge_points_data.len() / 2;

    // Build regions for each generator point
    let mut point_ridges: Vec<Vec<usize>> = vec![Vec::new(); n];
    for r in 0..n_ridges {
        let p1 = ridge_points_data[r * 2] as usize;
        let p2 = ridge_points_data[r * 2 + 1] as usize;
        point_ridges[p1].push(r);
        point_ridges[p2].push(r);
    }

    // Build regions CSR
    let mut regions_indices: Vec<i64> = Vec::new();
    let mut regions_indptr: Vec<i64> = vec![0];

    for ridges in point_ridges.iter().take(n) {
        for &r in ridges {
            regions_indices.push(r as i64);
        }
        regions_indptr.push(regions_indices.len() as i64);
    }

    // Find points with unbounded regions (on convex hull)
    let hull_data: Vec<i64> = tri.convex_hull.to_vec();
    let hull_set: HashSet<i64> = hull_data.iter().copied().collect();
    let point_region: Vec<i64> = (0..n as i64).filter(|i| hull_set.contains(i)).collect();

    // Handle empty arrays by using dummy data when necessary
    // Tensor::from_slice requires non-empty slices to match shape
    let ridge_vertices_final = if ridge_vertices_data.is_empty() {
        vec![-1i64, -1] // Sentinel for "no ridges"
    } else {
        ridge_vertices_data
    };

    let ridge_points_final = if ridge_points_data.is_empty() {
        vec![-1i64, -1] // Sentinel for "no ridges"
    } else {
        ridge_points_data
    };

    let regions_indices_final = if regions_indices.is_empty() {
        vec![-1i64] // Sentinel for "no regions"
    } else {
        regions_indices
    };

    let point_region_final = if point_region.is_empty() {
        vec![-1i64] // Sentinel
    } else {
        point_region
    };

    Ok(Voronoi {
        points: tri.points.clone(),
        vertices: Tensor::<R>::from_slice(&vertices, &[n_simplices, 2], device),
        ridge_vertices: Tensor::<R>::from_slice(
            &ridge_vertices_final,
            &[n_ridges.max(1), 2],
            device,
        ),
        ridge_points: Tensor::<R>::from_slice(&ridge_points_final, &[n_ridges.max(1), 2], device),
        regions_indices: Tensor::<R>::from_slice(
            &regions_indices_final,
            &[regions_indices_final.len()],
            device,
        ),
        regions_indptr: Tensor::<R>::from_slice(&regions_indptr, &[regions_indptr.len()], device),
        point_region: Tensor::<R>::from_slice(
            &point_region_final,
            &[point_region_final.len()],
            device,
        ),
    })
}

fn circumcenter(points: &[f64], v0: usize, v1: usize, v2: usize) -> (f64, f64) {
    let ax = points[v0 * 2];
    let ay = points[v0 * 2 + 1];
    let bx = points[v1 * 2];
    let by = points[v1 * 2 + 1];
    let cx = points[v2 * 2];
    let cy = points[v2 * 2 + 1];

    let d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
    if d.abs() < 1e-10 {
        // Degenerate - return centroid
        return ((ax + bx + cx) / 3.0, (ay + by + cy) / 3.0);
    }

    let ux = ((ax * ax + ay * ay) * (by - cy)
        + (bx * bx + by * by) * (cy - ay)
        + (cx * cx + cy * cy) * (ay - by))
        / d;
    let uy = ((ax * ax + ay * ay) * (cx - bx)
        + (bx * bx + by * by) * (ax - cx)
        + (cx * cx + cy * cy) * (bx - ax))
        / d;

    (ux, uy)
}

/// Find which Voronoi region contains each query point.
///
/// Uses tensor operations - data stays on device.
/// The region containing a point is the generator closest to it.
pub fn voronoi_find_region_impl<R, C>(
    client: &C,
    vor: &Voronoi<R>,
    query: &Tensor<R>,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + DistanceOps<R> + IndexingOps<R> + RuntimeClient<R>,
{
    // Compute distances from each query point to all generators
    let distances = client.cdist(query, &vor.points, numr::ops::DistanceMetric::Euclidean)?;

    // Find the nearest generator for each query (argmin along dim 1)
    let nearest = client.argmin(&distances, 1, false)?;

    Ok(nearest)
}
