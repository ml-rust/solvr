//! Generic Delaunay triangulation implementation.
//!
//! Bowyer-Watson incremental algorithm.

use crate::spatial::traits::delaunay::Delaunay;
use crate::spatial::{validate_points_2d, validate_points_dtype};
use numr::dtype::DType;
use numr::error::{Error, Result};
use numr::ops::{
    CompareOps, IndexingOps, LogicalOps, ReduceOps, ScalarOps, TensorOps, TypeConversionOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use std::collections::{HashMap, HashSet};

/// Compute Delaunay triangulation using Bowyer-Watson algorithm.
pub fn delaunay_impl<R, C>(_client: &C, points: &Tensor<R>) -> Result<Delaunay<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + ReduceOps<R> + RuntimeClient<R>,
{
    validate_points_dtype(points.dtype(), "delaunay")?;
    validate_points_2d(points.shape(), "delaunay")?;

    let n = points.shape()[0];
    let d = points.shape()[1];

    if d != 2 {
        return Err(Error::InvalidArgument {
            arg: "points",
            reason: format!("Delaunay currently only supports 2D, got {}D", d),
        });
    }

    if n < 3 {
        return Err(Error::InvalidArgument {
            arg: "points",
            reason: "Need at least 3 points for 2D Delaunay".to_string(),
        });
    }

    let device = points.device();
    let points_data: Vec<f64> = points.to_vec();

    // Find bounding box
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for i in 0..n {
        let x = points_data[i * 2];
        let y = points_data[i * 2 + 1];
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }

    let dx = max_x - min_x;
    let dy = max_y - min_y;
    let delta = dx.max(dy) * 10.0;

    // Create super-triangle vertices (virtual points at indices n, n+1, n+2)
    let super_p0 = (min_x - delta, min_y - delta);
    let super_p1 = (min_x + dx / 2.0, max_y + delta);
    let super_p2 = (max_x + delta, min_y - delta);

    // Extended points including super-triangle
    let mut extended_points = points_data.clone();
    extended_points.push(super_p0.0);
    extended_points.push(super_p0.1);
    extended_points.push(super_p1.0);
    extended_points.push(super_p1.1);
    extended_points.push(super_p2.0);
    extended_points.push(super_p2.1);

    // Start with super-triangle
    let mut triangles: Vec<[usize; 3]> = vec![[n, n + 1, n + 2]];

    // Insert each point
    for i in 0..n {
        let px = extended_points[i * 2];
        let py = extended_points[i * 2 + 1];

        // Find triangles whose circumcircle contains this point
        let mut bad_triangles: Vec<usize> = Vec::new();
        for (ti, tri) in triangles.iter().enumerate() {
            if in_circumcircle(&extended_points, tri, px, py) {
                bad_triangles.push(ti);
            }
        }

        // Find boundary polygon (edges not shared by multiple bad triangles)
        let mut edge_count: HashMap<(usize, usize), usize> = HashMap::new();
        for &ti in &bad_triangles {
            let tri = &triangles[ti];
            for k in 0..3 {
                let e = normalize_edge(tri[k], tri[(k + 1) % 3]);
                *edge_count.entry(e).or_insert(0) += 1;
            }
        }

        let polygon: Vec<(usize, usize)> = edge_count
            .into_iter()
            .filter(|&(_, count)| count == 1)
            .map(|(e, _)| e)
            .collect();

        // Remove bad triangles
        let mut new_triangles: Vec<[usize; 3]> = Vec::new();
        for (ti, tri) in triangles.iter().enumerate() {
            if !bad_triangles.contains(&ti) {
                new_triangles.push(*tri);
            }
        }

        // Add new triangles
        for (a, b) in polygon {
            new_triangles.push([a, b, i]);
        }

        triangles = new_triangles;
    }

    // Remove triangles containing super-triangle vertices
    let mut final_triangles: Vec<[usize; 3]> = Vec::new();
    for tri in triangles {
        if tri[0] < n && tri[1] < n && tri[2] < n {
            final_triangles.push(tri);
        }
    }

    let n_triangles = final_triangles.len();

    if n_triangles == 0 {
        return Err(Error::InvalidArgument {
            arg: "points",
            reason: "Delaunay triangulation produced no triangles (points may be collinear)"
                .to_string(),
        });
    }

    // Convert to tensors
    let mut simplices_data: Vec<i64> = Vec::with_capacity(n_triangles * 3);
    for tri in &final_triangles {
        simplices_data.push(tri[0] as i64);
        simplices_data.push(tri[1] as i64);
        simplices_data.push(tri[2] as i64);
    }

    // Build neighbor relationships
    let mut edge_to_triangle: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    for (ti, tri) in final_triangles.iter().enumerate() {
        for k in 0..3 {
            let e = normalize_edge(tri[k], tri[(k + 1) % 3]);
            edge_to_triangle.entry(e).or_default().push(ti);
        }
    }

    let mut neighbors_data: Vec<i64> = vec![-1; n_triangles * 3];
    for (ti, tri) in final_triangles.iter().enumerate() {
        for k in 0..3 {
            let e = normalize_edge(tri[k], tri[(k + 1) % 3]);
            if let Some(tris) = edge_to_triangle.get(&e) {
                for &other_ti in tris {
                    if other_ti != ti {
                        neighbors_data[ti * 3 + k] = other_ti as i64;
                    }
                }
            }
        }
    }

    // Vertex to simplex mapping
    let mut vertex_to_simplex: Vec<i64> = vec![-1; n];
    for (ti, tri) in final_triangles.iter().enumerate() {
        for &v in tri {
            if vertex_to_simplex[v] == -1 {
                vertex_to_simplex[v] = ti as i64;
            }
        }
    }

    // Find convex hull vertices (vertices on boundary triangles)
    let mut hull_vertices: HashSet<usize> = HashSet::new();
    for (ti, tri) in final_triangles.iter().enumerate() {
        for k in 0..3 {
            if neighbors_data[ti * 3 + k] == -1 {
                // This edge is on the boundary
                hull_vertices.insert(tri[k]);
                hull_vertices.insert(tri[(k + 1) % 3]);
            }
        }
    }
    let hull: Vec<i64> = hull_vertices.iter().map(|&v| v as i64).collect();

    Ok(Delaunay {
        points: points.clone(),
        simplices: Tensor::<R>::from_slice(&simplices_data, &[n_triangles, 3], device),
        neighbors: Tensor::<R>::from_slice(&neighbors_data, &[n_triangles, 3], device),
        vertex_to_simplex: Tensor::<R>::from_slice(&vertex_to_simplex, &[n], device),
        convex_hull: Tensor::<R>::from_slice(&hull, &[hull.len()], device),
    })
}

fn normalize_edge(a: usize, b: usize) -> (usize, usize) {
    if a < b { (a, b) } else { (b, a) }
}

fn in_circumcircle(points: &[f64], tri: &[usize; 3], px: f64, py: f64) -> bool {
    let ax = points[tri[0] * 2];
    let ay = points[tri[0] * 2 + 1];
    let bx = points[tri[1] * 2];
    let by = points[tri[1] * 2 + 1];
    let cx = points[tri[2] * 2];
    let cy = points[tri[2] * 2 + 1];

    let d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
    if d.abs() < 1e-10 {
        return false;
    }

    let ux = ((ax * ax + ay * ay) * (by - cy)
        + (bx * bx + by * by) * (cy - ay)
        + (cx * cx + cy * cy) * (ay - by))
        / d;
    let uy = ((ax * ax + ay * ay) * (cx - bx)
        + (bx * bx + by * by) * (ax - cx)
        + (cx * cx + cy * cy) * (bx - ax))
        / d;

    let r_sq = (ax - ux) * (ax - ux) + (ay - uy) * (ay - uy);
    let d_sq = (px - ux) * (px - ux) + (py - uy) * (py - uy);

    d_sq < r_sq
}

/// Find the simplex containing each query point.
///
/// Uses tensor operations - data stays on device.
pub fn delaunay_find_simplex_impl<R, C>(
    client: &C,
    tri: &Delaunay<R>,
    query: &Tensor<R>,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R>
        + ScalarOps<R>
        + ReduceOps<R>
        + CompareOps<R>
        + IndexingOps<R>
        + LogicalOps<R>
        + TypeConversionOps<R>
        + RuntimeClient<R>,
{
    let n_queries = query.shape()[0];
    let n_simplices = tri.simplices.shape()[0];
    let device = query.device();
    let dtype = query.dtype();

    // Get triangle vertex indices (need contiguous() before reshape)
    let v0_idx = tri
        .simplices
        .narrow(1, 0, 1)?
        .contiguous()
        .reshape(&[n_simplices])?;
    let v1_idx = tri
        .simplices
        .narrow(1, 1, 1)?
        .contiguous()
        .reshape(&[n_simplices])?;
    let v2_idx = tri
        .simplices
        .narrow(1, 2, 1)?
        .contiguous()
        .reshape(&[n_simplices])?;

    // Get vertex coordinates
    let p0 = client.index_select(&tri.points, 0, &v0_idx)?; // [n_simplices, 2]
    let p1 = client.index_select(&tri.points, 0, &v1_idx)?;
    let p2 = client.index_select(&tri.points, 0, &v2_idx)?;

    // Extract coordinates (need contiguous() before reshape)
    let x0 = p0.narrow(1, 0, 1)?.contiguous().reshape(&[n_simplices])?;
    let y0 = p0.narrow(1, 1, 1)?.contiguous().reshape(&[n_simplices])?;
    let x1 = p1.narrow(1, 0, 1)?.contiguous().reshape(&[n_simplices])?;
    let y1 = p1.narrow(1, 1, 1)?.contiguous().reshape(&[n_simplices])?;
    let x2 = p2.narrow(1, 0, 1)?.contiguous().reshape(&[n_simplices])?;
    let y2 = p2.narrow(1, 1, 1)?.contiguous().reshape(&[n_simplices])?;

    // Query point coordinates (need contiguous() before reshape)
    let qx = query.narrow(1, 0, 1)?.contiguous().reshape(&[n_queries])?;
    let qy = query.narrow(1, 1, 1)?.contiguous().reshape(&[n_queries])?;

    // Broadcast to [n_queries, n_simplices]
    let qx_exp = qx
        .unsqueeze(1)?
        .broadcast_to(&[n_queries, n_simplices])?
        .contiguous();
    let qy_exp = qy
        .unsqueeze(1)?
        .broadcast_to(&[n_queries, n_simplices])?
        .contiguous();

    let x0_exp = x0
        .unsqueeze(0)?
        .broadcast_to(&[n_queries, n_simplices])?
        .contiguous();
    let y0_exp = y0
        .unsqueeze(0)?
        .broadcast_to(&[n_queries, n_simplices])?
        .contiguous();
    let x1_exp = x1
        .unsqueeze(0)?
        .broadcast_to(&[n_queries, n_simplices])?
        .contiguous();
    let y1_exp = y1
        .unsqueeze(0)?
        .broadcast_to(&[n_queries, n_simplices])?
        .contiguous();
    let x2_exp = x2
        .unsqueeze(0)?
        .broadcast_to(&[n_queries, n_simplices])?
        .contiguous();
    let y2_exp = y2
        .unsqueeze(0)?
        .broadcast_to(&[n_queries, n_simplices])?
        .contiguous();

    // Compute barycentric coordinates
    // denom = (y1-y2)*(x0-x2) + (x2-x1)*(y0-y2)
    let y1_y2 = client.sub(&y1_exp, &y2_exp)?;
    let x0_x2 = client.sub(&x0_exp, &x2_exp)?;
    let x2_x1 = client.sub(&x2_exp, &x1_exp)?;
    let y0_y2 = client.sub(&y0_exp, &y2_exp)?;

    let denom = client.add(&client.mul(&y1_y2, &x0_x2)?, &client.mul(&x2_x1, &y0_y2)?)?;

    // a = ((y1-y2)*(px-x2) + (x2-x1)*(py-y2)) / denom
    let px_x2 = client.sub(&qx_exp, &x2_exp)?;
    let py_y2 = client.sub(&qy_exp, &y2_exp)?;

    let a_num = client.add(&client.mul(&y1_y2, &px_x2)?, &client.mul(&x2_x1, &py_y2)?)?;

    // b = ((y2-y0)*(px-x2) + (x0-x2)*(py-y2)) / denom
    let y2_y0 = client.sub(&y2_exp, &y0_exp)?;
    let b_num = client.add(&client.mul(&y2_y0, &px_x2)?, &client.mul(&x0_x2, &py_y2)?)?;

    // Safe division: where denom is too small, set a,b to invalid values
    let eps = Tensor::<R>::full_scalar(&[], dtype, 1e-10, device);
    let denom_abs = client.abs(&denom)?;
    let denom_valid_raw = client.gt(&denom_abs, &eps)?;
    // Cast to U8 for boolean (comparison ops may return same dtype as input)
    let denom_valid = client.cast(&denom_valid_raw, DType::U8)?;

    // Compute a, b (set to -1 where denom is invalid to ensure point is not "inside")
    let safe_denom = client.add(&denom, &eps)?; // Avoid div by zero
    let a = client.div(&a_num, &safe_denom)?;
    let b = client.div(&b_num, &safe_denom)?;
    let c = client.sub(
        &client.sub(
            &Tensor::<R>::ones(&[n_queries, n_simplices], dtype, device),
            &a,
        )?,
        &b,
    )?;

    // Point is inside if a >= -eps, b >= -eps, c >= -eps, and denom was valid
    let neg_eps = Tensor::<R>::full_scalar(&[], dtype, -1e-10, device);
    let a_ok_raw = client.ge(&a, &neg_eps)?;
    let b_ok_raw = client.ge(&b, &neg_eps)?;
    let c_ok_raw = client.ge(&c, &neg_eps)?;
    // Cast to U8 for boolean (comparison ops may return same dtype as input)
    let a_ok = client.cast(&a_ok_raw, DType::U8)?;
    let b_ok = client.cast(&b_ok_raw, DType::U8)?;
    let c_ok = client.cast(&c_ok_raw, DType::U8)?;

    let inside = client.logical_and(
        &client.logical_and(&a_ok, &b_ok)?,
        &client.logical_and(&c_ok, &denom_valid)?,
    )?;

    // Find first simplex containing each query (argmax of inside mask)
    let inside_f = client.cast(&inside, dtype)?;

    // Use argmax to find first True value along simplex dimension
    // If none found, argmax returns arbitrary index - we need to check
    let max_val = client.max(&inside_f, &[1], true)?; // [n_queries, 1]
    let found_raw = client.gt(
        &max_val.reshape(&[n_queries])?,
        &Tensor::<R>::zeros(&[n_queries], dtype, device),
    )?;
    // Cast to U8 for boolean (comparison ops may return same dtype as input)
    let found = client.cast(&found_raw, DType::U8)?;

    let simplex_idx = client.argmax(&inside_f, 1, false)?; // [n_queries]

    // Where not found, set to -1
    let neg_one = Tensor::<R>::full_scalar(&[n_queries], DType::I64, -1.0, device);
    let found_i64 = client.cast(&found, DType::I64)?;
    let not_found_i64 = client.sub(
        &Tensor::<R>::ones(&[n_queries], DType::I64, device),
        &found_i64,
    )?;

    // result = simplex_idx * found + (-1) * not_found
    let result = client.add(
        &client.mul(&simplex_idx, &found_i64)?,
        &client.mul(&neg_one, &not_found_i64)?,
    )?;

    Ok(result)
}

/// Get vertex neighbors.
pub fn delaunay_vertex_neighbors_impl<R, C>(
    _client: &C,
    tri: &Delaunay<R>,
) -> Result<(Tensor<R>, Tensor<R>)>
where
    R: Runtime,
    C: RuntimeClient<R>,
{
    let n = tri.points.shape()[0];
    let simplices_data: Vec<i64> = tri.simplices.to_vec();
    let n_simplices = tri.simplices.shape()[0];
    let device = tri.points.device();

    // Build adjacency
    let mut neighbors: Vec<HashSet<usize>> = vec![HashSet::new(); n];
    for s in 0..n_simplices {
        let v0 = simplices_data[s * 3] as usize;
        let v1 = simplices_data[s * 3 + 1] as usize;
        let v2 = simplices_data[s * 3 + 2] as usize;

        neighbors[v0].insert(v1);
        neighbors[v0].insert(v2);
        neighbors[v1].insert(v0);
        neighbors[v1].insert(v2);
        neighbors[v2].insert(v0);
        neighbors[v2].insert(v1);
    }

    // Convert to CSR format
    let mut indices: Vec<i64> = Vec::new();
    let mut indptr: Vec<i64> = vec![0];

    for vertex_neighbors in neighbors.iter().take(n) {
        for &neighbor in vertex_neighbors {
            indices.push(neighbor as i64);
        }
        indptr.push(indices.len() as i64);
    }

    Ok((
        Tensor::<R>::from_slice(&indices, &[indices.len()], device),
        Tensor::<R>::from_slice(&indptr, &[indptr.len()], device),
    ))
}
