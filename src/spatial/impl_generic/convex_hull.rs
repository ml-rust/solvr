//! Generic convex hull implementation.
//!
//! Gift wrapping for 2D, incremental convex hull for 3D.
//! All heavy computation uses tensor operations - data stays on device.

use crate::spatial::traits::convex_hull::ConvexHull;
use crate::spatial::{validate_points_2d, validate_points_dtype};
use numr::dtype::DType;
use numr::error::{Error, Result};
use numr::ops::{
    CompareOps, IndexingOps, ReduceOps, ScalarOps, SortingOps, TensorOps, TypeConversionOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Compute convex hull of a point set.
///
/// Uses tensor operations throughout - data stays on device.
pub fn convex_hull_impl<R, C>(client: &C, points: &Tensor<R>) -> Result<ConvexHull<R>>
where
    R: Runtime,
    C: TensorOps<R>
        + ScalarOps<R>
        + ReduceOps<R>
        + CompareOps<R>
        + IndexingOps<R>
        + SortingOps<R>
        + TypeConversionOps<R>
        + RuntimeClient<R>,
{
    validate_points_dtype(points.dtype(), "convex_hull")?;
    validate_points_2d(points.shape(), "convex_hull")?;

    let n = points.shape()[0];
    let d = points.shape()[1];

    if n < d + 1 {
        return Err(Error::InvalidArgument {
            arg: "points",
            reason: format!("Need at least {} points for {}-D convex hull", d + 1, d),
        });
    }

    match d {
        2 => convex_hull_2d_tensor(client, points),
        3 => convex_hull_3d_tensor(client, points),
        _ => Err(Error::InvalidArgument {
            arg: "points",
            reason: format!("Convex hull only implemented for 2D and 3D, got {}D", d),
        }),
    }
}

/// 2D convex hull using Gift Wrapping algorithm with tensor operations.
///
/// Algorithm:
/// 1. Find leftmost point using argmin (tensor op)
/// 2. For each hull vertex, find next vertex by computing cross products to all points (tensor op)
/// 3. Use argmax on cross products to find most counter-clockwise point
/// 4. Repeat until returning to start
fn convex_hull_2d_tensor<R, C>(client: &C, points: &Tensor<R>) -> Result<ConvexHull<R>>
where
    R: Runtime,
    C: TensorOps<R>
        + ScalarOps<R>
        + ReduceOps<R>
        + CompareOps<R>
        + IndexingOps<R>
        + SortingOps<R>
        + TypeConversionOps<R>
        + RuntimeClient<R>,
{
    let n = points.shape()[0];
    let device = points.device();
    let dtype = points.dtype();

    // Find leftmost point (smallest x coordinate) using tensor argmin
    let x_coords = points.narrow(1, 0, 1)?.contiguous().reshape(&[n])?;
    let start_idx_tensor = client.argmin(&x_coords, 0, false)?;

    // Extract start index (single scalar transfer - unavoidable for loop control)
    let start_idx: i64 = start_idx_tensor.to_vec::<i64>()[0];

    // Gift wrapping: iterate to find hull vertices
    let mut hull_indices: Vec<i64> = vec![start_idx];
    let mut current_idx = start_idx;

    // Maximum iterations is n (safeguard against infinite loops)
    for _ in 0..n {
        // Get current point coordinates using tensor index_select
        let curr_idx_tensor = Tensor::<R>::from_slice(&[current_idx], &[1], device);
        let curr_point = client.index_select(points, 0, &curr_idx_tensor)?; // [1, 2]

        // Broadcast current point to [n, 2] for vectorized subtraction
        let curr_broadcast = curr_point.broadcast_to(&[n, 2])?.contiguous();

        // Compute vectors from current point to all points: v = points - current
        let vectors = client.sub(points, &curr_broadcast)?; // [n, 2]

        // Extract vector components
        let vx = vectors.narrow(1, 0, 1)?.contiguous().reshape(&[n])?;
        let vy = vectors.narrow(1, 1, 1)?.contiguous().reshape(&[n])?;

        // For Gift Wrapping, we need to find the point that makes all other points
        // lie to the left (positive cross product).
        // For each candidate point q, compute cross product with all other points:
        // We want the point q such that for all other points p: cross(q-curr, p-curr) >= 0
        //
        // Simplified approach: for each point q, compute sum of signs of cross products
        // with all other points. The hull vertex maximizes this.
        //
        // Even simpler: use polar angle sorting from current point
        // angle = atan2(vy, vx), then find point with smallest angle > previous angle

        // Compute squared distances to handle zero vectors
        let dist_sq = client.add(&client.mul(&vx, &vx)?, &client.mul(&vy, &vy)?)?;

        // For each potential next point q, we compute cross products with all points
        // This is O(n²) but fully parallelizable on GPU
        // cross(v_q, v_p) = vx_q * vy_p - vy_q * vx_p

        // Broadcast for pairwise computation [n, n]
        let vx_row = vx.unsqueeze(0)?.broadcast_to(&[n, n])?.contiguous(); // v_p (columns)
        let vy_row = vy.unsqueeze(0)?.broadcast_to(&[n, n])?.contiguous();
        let vx_col = vx.unsqueeze(1)?.broadcast_to(&[n, n])?.contiguous(); // v_q (rows)
        let vy_col = vy.unsqueeze(1)?.broadcast_to(&[n, n])?.contiguous();

        // cross[q, p] = vx_q * vy_p - vy_q * vx_p
        let cross = client.sub(
            &client.mul(&vx_col, &vy_row)?,
            &client.mul(&vy_col, &vx_row)?,
        )?; // [n, n]

        // For point q to be valid next hull vertex, cross[q, p] >= 0 for all p
        // Count how many points are to the left (cross >= 0) for each candidate q
        let zero = Tensor::<R>::zeros(&[], dtype, device);
        let left_of_raw = client.ge(&cross, &zero)?;
        let left_of = client.cast(&left_of_raw, dtype)?;
        let left_count = client.sum(&left_of, &[1], false)?; // [n] - count of points to left of each candidate

        // The next hull vertex is the one with ALL other points to its left (count = n)
        // But we also need to exclude the current point (distance = 0)
        // Set score to -inf for current point
        let large_neg = Tensor::<R>::full_scalar(&[n], dtype, -1e10, device);
        let eps = Tensor::<R>::full_scalar(&[n], dtype, 1e-10, device);
        let is_current_raw = client.lt(&dist_sq, &eps)?;
        let is_current = client.cast(&is_current_raw, dtype)?;

        // score = left_count * (1 - is_current) + large_neg * is_current
        let one = Tensor::<R>::ones(&[n], dtype, device);
        let not_current = client.sub(&one, &is_current)?;
        let score = client.add(
            &client.mul(&left_count, &not_current)?,
            &client.mul(&large_neg, &is_current)?,
        )?;

        // Find point with maximum score (most points to its left)
        let next_idx_tensor = client.argmax(&score, 0, false)?;

        // Extract next index (single scalar transfer)
        let next_idx: i64 = next_idx_tensor.to_vec::<i64>()[0];

        // Check if we've returned to start
        if next_idx == start_idx {
            break;
        }

        hull_indices.push(next_idx);
        current_idx = next_idx;
    }

    let n_vertices = hull_indices.len();

    // Build hull tensors on device
    let hull_tensor = Tensor::<R>::from_slice(&hull_indices, &[n_vertices], device);

    // Create simplices (edges) as pairs of consecutive vertices
    let mut simplices_data: Vec<i64> = Vec::with_capacity(n_vertices * 2);
    for i in 0..n_vertices {
        simplices_data.push(hull_indices[i]);
        simplices_data.push(hull_indices[(i + 1) % n_vertices]);
    }

    // Compute area and perimeter using tensor operations
    let hull_coords = client.index_select(points, 0, &hull_tensor)?; // [n_vertices, 2]

    // Shifted coordinates for edge computations
    let shift_indices: Vec<i64> = (0..n_vertices)
        .map(|i| ((i + 1) % n_vertices) as i64)
        .collect();
    let shift_tensor = Tensor::<R>::from_slice(&shift_indices, &[n_vertices], device);
    let hull_coords_shifted = client.index_select(&hull_coords, 0, &shift_tensor)?;

    // Extract x, y coordinates
    let hx = hull_coords
        .narrow(1, 0, 1)?
        .contiguous()
        .reshape(&[n_vertices])?;
    let hy = hull_coords
        .narrow(1, 1, 1)?
        .contiguous()
        .reshape(&[n_vertices])?;
    let hx_next = hull_coords_shifted
        .narrow(1, 0, 1)?
        .contiguous()
        .reshape(&[n_vertices])?;
    let hy_next = hull_coords_shifted
        .narrow(1, 1, 1)?
        .contiguous()
        .reshape(&[n_vertices])?;

    // Perimeter: sum of edge lengths
    let dx = client.sub(&hx_next, &hx)?;
    let dy = client.sub(&hy_next, &hy)?;
    let edge_len_sq = client.add(&client.mul(&dx, &dx)?, &client.mul(&dy, &dy)?)?;
    let edge_len = client.sqrt(&edge_len_sq)?;
    let perimeter_tensor = client.sum(&edge_len, &[0], false)?;
    let perimeter: f64 = perimeter_tensor.to_vec::<f64>()[0];

    // Area using shoelace formula: 0.5 * |sum(x_i * y_{i+1} - x_{i+1} * y_i)|
    let cross_terms = client.sub(&client.mul(&hx, &hy_next)?, &client.mul(&hx_next, &hy)?)?;
    let area_sum = client.sum(&cross_terms, &[0], false)?;
    let area_tensor = client.abs(&area_sum)?;
    let area_scaled = client.mul_scalar(&area_tensor, 0.5)?;
    let area: f64 = area_scaled.to_vec::<f64>()[0];

    Ok(ConvexHull {
        points: points.clone(),
        vertices: hull_tensor,
        simplices: Tensor::<R>::from_slice(&simplices_data, &[n_vertices, 2], device),
        neighbors: None,
        equations: None,
        volume: area,
        area: perimeter,
    })
}

/// 3D convex hull using incremental algorithm with tensor operations.
///
/// Algorithm:
/// 1. Find initial tetrahedron using extreme points (tensor argmin/argmax)
/// 2. For each remaining point, use tensor ops to find visible faces
/// 3. Update face list incrementally
fn convex_hull_3d_tensor<R, C>(client: &C, points: &Tensor<R>) -> Result<ConvexHull<R>>
where
    R: Runtime,
    C: TensorOps<R>
        + ScalarOps<R>
        + ReduceOps<R>
        + CompareOps<R>
        + IndexingOps<R>
        + SortingOps<R>
        + TypeConversionOps<R>
        + RuntimeClient<R>,
{
    let n = points.shape()[0];
    let device = points.device();
    let dtype = points.dtype();

    // Find extreme points using tensor operations
    let x_coords = points.narrow(1, 0, 1)?.contiguous().reshape(&[n])?;
    let _y_coords = points.narrow(1, 1, 1)?.contiguous().reshape(&[n])?;
    let _z_coords = points.narrow(1, 2, 1)?.contiguous().reshape(&[n])?;

    // Find min/max x points
    let min_x_idx = client.argmin(&x_coords, 0, false)?;
    let max_x_idx = client.argmax(&x_coords, 0, false)?;

    let p0: i64 = min_x_idx.to_vec::<i64>()[0];
    let p1: i64 = max_x_idx.to_vec::<i64>()[0];

    // Find point furthest from line p0-p1 using tensor ops
    let p0_tensor = Tensor::<R>::from_slice(&[p0], &[1], device);
    let p1_tensor = Tensor::<R>::from_slice(&[p1], &[1], device);

    let pt0 = client.index_select(points, 0, &p0_tensor)?; // [1, 3]
    let pt1 = client.index_select(points, 0, &p1_tensor)?;

    // Direction vector d = pt1 - pt0
    let d = client.sub(&pt1, &pt0)?; // [1, 3]
    let d_broadcast = d.broadcast_to(&[n, 3])?.contiguous();

    // Vector from p0 to each point: v = points - pt0
    let pt0_broadcast = pt0.broadcast_to(&[n, 3])?.contiguous();
    let v = client.sub(points, &pt0_broadcast)?; // [n, 3]

    // Cross product d × v gives distance from line (magnitude)
    let dx = d_broadcast.narrow(1, 0, 1)?.contiguous().reshape(&[n])?;
    let dy = d_broadcast.narrow(1, 1, 1)?.contiguous().reshape(&[n])?;
    let dz = d_broadcast.narrow(1, 2, 1)?.contiguous().reshape(&[n])?;
    let vx = v.narrow(1, 0, 1)?.contiguous().reshape(&[n])?;
    let vy = v.narrow(1, 1, 1)?.contiguous().reshape(&[n])?;
    let vz = v.narrow(1, 2, 1)?.contiguous().reshape(&[n])?;

    // cross = d × v
    let cx = client.sub(&client.mul(&dy, &vz)?, &client.mul(&dz, &vy)?)?;
    let cy = client.sub(&client.mul(&dz, &vx)?, &client.mul(&dx, &vz)?)?;
    let cz = client.sub(&client.mul(&dx, &vy)?, &client.mul(&dy, &vx)?)?;

    // Distance squared from line
    let dist_line_sq = client.add(
        &client.add(&client.mul(&cx, &cx)?, &client.mul(&cy, &cy)?)?,
        &client.mul(&cz, &cz)?,
    )?;

    // Exclude p0 and p1 by setting their distances to -inf
    let neg_inf = Tensor::<R>::full_scalar(&[n], dtype, -1e30, device);
    let idx_f = Tensor::<R>::from_slice(&(0..n as i64).collect::<Vec<_>>(), &[n], device);
    let idx_f = client.cast(&idx_f, dtype)?;
    let p0_f = Tensor::<R>::full_scalar(&[n], dtype, p0 as f64, device);
    let p1_f = Tensor::<R>::full_scalar(&[n], dtype, p1 as f64, device);
    let is_p0_raw = client.eq(&idx_f, &p0_f)?;
    let is_p1_raw = client.eq(&idx_f, &p1_f)?;
    let is_p0 = client.cast(&is_p0_raw, dtype)?;
    let is_p1 = client.cast(&is_p1_raw, dtype)?;
    let is_excluded = client.add(&is_p0, &is_p1)?;
    let one = Tensor::<R>::ones(&[n], dtype, device);
    let not_excluded = client.sub(&one, &is_excluded)?;

    let dist_masked = client.add(
        &client.mul(&dist_line_sq, &not_excluded)?,
        &client.mul(&neg_inf, &is_excluded)?,
    )?;

    let p2_idx = client.argmax(&dist_masked, 0, false)?;
    let p2: i64 = p2_idx.to_vec::<i64>()[0];

    // Find point furthest from plane p0-p1-p2
    let p2_tensor = Tensor::<R>::from_slice(&[p2], &[1], device);
    let pt2 = client.index_select(points, 0, &p2_tensor)?;

    // Compute plane normal: (pt1-pt0) × (pt2-pt0)
    let e1 = client.sub(&pt1, &pt0)?; // [1, 3]
    let e2 = client.sub(&pt2, &pt0)?;

    let e1x = e1.narrow(1, 0, 1)?.contiguous().reshape(&[1])?;
    let e1y = e1.narrow(1, 1, 1)?.contiguous().reshape(&[1])?;
    let e1z = e1.narrow(1, 2, 1)?.contiguous().reshape(&[1])?;
    let e2x = e2.narrow(1, 0, 1)?.contiguous().reshape(&[1])?;
    let e2y = e2.narrow(1, 1, 1)?.contiguous().reshape(&[1])?;
    let e2z = e2.narrow(1, 2, 1)?.contiguous().reshape(&[1])?;

    let nx = client.sub(&client.mul(&e1y, &e2z)?, &client.mul(&e1z, &e2y)?)?;
    let ny = client.sub(&client.mul(&e1z, &e2x)?, &client.mul(&e1x, &e2z)?)?;
    let nz = client.sub(&client.mul(&e1x, &e2y)?, &client.mul(&e1y, &e2x)?)?;

    // Distance from plane for each point: dot(normal, point - pt0)
    let nx_b = nx.broadcast_to(&[n])?.contiguous();
    let ny_b = ny.broadcast_to(&[n])?.contiguous();
    let nz_b = nz.broadcast_to(&[n])?.contiguous();

    let dist_plane = client.add(
        &client.add(&client.mul(&nx_b, &vx)?, &client.mul(&ny_b, &vy)?)?,
        &client.mul(&nz_b, &vz)?,
    )?;
    let dist_plane_abs = client.abs(&dist_plane)?;

    // Exclude p0, p1, p2
    let p2_f = Tensor::<R>::full_scalar(&[n], dtype, p2 as f64, device);
    let is_p2_raw = client.eq(&idx_f, &p2_f)?;
    let is_p2 = client.cast(&is_p2_raw, dtype)?;
    let is_excluded3 = client.add(&is_excluded, &is_p2)?;
    let not_excluded3 = client.sub(&one, &client.minimum(&is_excluded3, &one)?)?;

    let dist_plane_masked = client.add(
        &client.mul(&dist_plane_abs, &not_excluded3)?,
        &client.mul(&neg_inf, &client.minimum(&is_excluded3, &one)?)?,
    )?;

    let p3_idx = client.argmax(&dist_plane_masked, 0, false)?;
    let p3: i64 = p3_idx.to_vec::<i64>()[0];

    // Check for coplanarity
    let max_dist: f64 = client.max(&dist_plane_masked, &[0], false)?.to_vec::<f64>()[0];
    if max_dist < 1e-10 {
        return Err(Error::InvalidArgument {
            arg: "points",
            reason: "Points are coplanar, cannot form 3D convex hull".to_string(),
        });
    }

    // Build initial tetrahedron faces
    // Faces: [p0,p1,p2], [p0,p1,p3], [p0,p2,p3], [p1,p2,p3]
    let mut faces: Vec<[i64; 3]> = vec![[p0, p1, p2], [p0, p1, p3], [p0, p2, p3], [p1, p2, p3]];

    // Ensure faces are oriented outward (normal points away from centroid)
    let p3_tensor = Tensor::<R>::from_slice(&[p3], &[1], device);
    let pt3 = client.index_select(points, 0, &p3_tensor)?;
    let centroid = client.mul_scalar(
        &client.add(&client.add(&client.add(&pt0, &pt1)?, &pt2)?, &pt3)?,
        0.25,
    )?; // [1, 3]

    orient_faces_outward(client, points, &mut faces, &centroid)?;

    // Incremental convex hull: add remaining points
    let initial_set: std::collections::HashSet<i64> = [p0, p1, p2, p3].iter().copied().collect();

    for i in 0..n {
        let i64_val = i as i64;
        if initial_set.contains(&i64_val) {
            continue;
        }

        // Get point i coordinates
        let pt_i = points.narrow(0, i, 1)?.contiguous(); // [1, 3]

        // Find faces visible from this point using tensor ops
        let visible = find_visible_faces_tensor(client, points, &faces, &pt_i)?;

        if visible.is_empty() {
            continue; // Point is inside hull
        }

        // Find horizon edges (edges shared by exactly one visible face)
        let horizon = find_horizon_edges(&faces, &visible);

        // Remove visible faces
        let mut new_faces: Vec<[i64; 3]> = Vec::new();
        for (j, face) in faces.iter().enumerate() {
            if !visible.contains(&j) {
                new_faces.push(*face);
            }
        }

        // Add new faces connecting horizon edges to point
        for (a, b) in horizon {
            new_faces.push([a, b, i64_val]);
        }

        faces = new_faces;

        // Re-orient new faces
        orient_faces_outward(client, points, &mut faces, &centroid)?;
    }

    // Collect unique vertices
    let mut vertex_set: std::collections::HashSet<i64> = std::collections::HashSet::new();
    for face in &faces {
        vertex_set.insert(face[0]);
        vertex_set.insert(face[1]);
        vertex_set.insert(face[2]);
    }
    let vertices: Vec<i64> = vertex_set.into_iter().collect();

    // Build simplices tensor
    let mut simplices_data: Vec<i64> = Vec::with_capacity(faces.len() * 3);
    for face in &faces {
        simplices_data.push(face[0]);
        simplices_data.push(face[1]);
        simplices_data.push(face[2]);
    }

    // Compute volume and surface area using tensor operations
    let (volume, area) = compute_3d_hull_metrics(client, points, &faces)?;

    Ok(ConvexHull {
        points: points.clone(),
        vertices: Tensor::<R>::from_slice(&vertices, &[vertices.len()], device),
        simplices: Tensor::<R>::from_slice(&simplices_data, &[faces.len(), 3], device),
        neighbors: None,
        equations: None,
        volume,
        area,
    })
}

/// Orient faces so normals point outward (away from centroid).
fn orient_faces_outward<R, C>(
    client: &C,
    points: &Tensor<R>,
    faces: &mut [[i64; 3]],
    centroid: &Tensor<R>,
) -> Result<()>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + IndexingOps<R> + RuntimeClient<R>,
{
    let device = points.device();

    for face in faces.iter_mut() {
        // Get face vertices
        let indices = Tensor::<R>::from_slice(&[face[0], face[1], face[2]], &[3], device);
        let verts = client.index_select(points, 0, &indices)?; // [3, 3]

        let v0 = verts.narrow(0, 0, 1)?.contiguous(); // [1, 3]
        let v1 = verts.narrow(0, 1, 1)?.contiguous();
        let v2 = verts.narrow(0, 2, 1)?.contiguous();

        // Compute face normal
        let e1 = client.sub(&v1, &v0)?;
        let e2 = client.sub(&v2, &v0)?;

        let e1x = e1.narrow(1, 0, 1)?.contiguous().reshape(&[1])?;
        let e1y = e1.narrow(1, 1, 1)?.contiguous().reshape(&[1])?;
        let e1z = e1.narrow(1, 2, 1)?.contiguous().reshape(&[1])?;
        let e2x = e2.narrow(1, 0, 1)?.contiguous().reshape(&[1])?;
        let e2y = e2.narrow(1, 1, 1)?.contiguous().reshape(&[1])?;
        let e2z = e2.narrow(1, 2, 1)?.contiguous().reshape(&[1])?;

        let nx = client.sub(&client.mul(&e1y, &e2z)?, &client.mul(&e1z, &e2y)?)?;
        let ny = client.sub(&client.mul(&e1z, &e2x)?, &client.mul(&e1x, &e2z)?)?;
        let nz = client.sub(&client.mul(&e1x, &e2y)?, &client.mul(&e1y, &e2x)?)?;

        // Face center
        let face_center =
            client.mul_scalar(&client.add(&client.add(&v0, &v1)?, &v2)?, 1.0 / 3.0)?;

        // Vector from face center to centroid
        let to_centroid = client.sub(centroid, &face_center)?;

        let tcx = to_centroid.narrow(1, 0, 1)?.contiguous().reshape(&[1])?;
        let tcy = to_centroid.narrow(1, 1, 1)?.contiguous().reshape(&[1])?;
        let tcz = to_centroid.narrow(1, 2, 1)?.contiguous().reshape(&[1])?;

        // Dot product: if positive, normal points inward -> flip
        let dot = client.add(
            &client.add(&client.mul(&nx, &tcx)?, &client.mul(&ny, &tcy)?)?,
            &client.mul(&nz, &tcz)?,
        )?;

        let dot_val: f64 = dot.to_vec::<f64>()[0];
        if dot_val > 0.0 {
            face.swap(1, 2);
        }
    }

    Ok(())
}

/// Find faces visible from a point using tensor operations.
fn find_visible_faces_tensor<R, C>(
    client: &C,
    points: &Tensor<R>,
    faces: &[[i64; 3]],
    point: &Tensor<R>,
) -> Result<Vec<usize>>
where
    R: Runtime,
    C: TensorOps<R>
        + ScalarOps<R>
        + CompareOps<R>
        + IndexingOps<R>
        + TypeConversionOps<R>
        + RuntimeClient<R>,
{
    let device = points.device();
    let n_faces = faces.len();
    let dtype = points.dtype();

    if n_faces == 0 {
        return Ok(Vec::new());
    }

    // Build face vertex indices tensor
    let mut face_indices: Vec<i64> = Vec::with_capacity(n_faces * 3);
    for face in faces {
        face_indices.push(face[0]);
        face_indices.push(face[1]);
        face_indices.push(face[2]);
    }
    let face_idx_tensor = Tensor::<R>::from_slice(&face_indices, &[n_faces * 3], device);

    // Get all face vertices at once
    let all_verts = client.index_select(points, 0, &face_idx_tensor)?; // [n_faces*3, 3]
    let all_verts = all_verts.reshape(&[n_faces, 3, 3])?; // [n_faces, 3 vertices, 3 coords]

    // Extract v0, v1, v2 for all faces
    let v0 = all_verts
        .narrow(1, 0, 1)?
        .contiguous()
        .reshape(&[n_faces, 3])?;
    let v1 = all_verts
        .narrow(1, 1, 1)?
        .contiguous()
        .reshape(&[n_faces, 3])?;
    let v2 = all_verts
        .narrow(1, 2, 1)?
        .contiguous()
        .reshape(&[n_faces, 3])?;

    // Compute face normals for all faces at once
    let e1 = client.sub(&v1, &v0)?; // [n_faces, 3]
    let e2 = client.sub(&v2, &v0)?;

    let e1x = e1.narrow(1, 0, 1)?.contiguous().reshape(&[n_faces])?;
    let e1y = e1.narrow(1, 1, 1)?.contiguous().reshape(&[n_faces])?;
    let e1z = e1.narrow(1, 2, 1)?.contiguous().reshape(&[n_faces])?;
    let e2x = e2.narrow(1, 0, 1)?.contiguous().reshape(&[n_faces])?;
    let e2y = e2.narrow(1, 1, 1)?.contiguous().reshape(&[n_faces])?;
    let e2z = e2.narrow(1, 2, 1)?.contiguous().reshape(&[n_faces])?;

    let nx = client.sub(&client.mul(&e1y, &e2z)?, &client.mul(&e1z, &e2y)?)?;
    let ny = client.sub(&client.mul(&e1z, &e2x)?, &client.mul(&e1x, &e2z)?)?;
    let nz = client.sub(&client.mul(&e1x, &e2y)?, &client.mul(&e1y, &e2x)?)?;

    // Vector from v0 to point for all faces
    let point_broadcast = point.broadcast_to(&[n_faces, 3])?.contiguous();
    let to_point = client.sub(&point_broadcast, &v0)?; // [n_faces, 3]

    let tpx = to_point.narrow(1, 0, 1)?.contiguous().reshape(&[n_faces])?;
    let tpy = to_point.narrow(1, 1, 1)?.contiguous().reshape(&[n_faces])?;
    let tpz = to_point.narrow(1, 2, 1)?.contiguous().reshape(&[n_faces])?;

    // Dot product: positive means point is above face (face is visible)
    let dot = client.add(
        &client.add(&client.mul(&nx, &tpx)?, &client.mul(&ny, &tpy)?)?,
        &client.mul(&nz, &tpz)?,
    )?; // [n_faces]

    // Face is visible if dot > epsilon
    let eps = Tensor::<R>::full_scalar(&[n_faces], dtype, 1e-10, device);
    let visible_raw = client.gt(&dot, &eps)?;
    let visible = client.cast(&visible_raw, DType::U8)?;

    // Extract visible face indices
    let visible_vec: Vec<u8> = visible.to_vec();
    let result: Vec<usize> = visible_vec
        .iter()
        .enumerate()
        .filter(|(_, v)| **v > 0)
        .map(|(i, _)| i)
        .collect();

    Ok(result)
}

/// Find horizon edges (edges on boundary between visible and non-visible faces).
fn find_horizon_edges(faces: &[[i64; 3]], visible: &[usize]) -> Vec<(i64, i64)> {
    use std::collections::HashMap;

    let mut edge_count: HashMap<(i64, i64), usize> = HashMap::new();

    for &fi in visible {
        let face = &faces[fi];
        for k in 0..3 {
            let a = face[k];
            let b = face[(k + 1) % 3];
            let e = if a < b { (a, b) } else { (b, a) };
            *edge_count.entry(e).or_insert(0) += 1;
        }
    }

    edge_count
        .into_iter()
        .filter(|&(_, count)| count == 1)
        .map(|(e, _)| e)
        .collect()
}

/// Compute volume and surface area for 3D hull using tensor operations.
fn compute_3d_hull_metrics<R, C>(
    client: &C,
    points: &Tensor<R>,
    faces: &[[i64; 3]],
) -> Result<(f64, f64)>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + ReduceOps<R> + IndexingOps<R> + RuntimeClient<R>,
{
    let device = points.device();
    let _dtype = points.dtype();
    let n_faces = faces.len();

    if n_faces == 0 {
        return Ok((0.0, 0.0));
    }

    // Build face vertex indices tensor
    let mut face_indices: Vec<i64> = Vec::with_capacity(n_faces * 3);
    for face in faces {
        face_indices.push(face[0]);
        face_indices.push(face[1]);
        face_indices.push(face[2]);
    }
    let face_idx_tensor = Tensor::<R>::from_slice(&face_indices, &[n_faces * 3], device);

    // Get all face vertices
    let all_verts = client.index_select(points, 0, &face_idx_tensor)?;
    let all_verts = all_verts.reshape(&[n_faces, 3, 3])?;

    let v0 = all_verts
        .narrow(1, 0, 1)?
        .contiguous()
        .reshape(&[n_faces, 3])?;
    let v1 = all_verts
        .narrow(1, 1, 1)?
        .contiguous()
        .reshape(&[n_faces, 3])?;
    let v2 = all_verts
        .narrow(1, 2, 1)?
        .contiguous()
        .reshape(&[n_faces, 3])?;

    // Volume using signed tetrahedra from origin
    // det = v0 · (v1 × v2)
    let v0x = v0.narrow(1, 0, 1)?.contiguous().reshape(&[n_faces])?;
    let v0y = v0.narrow(1, 1, 1)?.contiguous().reshape(&[n_faces])?;
    let v0z = v0.narrow(1, 2, 1)?.contiguous().reshape(&[n_faces])?;
    let v1x = v1.narrow(1, 0, 1)?.contiguous().reshape(&[n_faces])?;
    let v1y = v1.narrow(1, 1, 1)?.contiguous().reshape(&[n_faces])?;
    let v1z = v1.narrow(1, 2, 1)?.contiguous().reshape(&[n_faces])?;
    let v2x = v2.narrow(1, 0, 1)?.contiguous().reshape(&[n_faces])?;
    let v2y = v2.narrow(1, 1, 1)?.contiguous().reshape(&[n_faces])?;
    let v2z = v2.narrow(1, 2, 1)?.contiguous().reshape(&[n_faces])?;

    // v1 × v2
    let cx = client.sub(&client.mul(&v1y, &v2z)?, &client.mul(&v1z, &v2y)?)?;
    let cy = client.sub(&client.mul(&v1z, &v2x)?, &client.mul(&v1x, &v2z)?)?;
    let cz = client.sub(&client.mul(&v1x, &v2y)?, &client.mul(&v1y, &v2x)?)?;

    // v0 · (v1 × v2)
    let det = client.add(
        &client.add(&client.mul(&v0x, &cx)?, &client.mul(&v0y, &cy)?)?,
        &client.mul(&v0z, &cz)?,
    )?;

    let volume_sum = client.sum(&det, &[0], false)?;
    let volume_abs = client.abs(&volume_sum)?;
    let volume_scaled = client.mul_scalar(&volume_abs, 1.0 / 6.0)?;
    let volume: f64 = volume_scaled.to_vec::<f64>()[0];

    // Surface area: sum of triangle areas
    // Area = 0.5 * |e1 × e2|
    let e1 = client.sub(&v1, &v0)?;
    let e2 = client.sub(&v2, &v0)?;

    let e1x = e1.narrow(1, 0, 1)?.contiguous().reshape(&[n_faces])?;
    let e1y = e1.narrow(1, 1, 1)?.contiguous().reshape(&[n_faces])?;
    let e1z = e1.narrow(1, 2, 1)?.contiguous().reshape(&[n_faces])?;
    let e2x = e2.narrow(1, 0, 1)?.contiguous().reshape(&[n_faces])?;
    let e2y = e2.narrow(1, 1, 1)?.contiguous().reshape(&[n_faces])?;
    let e2z = e2.narrow(1, 2, 1)?.contiguous().reshape(&[n_faces])?;

    let ax = client.sub(&client.mul(&e1y, &e2z)?, &client.mul(&e1z, &e2y)?)?;
    let ay = client.sub(&client.mul(&e1z, &e2x)?, &client.mul(&e1x, &e2z)?)?;
    let az = client.sub(&client.mul(&e1x, &e2y)?, &client.mul(&e1y, &e2x)?)?;

    let area_sq = client.add(
        &client.add(&client.mul(&ax, &ax)?, &client.mul(&ay, &ay)?)?,
        &client.mul(&az, &az)?,
    )?;
    let face_areas = client.sqrt(&area_sq)?;
    let face_areas_half = client.mul_scalar(&face_areas, 0.5)?;
    let total_area = client.sum(&face_areas_half, &[0], false)?;
    let area: f64 = total_area.to_vec::<f64>()[0];

    Ok((volume, area))
}

/// Test if points are inside the convex hull.
///
/// Uses tensor operations - data stays on device.
pub fn convex_hull_contains_impl<R, C>(
    client: &C,
    hull: &ConvexHull<R>,
    points: &Tensor<R>,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R>
        + ScalarOps<R>
        + ReduceOps<R>
        + CompareOps<R>
        + IndexingOps<R>
        + TypeConversionOps<R>
        + RuntimeClient<R>,
{
    let d = hull.points.shape()[1];

    match d {
        2 => convex_hull_contains_2d_tensor(client, hull, points),
        3 => convex_hull_contains_3d_tensor(client, hull, points),
        _ => Err(Error::InvalidArgument {
            arg: "hull",
            reason: format!("Contains only implemented for 2D and 3D, got {}D", d),
        }),
    }
}

/// 2D point-in-polygon using tensor operations - data stays on device.
fn convex_hull_contains_2d_tensor<R, C>(
    client: &C,
    hull: &ConvexHull<R>,
    points: &Tensor<R>,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R>
        + ScalarOps<R>
        + ReduceOps<R>
        + CompareOps<R>
        + IndexingOps<R>
        + TypeConversionOps<R>
        + RuntimeClient<R>,
{
    let n_test = points.shape()[0];
    let n_hull = hull.vertices.shape()[0];
    let device = points.device();
    let dtype = points.dtype();

    // Get hull vertex coordinates using index_select
    let hull_coords = client.index_select(&hull.points, 0, &hull.vertices)?;

    // Create shifted indices for edge endpoints
    let shift_indices: Vec<i64> = (0..n_hull).map(|i| ((i + 1) % n_hull) as i64).collect();
    let shift_idx_tensor = Tensor::<R>::from_slice(&shift_indices, &[n_hull], device);
    let hull_coords_shifted = client.index_select(&hull_coords, 0, &shift_idx_tensor)?;

    // Extract coordinates
    let x1 = hull_coords
        .narrow(1, 0, 1)?
        .contiguous()
        .reshape(&[n_hull])?;
    let y1 = hull_coords
        .narrow(1, 1, 1)?
        .contiguous()
        .reshape(&[n_hull])?;
    let x2 = hull_coords_shifted
        .narrow(1, 0, 1)?
        .contiguous()
        .reshape(&[n_hull])?;
    let y2 = hull_coords_shifted
        .narrow(1, 1, 1)?
        .contiguous()
        .reshape(&[n_hull])?;

    let px = points.narrow(1, 0, 1)?.contiguous().reshape(&[n_test])?;
    let py = points.narrow(1, 1, 1)?.contiguous().reshape(&[n_test])?;

    // Broadcast for pairwise computation
    let px_exp = px
        .unsqueeze(1)?
        .broadcast_to(&[n_test, n_hull])?
        .contiguous();
    let py_exp = py
        .unsqueeze(1)?
        .broadcast_to(&[n_test, n_hull])?
        .contiguous();
    let x1_exp = x1
        .unsqueeze(0)?
        .broadcast_to(&[n_test, n_hull])?
        .contiguous();
    let y1_exp = y1
        .unsqueeze(0)?
        .broadcast_to(&[n_test, n_hull])?
        .contiguous();
    let x2_exp = x2
        .unsqueeze(0)?
        .broadcast_to(&[n_test, n_hull])?
        .contiguous();
    let y2_exp = y2
        .unsqueeze(0)?
        .broadcast_to(&[n_test, n_hull])?
        .contiguous();

    // Cross products
    let dx = client.sub(&x2_exp, &x1_exp)?;
    let dy = client.sub(&y2_exp, &y1_exp)?;
    let dpx = client.sub(&px_exp, &x1_exp)?;
    let dpy = client.sub(&py_exp, &y1_exp)?;

    let cross = client.sub(&client.mul(&dx, &dpy)?, &client.mul(&dy, &dpx)?)?;

    // Point is inside if all cross products >= -epsilon
    let epsilon = Tensor::<R>::full_scalar(&[], dtype, -1e-10, device);
    let inside_edge_raw = client.ge(&cross, &epsilon)?;
    let inside_edge = client.cast(&inside_edge_raw, DType::U8)?;

    let inside_f = client.cast(&inside_edge, dtype)?;
    let all_inside = client.min(&inside_f, &[1], false)?;

    Ok(all_inside)
}

/// 3D point-in-hull using tensor operations - data stays on device.
fn convex_hull_contains_3d_tensor<R, C>(
    client: &C,
    hull: &ConvexHull<R>,
    points: &Tensor<R>,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R>
        + ScalarOps<R>
        + ReduceOps<R>
        + CompareOps<R>
        + IndexingOps<R>
        + TypeConversionOps<R>
        + RuntimeClient<R>,
{
    let n_test = points.shape()[0];
    let n_faces = hull.simplices.shape()[0];
    let device = points.device();
    let dtype = points.dtype();

    // Get face vertex indices
    let v0_idx = hull
        .simplices
        .narrow(1, 0, 1)?
        .contiguous()
        .reshape(&[n_faces])?;
    let v1_idx = hull
        .simplices
        .narrow(1, 1, 1)?
        .contiguous()
        .reshape(&[n_faces])?;
    let v2_idx = hull
        .simplices
        .narrow(1, 2, 1)?
        .contiguous()
        .reshape(&[n_faces])?;

    // Get vertex coordinates
    let v0 = client.index_select(&hull.points, 0, &v0_idx)?;
    let v1 = client.index_select(&hull.points, 0, &v1_idx)?;
    let v2 = client.index_select(&hull.points, 0, &v2_idx)?;

    // Compute face normals
    let e1 = client.sub(&v1, &v0)?;
    let e2 = client.sub(&v2, &v0)?;

    let e1x = e1.narrow(1, 0, 1)?.contiguous().reshape(&[n_faces])?;
    let e1y = e1.narrow(1, 1, 1)?.contiguous().reshape(&[n_faces])?;
    let e1z = e1.narrow(1, 2, 1)?.contiguous().reshape(&[n_faces])?;
    let e2x = e2.narrow(1, 0, 1)?.contiguous().reshape(&[n_faces])?;
    let e2y = e2.narrow(1, 1, 1)?.contiguous().reshape(&[n_faces])?;
    let e2z = e2.narrow(1, 2, 1)?.contiguous().reshape(&[n_faces])?;

    let nx = client.sub(&client.mul(&e1y, &e2z)?, &client.mul(&e1z, &e2y)?)?;
    let ny = client.sub(&client.mul(&e1z, &e2x)?, &client.mul(&e1x, &e2z)?)?;
    let nz = client.sub(&client.mul(&e1x, &e2y)?, &client.mul(&e1y, &e2x)?)?;

    // Test point coordinates
    let test_x = points.narrow(1, 0, 1)?.contiguous().reshape(&[n_test])?;
    let test_y = points.narrow(1, 1, 1)?.contiguous().reshape(&[n_test])?;
    let test_z = points.narrow(1, 2, 1)?.contiguous().reshape(&[n_test])?;

    let v0x = v0.narrow(1, 0, 1)?.contiguous().reshape(&[n_faces])?;
    let v0y = v0.narrow(1, 1, 1)?.contiguous().reshape(&[n_faces])?;
    let v0z = v0.narrow(1, 2, 1)?.contiguous().reshape(&[n_faces])?;

    // Broadcast for pairwise computation
    let test_x_exp = test_x
        .unsqueeze(1)?
        .broadcast_to(&[n_test, n_faces])?
        .contiguous();
    let test_y_exp = test_y
        .unsqueeze(1)?
        .broadcast_to(&[n_test, n_faces])?
        .contiguous();
    let test_z_exp = test_z
        .unsqueeze(1)?
        .broadcast_to(&[n_test, n_faces])?
        .contiguous();

    let v0x_exp = v0x
        .unsqueeze(0)?
        .broadcast_to(&[n_test, n_faces])?
        .contiguous();
    let v0y_exp = v0y
        .unsqueeze(0)?
        .broadcast_to(&[n_test, n_faces])?
        .contiguous();
    let v0z_exp = v0z
        .unsqueeze(0)?
        .broadcast_to(&[n_test, n_faces])?
        .contiguous();

    let nx_exp = nx
        .unsqueeze(0)?
        .broadcast_to(&[n_test, n_faces])?
        .contiguous();
    let ny_exp = ny
        .unsqueeze(0)?
        .broadcast_to(&[n_test, n_faces])?
        .contiguous();
    let nz_exp = nz
        .unsqueeze(0)?
        .broadcast_to(&[n_test, n_faces])?
        .contiguous();

    // Compute (point - v0)
    let dx = client.sub(&test_x_exp, &v0x_exp)?;
    let dy = client.sub(&test_y_exp, &v0y_exp)?;
    let dz = client.sub(&test_z_exp, &v0z_exp)?;

    // Dot product
    let dot = client.add(
        &client.add(&client.mul(&nx_exp, &dx)?, &client.mul(&ny_exp, &dy)?)?,
        &client.mul(&nz_exp, &dz)?,
    )?;

    // Point is inside if dot <= epsilon for all faces
    let epsilon = Tensor::<R>::full_scalar(&[], dtype, 1e-10, device);
    let below_face_raw = client.le(&dot, &epsilon)?;
    let below_face = client.cast(&below_face_raw, DType::U8)?;

    let below_f = client.cast(&below_face, dtype)?;
    let all_below = client.min(&below_f, &[1], false)?;

    Ok(all_below)
}
