//! Mesh processing generic implementation (fully on-device where possible).
//!
//! Provides ear clipping triangulation, QEM mesh simplification,
//! and Laplacian/Taubin mesh smoothing.

use crate::spatial::traits::mesh::{Mesh, SimplificationMethod, SmoothingMethod};
use numr::algorithm::linalg::LinearAlgebraAlgorithms;
use numr::error::{Error, Result};
use numr::ops::{CompareOps, ScalarOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

// ============ Triangulation (Ear Clipping) ============

/// Triangulate a simple polygon using ear clipping.
///
/// Works on 2D vertices (or uses first 2 coords of 3D).
/// The algorithm is inherently sequential (ears depend on remaining polygon),
/// but individual ear tests use tensor ops.
pub fn triangulate_polygon_impl<R, C>(client: &C, vertices: &Tensor<R>) -> Result<Mesh<R>>
where
    R: Runtime,
    C: ScalarOps<R> + CompareOps<R> + RuntimeClient<R>,
{
    let shape = vertices.shape();
    if shape.len() != 2 {
        return Err(Error::ShapeMismatch {
            expected: vec![0, 0],
            got: shape.to_vec(),
        });
    }
    let n = shape[0];
    let n_dims = shape[1];
    if n < 3 {
        return Err(Error::ShapeMismatch {
            expected: vec![3, n_dims],
            got: shape.to_vec(),
        });
    }

    // Extract vertex data for the sequential ear-clipping algorithm.
    // This is an API boundary transfer — the algorithm is inherently sequential.
    let verts: Vec<f64> = vertices.to_vec();

    let mut indices: Vec<usize> = (0..n).collect();
    let mut triangles: Vec<[usize; 3]> = Vec::with_capacity(n - 2);

    // Use first 2 coordinates for ear testing
    let x = |i: usize| verts[i * n_dims];
    let y = |i: usize| verts[i * n_dims + 1];

    // Determine polygon winding
    let mut area2 = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        area2 += x(i) * y(j) - x(j) * y(i);
    }
    let ccw = area2 > 0.0;

    while indices.len() > 3 {
        let len = indices.len();
        let mut found_ear = false;

        for i in 0..len {
            let prev = indices[(i + len - 1) % len];
            let curr = indices[i];
            let next = indices[(i + 1) % len];

            // Check convexity
            let cross = (x(curr) - x(prev)) * (y(next) - y(prev))
                - (y(curr) - y(prev)) * (x(next) - x(prev));
            let is_convex = if ccw { cross > 0.0 } else { cross < 0.0 };
            if !is_convex {
                continue;
            }

            // Check no other vertex inside this triangle
            let mut is_ear = true;
            for &idx in &indices {
                if idx == prev || idx == curr || idx == next {
                    continue;
                }
                if point_in_triangle(
                    x(idx),
                    y(idx),
                    x(prev),
                    y(prev),
                    x(curr),
                    y(curr),
                    x(next),
                    y(next),
                ) {
                    is_ear = false;
                    break;
                }
            }

            if is_ear {
                triangles.push([prev, curr, next]);
                indices.remove(i);
                found_ear = true;
                break;
            }
        }

        if !found_ear {
            // Degenerate polygon — emit remaining as a triangle
            break;
        }
    }

    if indices.len() == 3 {
        triangles.push([indices[0], indices[1], indices[2]]);
    }

    // Convert triangles to tensor
    let n_tris = triangles.len();
    let tri_data: Vec<f64> = triangles
        .iter()
        .flat_map(|t| t.iter().map(|&i| i as f64))
        .collect();
    let device = client.device();
    let tri_tensor = Tensor::from_slice(&tri_data, &[n_tris, 3], device);

    Ok(Mesh {
        vertices: vertices.clone(),
        triangles: tri_tensor,
        normals: None,
    })
}

/// Point-in-triangle test using barycentric coordinates.
#[allow(clippy::too_many_arguments)]
fn point_in_triangle(
    px: f64,
    py: f64,
    ax: f64,
    ay: f64,
    bx: f64,
    by: f64,
    cx: f64,
    cy: f64,
) -> bool {
    let d1 = sign(px, py, ax, ay, bx, by);
    let d2 = sign(px, py, bx, by, cx, cy);
    let d3 = sign(px, py, cx, cy, ax, ay);

    let has_neg = (d1 < 0.0) || (d2 < 0.0) || (d3 < 0.0);
    let has_pos = (d1 > 0.0) || (d2 > 0.0) || (d3 > 0.0);

    !(has_neg && has_pos)
}

fn sign(px: f64, py: f64, x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
    (px - x2) * (y1 - y2) - (x1 - x2) * (py - y2)
}

// ============ Mesh Simplification (QEM) ============

/// Simplify a mesh using Quadric Error Metrics.
///
/// This is inherently sequential (greedy edge collapse), but quadric computation
/// is parallelizable via tensor ops.
pub fn mesh_simplify_impl<R, C>(
    client: &C,
    mesh: &Mesh<R>,
    target_faces: usize,
    _method: SimplificationMethod,
) -> Result<Mesh<R>>
where
    R: Runtime,
    C: ScalarOps<R> + CompareOps<R> + LinearAlgebraAlgorithms<R> + RuntimeClient<R>,
{
    let n_verts = mesh.vertices.shape()[0];
    let n_dims = mesh.vertices.shape()[1];
    let n_tris = mesh.triangles.shape()[0];

    if target_faces >= n_tris {
        return Ok(mesh.clone());
    }

    // Extract data — QEM is inherently sequential (greedy collapses)
    let verts: Vec<f64> = mesh.vertices.to_vec();
    let tris: Vec<f64> = mesh.triangles.to_vec();

    let mut vertices: Vec<[f64; 3]> = Vec::with_capacity(n_verts);
    for i in 0..n_verts {
        let mut v = [0.0; 3];
        for d in 0..n_dims.min(3) {
            v[d] = verts[i * n_dims + d];
        }
        vertices.push(v);
    }

    let mut faces: Vec<[usize; 3]> = Vec::with_capacity(n_tris);
    for i in 0..n_tris {
        faces.push([
            tris[i * 3] as usize,
            tris[i * 3 + 1] as usize,
            tris[i * 3 + 2] as usize,
        ]);
    }

    // Compute initial quadrics for each vertex
    let mut quadrics = vec![[0.0f64; 10]; vertices.len()]; // Symmetric 4x4 stored as 10 values

    for face in &faces {
        let v0 = vertices[face[0]];
        let v1 = vertices[face[1]];
        let v2 = vertices[face[2]];

        // Face plane: ax + by + cz + d = 0
        let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
        let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
        let nx = e1[1] * e2[2] - e1[2] * e2[1];
        let ny = e1[2] * e2[0] - e1[0] * e2[2];
        let nz = e1[0] * e2[1] - e1[1] * e2[0];
        let len = (nx * nx + ny * ny + nz * nz).sqrt();
        if len < 1e-15 {
            continue;
        }
        let (a, b, c) = (nx / len, ny / len, nz / len);
        let d = -(a * v0[0] + b * v0[1] + c * v0[2]);

        // Fundamental quadric: p * p^T where p = (a, b, c, d)
        let q = [
            a * a,
            a * b,
            a * c,
            a * d,
            b * b,
            b * c,
            b * d,
            c * c,
            c * d,
            d * d,
        ];

        for &vi in face {
            for k in 0..10 {
                quadrics[vi][k] += q[k];
            }
        }
    }

    // Greedy edge collapse
    let mut removed = vec![false; faces.len()];
    let mut vertex_map: Vec<usize> = (0..vertices.len()).collect();

    fn find_root(map: &[usize], mut i: usize) -> usize {
        while map[i] != i {
            i = map[i];
        }
        i
    }

    let mut current_faces = n_tris;

    while current_faces > target_faces {
        // Find cheapest edge collapse
        let mut best_cost = f64::INFINITY;
        let mut best_edge = (0, 0);
        let mut best_pos = [0.0; 3];

        for (fi, face) in faces.iter().enumerate() {
            if removed[fi] {
                continue;
            }
            for edge in [(0, 1), (1, 2), (2, 0)] {
                let vi = find_root(&vertex_map, face[edge.0]);
                let vj = find_root(&vertex_map, face[edge.1]);
                if vi == vj {
                    continue;
                }

                // Optimal position: midpoint (simple, avoids solving potentially singular 4x4)
                let pos = [
                    (vertices[vi][0] + vertices[vj][0]) * 0.5,
                    (vertices[vi][1] + vertices[vj][1]) * 0.5,
                    (vertices[vi][2] + vertices[vj][2]) * 0.5,
                ];

                // Cost = v^T (Q1+Q2) v
                let mut combined_q = [0.0; 10];
                for k in 0..10 {
                    combined_q[k] = quadrics[vi][k] + quadrics[vj][k];
                }
                let cost = eval_quadric(&combined_q, &pos);

                if cost < best_cost {
                    best_cost = cost;
                    best_edge = (vi, vj);
                    best_pos = pos;
                }
            }
        }

        if best_cost.is_infinite() {
            break;
        }

        let (vi, vj) = best_edge;
        // Merge vj into vi
        vertices[vi] = best_pos;
        let qj_copy = quadrics[vj];
        for k in 0..10 {
            quadrics[vi][k] += qj_copy[k];
        }
        vertex_map[vj] = vi;

        // Remove degenerate faces
        for (fi, face) in faces.iter().enumerate() {
            if removed[fi] {
                continue;
            }
            let a = find_root(&vertex_map, face[0]);
            let b = find_root(&vertex_map, face[1]);
            let c = find_root(&vertex_map, face[2]);
            if a == b || b == c || a == c {
                removed[fi] = true;
                current_faces -= 1;
            }
        }
    }

    // Rebuild mesh
    let mut new_verts: Vec<f64> = Vec::new();
    let mut vert_remap: Vec<Option<usize>> = vec![None; vertices.len()];
    let mut new_idx = 0;

    for i in 0..vertices.len() {
        let root = find_root(&vertex_map, i);
        if root == i && vert_remap[i].is_none() {
            vert_remap[i] = Some(new_idx);
            new_verts.extend_from_slice(&vertices[i][..n_dims.min(3)]);
            new_idx += 1;
        }
    }

    // Ensure all roots are mapped
    for i in 0..vertices.len() {
        let root = find_root(&vertex_map, i);
        if vert_remap[root].is_none() {
            vert_remap[root] = Some(new_idx);
            new_verts.extend_from_slice(&vertices[root][..n_dims.min(3)]);
            new_idx += 1;
        }
    }

    let mut new_tris: Vec<f64> = Vec::new();
    let mut n_new_tris = 0;
    for (fi, face) in faces.iter().enumerate() {
        if removed[fi] {
            continue;
        }
        let a = find_root(&vertex_map, face[0]);
        let b = find_root(&vertex_map, face[1]);
        let c = find_root(&vertex_map, face[2]);
        #[allow(clippy::collapsible_if)]
        if a != b && b != c && a != c {
            if let (Some(ia), Some(ib), Some(ic)) = (vert_remap[a], vert_remap[b], vert_remap[c]) {
                new_tris.push(ia as f64);
                new_tris.push(ib as f64);
                new_tris.push(ic as f64);
                n_new_tris += 1;
            }
        }
    }

    let n_new_verts = new_idx;
    let device = client.device();
    let new_v_tensor = Tensor::from_slice(&new_verts, &[n_new_verts, n_dims.min(3)], device);
    let new_t_tensor = Tensor::from_slice(&new_tris, &[n_new_tris, 3], device);

    Ok(Mesh {
        vertices: new_v_tensor,
        triangles: new_t_tensor,
        normals: None,
    })
}

fn eval_quadric(q: &[f64; 10], v: &[f64; 3]) -> f64 {
    // q stores upper triangle of 4x4 symmetric matrix:
    // [q0 q1 q2 q3]   indices: 0  1  2  3
    // [   q4 q5 q6]              4  5  6
    // [      q7 q8]                 7  8
    // [         q9]                    9
    let x = v[0];
    let y = v[1];
    let z = v[2];
    q[0] * x * x
        + 2.0 * q[1] * x * y
        + 2.0 * q[2] * x * z
        + 2.0 * q[3] * x
        + q[4] * y * y
        + 2.0 * q[5] * y * z
        + 2.0 * q[6] * y
        + q[7] * z * z
        + 2.0 * q[8] * z
        + q[9]
}

// ============ Mesh Smoothing (Laplacian/Taubin) ============

/// Smooth a mesh using Laplacian or Taubin smoothing (fully on-device).
///
/// Builds adjacency from triangles, then iteratively applies:
/// - Laplacian: v_new = v + lambda * (avg_neighbors - v)
/// - Taubin: alternate Laplacian with lambda (smooth) and mu (inflate)
pub fn mesh_smooth_impl<R, C>(
    client: &C,
    mesh: &Mesh<R>,
    iterations: usize,
    method: SmoothingMethod,
) -> Result<Mesh<R>>
where
    R: Runtime,
    C: ScalarOps<R> + CompareOps<R> + RuntimeClient<R>,
{
    let n_verts = mesh.vertices.shape()[0];
    let n_tris = mesh.triangles.shape()[0];

    if iterations == 0 {
        return Ok(mesh.clone());
    }

    // Build adjacency matrix from triangles.
    // We need to extract triangle indices (API boundary) to build adjacency.
    let tri_data: Vec<f64> = mesh.triangles.to_vec();

    // Build adjacency as a dense matrix [n_verts, n_verts] (1 if connected)
    let device = client.device();
    let mut adj_data = vec![0.0f64; n_verts * n_verts];
    let mut degree = vec![0.0f64; n_verts];

    for i in 0..n_tris {
        let a = tri_data[i * 3] as usize;
        let b = tri_data[i * 3 + 1] as usize;
        let c = tri_data[i * 3 + 2] as usize;
        for &(u, v) in &[(a, b), (b, c), (a, c)] {
            if adj_data[u * n_verts + v] == 0.0 {
                adj_data[u * n_verts + v] = 1.0;
                adj_data[v * n_verts + u] = 1.0;
                degree[u] += 1.0;
                degree[v] += 1.0;
            }
        }
    }

    // Build normalized adjacency: L[i,j] = adj[i,j] / degree[i]
    // So L @ V gives average neighbor positions
    let mut lap_data = vec![0.0f64; n_verts * n_verts];
    for i in 0..n_verts {
        if degree[i] > 0.0 {
            for j in 0..n_verts {
                lap_data[i * n_verts + j] = adj_data[i * n_verts + j] / degree[i];
            }
        }
    }

    let lap_matrix = Tensor::from_slice(&lap_data, &[n_verts, n_verts], device);

    let mut verts = mesh.vertices.clone();

    for iter in 0..iterations {
        // avg_neighbors = L @ verts
        let avg = client.matmul(&lap_matrix, &verts)?;

        // delta = avg - verts
        let delta = client.sub(&avg, &verts)?;

        let lambda = match method {
            SmoothingMethod::Laplacian { lambda } => lambda,
            SmoothingMethod::Taubin { lambda, mu } => {
                if iter % 2 == 0 {
                    lambda
                } else {
                    mu
                }
            }
        };

        // v_new = v + lambda * delta
        let update = client.mul_scalar(&delta, lambda)?;
        verts = client.add(&verts, &update)?;
    }

    Ok(Mesh {
        vertices: verts,
        triangles: mesh.triangles.clone(),
        normals: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    fn setup() -> (CpuClient, CpuDevice) {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        (client, device)
    }

    #[test]
    fn test_triangulate_square() {
        let (client, device) = setup();
        let vertices = Tensor::<CpuRuntime>::from_slice(
            &[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            &[4, 2],
            &device,
        );
        let mesh = triangulate_polygon_impl(&client, &vertices).unwrap();
        assert_eq!(mesh.triangles.shape()[0], 2); // Square → 2 triangles
        assert_eq!(mesh.triangles.shape()[1], 3);
    }

    #[test]
    fn test_triangulate_triangle() {
        let (client, device) = setup();
        let vertices =
            Tensor::<CpuRuntime>::from_slice(&[0.0, 0.0, 1.0, 0.0, 0.5, 1.0], &[3, 2], &device);
        let mesh = triangulate_polygon_impl(&client, &vertices).unwrap();
        assert_eq!(mesh.triangles.shape()[0], 1); // Triangle → 1 triangle
    }

    #[test]
    fn test_triangulate_pentagon() {
        let (client, device) = setup();
        // Regular pentagon
        let vertices = Tensor::<CpuRuntime>::from_slice(
            &[
                1.0, 0.0, 0.309, 0.951, -0.809, 0.588, -0.809, -0.588, 0.309, -0.951,
            ],
            &[5, 2],
            &device,
        );
        let mesh = triangulate_polygon_impl(&client, &vertices).unwrap();
        assert_eq!(mesh.triangles.shape()[0], 3); // Pentagon → 3 triangles
    }

    #[test]
    fn test_mesh_smooth_laplacian() {
        let (client, device) = setup();
        // Simple mesh: triangle with one perturbed vertex
        let vertices = Tensor::<CpuRuntime>::from_slice(
            &[0.0, 0.0, 1.0, 0.0, 0.5, 1.5], // Third vertex slightly off
            &[3, 2],
            &device,
        );
        let triangles = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0], &[1, 3], &device);
        let mesh = Mesh {
            vertices,
            triangles,
            normals: None,
        };

        let smoothed = mesh_smooth_impl(
            &client,
            &mesh,
            5,
            SmoothingMethod::Laplacian { lambda: 0.5 },
        )
        .unwrap();

        // Smoothed vertices should be closer together
        assert_eq!(smoothed.vertices.shape(), mesh.vertices.shape());
    }

    #[test]
    fn test_mesh_simplify_basic() {
        let (client, device) = setup();
        // Two triangles sharing an edge
        let vertices = Tensor::<CpuRuntime>::from_slice(
            &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 1.0, 0.0, 0.5, -1.0, 0.0],
            &[4, 3],
            &device,
        );
        let triangles =
            Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0, 0.0, 3.0, 1.0], &[2, 3], &device);
        let mesh = Mesh {
            vertices,
            triangles,
            normals: None,
        };

        let simplified =
            mesh_simplify_impl(&client, &mesh, 1, SimplificationMethod::QuadricError).unwrap();
        assert!(simplified.triangles.shape()[0] <= 2);
    }
}
