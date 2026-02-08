//! Generic spherical Voronoi diagram implementation.
//!
//! Algorithm: 3D convex hull → dual → circumcenters projected to sphere.

use crate::spatial::impl_generic::convex_hull::convex_hull_impl;
use crate::spatial::traits::spherical_voronoi::SphericalVoronoi;
use crate::spatial::{validate_points_2d, validate_points_dtype};
use numr::error::{Error, Result};
use numr::ops::{
    CompareOps, IndexingOps, ReduceOps, ScalarOps, SortingOps, TensorOps, TypeConversionOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use std::collections::HashMap;

/// Compute spherical Voronoi diagram.
pub fn spherical_voronoi_impl<R, C>(
    client: &C,
    points: &Tensor<R>,
    radius: f64,
    center: Option<&Tensor<R>>,
) -> Result<SphericalVoronoi<R>>
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
    validate_points_dtype(points.dtype(), "spherical_voronoi")?;
    validate_points_2d(points.shape(), "spherical_voronoi")?;

    let n = points.shape()[0];
    let d = points.shape()[1];
    let device = points.device();
    let dtype = points.dtype();

    if d != 3 {
        return Err(Error::InvalidArgument {
            arg: "points",
            reason: format!("Spherical Voronoi requires 3D points, got {}D", d),
        });
    }
    if n < 3 {
        return Err(Error::InvalidArgument {
            arg: "points",
            reason: format!("Need at least 3 points, got {}", n),
        });
    }
    if radius <= 0.0 {
        return Err(Error::InvalidArgument {
            arg: "radius",
            reason: "Radius must be positive".to_string(),
        });
    }

    let center_tensor = match center {
        Some(c) => c.clone(),
        None => Tensor::<R>::zeros(&[3], dtype, device),
    };

    // Compute convex hull of the points (they lie on a sphere, hull = all points)
    let hull = convex_hull_impl(client, points)?;

    // Hull simplices are triangular faces [n_faces, 3] with vertex indices
    let simplices_data: Vec<i64> = hull.simplices.to_vec();
    let n_faces = hull.simplices.shape()[0];

    // Get all point coordinates (API boundary transfer)
    let points_data: Vec<f64> = points.to_vec();
    let center_data: Vec<f64> = center_tensor.to_vec();

    // For each face, compute circumcenter projected onto sphere
    let mut vertices: Vec<f64> = Vec::with_capacity(n_faces * 3);
    // Map: point_index -> list of voronoi vertex indices
    let mut point_to_verts: HashMap<i64, Vec<usize>> = HashMap::new();

    for f in 0..n_faces {
        let i0 = simplices_data[f * 3] as usize;
        let i1 = simplices_data[f * 3 + 1] as usize;
        let i2 = simplices_data[f * 3 + 2] as usize;

        let (vx, vy, vz) = circumcenter_3d(&points_data, i0, i1, i2, &center_data, radius);

        let vert_idx = f;
        vertices.push(vx);
        vertices.push(vy);
        vertices.push(vz);

        // Each face contributes this vertex to the regions of its 3 generator points
        for &pi in &[
            simplices_data[f * 3],
            simplices_data[f * 3 + 1],
            simplices_data[f * 3 + 2],
        ] {
            point_to_verts.entry(pi).or_default().push(vert_idx);
        }
    }

    // Build CSR regions
    let mut regions_indices: Vec<i64> = Vec::new();
    let mut regions_indptr: Vec<i64> = vec![0];

    for i in 0..n {
        if let Some(verts) = point_to_verts.get(&(i as i64)) {
            for &v in verts {
                regions_indices.push(v as i64);
            }
        }
        regions_indptr.push(regions_indices.len() as i64);
    }

    let regions_indices_final = if regions_indices.is_empty() {
        vec![-1i64]
    } else {
        regions_indices
    };

    Ok(SphericalVoronoi {
        points: points.clone(),
        center: center_tensor,
        radius,
        vertices: Tensor::<R>::from_slice(&vertices, &[n_faces, 3], device),
        regions_indices: Tensor::<R>::from_slice(
            &regions_indices_final,
            &[regions_indices_final.len()],
            device,
        ),
        regions_indptr: Tensor::<R>::from_slice(&regions_indptr, &[regions_indptr.len()], device),
    })
}

/// Sort region vertices in counter-clockwise order.
///
/// For each region, sorts vertices by angle around the generator point
/// when projected onto the tangent plane.
pub fn spherical_voronoi_sort_regions_impl<R, C>(
    _client: &C,
    sv: &SphericalVoronoi<R>,
) -> Result<SphericalVoronoi<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
    let n = sv.points.shape()[0];
    let device = sv.points.device();

    let points_data: Vec<f64> = sv.points.to_vec();
    let vertices_data: Vec<f64> = sv.vertices.to_vec();
    let indices_data: Vec<i64> = sv.regions_indices.to_vec();
    let indptr_data: Vec<i64> = sv.regions_indptr.to_vec();
    let center_data: Vec<f64> = sv.center.to_vec();

    let mut new_indices: Vec<i64> = Vec::with_capacity(indices_data.len());
    let mut new_indptr: Vec<i64> = vec![0];

    for i in 0..n {
        let start = indptr_data[i] as usize;
        let end = indptr_data[i + 1] as usize;

        if start >= end || (end - start == 1 && indices_data[start] == -1) {
            new_indptr.push(new_indices.len() as i64);
            continue;
        }

        let region_verts: Vec<i64> = indices_data[start..end].to_vec();

        // Generator point (shifted to origin-centered)
        let gx = points_data[i * 3] - center_data[0];
        let gy = points_data[i * 3 + 1] - center_data[1];
        let gz = points_data[i * 3 + 2] - center_data[2];
        let glen = (gx * gx + gy * gy + gz * gz).sqrt();
        let (gnx, gny, gnz) = (gx / glen, gy / glen, gz / glen);

        // Build orthonormal basis on tangent plane
        // Pick a reference vector not parallel to generator normal
        let (refx, refy, refz) = if gnx.abs() < 0.9 {
            (1.0, 0.0, 0.0)
        } else {
            (0.0, 1.0, 0.0)
        };
        // e1 = ref - (ref·gn)*gn, normalized
        let dot = refx * gnx + refy * gny + refz * gnz;
        let e1x = refx - dot * gnx;
        let e1y = refy - dot * gny;
        let e1z = refz - dot * gnz;
        let e1len = (e1x * e1x + e1y * e1y + e1z * e1z).sqrt();
        let (e1x, e1y, e1z) = (e1x / e1len, e1y / e1len, e1z / e1len);

        // e2 = gn × e1
        let e2x = gny * e1z - gnz * e1y;
        let e2y = gnz * e1x - gnx * e1z;
        let e2z = gnx * e1y - gny * e1x;

        // Compute angles for each vertex
        let mut vert_angles: Vec<(i64, f64)> = region_verts
            .iter()
            .map(|&vi| {
                let vi_usize = vi as usize;
                let vx = vertices_data[vi_usize * 3] - center_data[0];
                let vy = vertices_data[vi_usize * 3 + 1] - center_data[1];
                let vz = vertices_data[vi_usize * 3 + 2] - center_data[2];
                // Project onto tangent plane
                let proj1 = vx * e1x + vy * e1y + vz * e1z;
                let proj2 = vx * e2x + vy * e2y + vz * e2z;
                let angle = proj2.atan2(proj1);
                (vi, angle)
            })
            .collect();

        // NaN angles are treated as greater (pushed to end)
        vert_angles.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Greater));

        for (vi, _) in &vert_angles {
            new_indices.push(*vi);
        }
        new_indptr.push(new_indices.len() as i64);
    }

    let new_indices_final = if new_indices.is_empty() {
        vec![-1i64]
    } else {
        new_indices
    };

    Ok(SphericalVoronoi {
        points: sv.points.clone(),
        center: sv.center.clone(),
        radius: sv.radius,
        vertices: sv.vertices.clone(),
        regions_indices: Tensor::<R>::from_slice(
            &new_indices_final,
            &[new_indices_final.len()],
            device,
        ),
        regions_indptr: Tensor::<R>::from_slice(&new_indptr, &[new_indptr.len()], device),
    })
}

/// Compute region areas on the sphere using spherical excess formula.
pub fn spherical_voronoi_region_areas_impl<R, C>(
    _client: &C,
    sv: &SphericalVoronoi<R>,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
    let n = sv.points.shape()[0];
    let device = sv.points.device();

    let vertices_data: Vec<f64> = sv.vertices.to_vec();
    let indices_data: Vec<i64> = sv.regions_indices.to_vec();
    let indptr_data: Vec<i64> = sv.regions_indptr.to_vec();
    let center_data: Vec<f64> = sv.center.to_vec();
    let r = sv.radius;

    let mut areas: Vec<f64> = Vec::with_capacity(n);

    for i in 0..n {
        let start = indptr_data[i] as usize;
        let end = indptr_data[i + 1] as usize;

        if start >= end || (end - start == 1 && indices_data[start] == -1) {
            areas.push(0.0);
            continue;
        }

        let region_verts: Vec<usize> = indices_data[start..end]
            .iter()
            .map(|&v| v as usize)
            .collect();
        let nv = region_verts.len();

        if nv < 3 {
            areas.push(0.0);
            continue;
        }

        // Spherical polygon area via spherical excess:
        // For a spherical polygon with vertices v0..v_{n-1}, decompose into
        // spherical triangles from v0, and sum their spherical excesses.
        let mut total_excess = 0.0;
        let v0 = &region_verts[0];

        for j in 1..nv - 1 {
            let v1 = &region_verts[j];
            let v2 = &region_verts[j + 1];

            // Get unit vectors from center to each vertex
            let a = unit_from_center(&vertices_data, *v0, &center_data);
            let b = unit_from_center(&vertices_data, *v1, &center_data);
            let c = unit_from_center(&vertices_data, *v2, &center_data);

            // Spherical excess = sum of angles - π
            let angle_a = spherical_angle(a, b, c);
            let angle_b = spherical_angle(b, c, a);
            let angle_c = spherical_angle(c, a, b);

            let excess = angle_a + angle_b + angle_c - std::f64::consts::PI;
            total_excess += excess.abs();
        }

        areas.push(total_excess * r * r);
    }

    Ok(Tensor::<R>::from_slice(&areas, &[n], device))
}

/// Compute circumcenter of a triangle in 3D, projected onto sphere.
fn circumcenter_3d(
    points: &[f64],
    i0: usize,
    i1: usize,
    i2: usize,
    center: &[f64],
    radius: f64,
) -> (f64, f64, f64) {
    let ax = points[i0 * 3] - center[0];
    let ay = points[i0 * 3 + 1] - center[1];
    let az = points[i0 * 3 + 2] - center[2];
    let bx = points[i1 * 3] - center[0];
    let by = points[i1 * 3 + 1] - center[1];
    let bz = points[i1 * 3 + 2] - center[2];
    let cx = points[i2 * 3] - center[0];
    let cy = points[i2 * 3 + 1] - center[1];
    let cz = points[i2 * 3 + 2] - center[2];

    // Circumcenter of triangle in 3D: find point equidistant from all three
    // Using the formula: circumcenter = a + s*(b-a) + t*(c-a)
    // where s,t solve the system from |p-a|²=|p-b|²=|p-c|²

    let abx = bx - ax;
    let aby = by - ay;
    let abz = bz - az;
    let acx = cx - ax;
    let acy = cy - ay;
    let acz = cz - az;

    let ab_dot_ab = abx * abx + aby * aby + abz * abz;
    let ab_dot_ac = abx * acx + aby * acy + abz * acz;
    let ac_dot_ac = acx * acx + acy * acy + acz * acz;

    let det = ab_dot_ab * ac_dot_ac - ab_dot_ac * ab_dot_ac;

    if det.abs() < 1e-30 {
        // Degenerate: return centroid projected to sphere
        let mx = (ax + bx + cx) / 3.0;
        let my = (ay + by + cy) / 3.0;
        let mz = (az + bz + cz) / 3.0;
        let len = (mx * mx + my * my + mz * mz).sqrt();
        return (
            mx / len * radius + center[0],
            my / len * radius + center[1],
            mz / len * radius + center[2],
        );
    }

    let s = (ac_dot_ac * ab_dot_ab - ab_dot_ac * ac_dot_ac) / (2.0 * det);
    let t = (ab_dot_ab * ac_dot_ac - ab_dot_ac * ab_dot_ab) / (2.0 * det);

    let px = ax + s * abx + t * acx;
    let py = ay + s * aby + t * acy;
    let pz = az + s * abz + t * acz;

    // Project onto sphere
    let len = (px * px + py * py + pz * pz).sqrt();
    (
        px / len * radius + center[0],
        py / len * radius + center[1],
        pz / len * radius + center[2],
    )
}

/// Get unit vector from center to vertex.
fn unit_from_center(vertices: &[f64], vi: usize, center: &[f64]) -> (f64, f64, f64) {
    let x = vertices[vi * 3] - center[0];
    let y = vertices[vi * 3 + 1] - center[1];
    let z = vertices[vi * 3 + 2] - center[2];
    let len = (x * x + y * y + z * z).sqrt();
    if len < 1e-30 {
        (0.0, 0.0, 1.0)
    } else {
        (x / len, y / len, z / len)
    }
}

/// Compute the spherical angle at vertex A in the spherical triangle ABC.
/// Each point is a unit vector (x, y, z).
fn spherical_angle(a: (f64, f64, f64), b: (f64, f64, f64), c: (f64, f64, f64)) -> f64 {
    // Project AB and AC onto the tangent plane at A
    let ab_dot_a = b.0 * a.0 + b.1 * a.1 + b.2 * a.2;
    let ac_dot_a = c.0 * a.0 + c.1 * a.1 + c.2 * a.2;

    let tab = (
        b.0 - ab_dot_a * a.0,
        b.1 - ab_dot_a * a.1,
        b.2 - ab_dot_a * a.2,
    );
    let tac = (
        c.0 - ac_dot_a * a.0,
        c.1 - ac_dot_a * a.1,
        c.2 - ac_dot_a * a.2,
    );

    let tab_len = (tab.0 * tab.0 + tab.1 * tab.1 + tab.2 * tab.2).sqrt();
    let tac_len = (tac.0 * tac.0 + tac.1 * tac.1 + tac.2 * tac.2).sqrt();

    if tab_len < 1e-30 || tac_len < 1e-30 {
        return 0.0;
    }

    let cos_angle = (tab.0 * tac.0 + tab.1 * tac.1 + tab.2 * tac.2) / (tab_len * tac_len);
    cos_angle.clamp(-1.0, 1.0).acos()
}
