//! Ordered Schur decomposition helper.
//!
//! Reorders a real Schur form so that eigenvalues satisfying a selection
//! criterion appear in the top-left block. Uses orthogonal similarity
//! transformations (Givens rotations / 2×2 block swaps).
//!
//! # Performance note
//!
//! This algorithm extracts the Schur factors to CPU for sequential
//! reordering (bubble-sort of diagonal blocks with Givens rotations).
//! Each swap depends on the previous one, so GPU parallelism cannot help.
//! Prefer `CpuRuntime` for matrix equation solvers — GPU transfers add
//! overhead with no computational benefit at typical problem sizes.
use crate::DType;

use numr::algorithm::linalg::LinearAlgebraAlgorithms;
use numr::error::{Error, Result};
use numr::ops::{MatmulOps, ScalarOps, ShapeOps, TensorOps, UtilityOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Selection criterion for eigenvalue reordering.
#[derive(Debug, Clone, Copy)]
pub enum EigenvalueSelector {
    /// Select eigenvalues with negative real part (for continuous-time stability).
    LeftHalfPlane,
    /// Select eigenvalues inside the unit circle (for discrete-time stability).
    InsideUnitCircle,
}

/// Result of ordered Schur decomposition.
pub struct OrderedSchur<R: Runtime<DType = DType>> {
    /// Orthogonal matrix Z (reordered).
    pub z: Tensor<R>,
    /// Upper quasi-triangular T (reordered).
    pub t: Tensor<R>,
    /// Number of selected eigenvalues in top-left block.
    pub num_selected: usize,
}

/// Reorder a real Schur decomposition so that selected eigenvalues
/// appear in the top-left block.
///
/// Given T = Z^T A Z (Schur form), computes orthogonal Q such that
/// T' = Q^T T Q has selected eigenvalues in positions 0..num_selected.
/// Returns updated Z' = Z Q and T'.
pub fn ordschur_impl<R, C>(
    _client: &C,
    z: &Tensor<R>,
    t: &Tensor<R>,
    selector: EigenvalueSelector,
) -> Result<OrderedSchur<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + MatmulOps<R>
        + UtilityOps<R>
        + LinearAlgebraAlgorithms<R>
        + RuntimeClient<R>,
{
    let shape = t.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(Error::InvalidArgument {
            arg: "t",
            reason: "ordschur requires square matrix".into(),
        });
    }
    let n = shape[0];
    if n == 0 {
        return Ok(OrderedSchur {
            z: z.clone(),
            t: t.clone(),
            num_selected: 0,
        });
    }

    // Extract T to CPU for the sequential reordering.
    // This is acceptable: ordschur is O(n^3) sequential Givens rotations
    // on small matrices (typically 2n×2n for ARE, n≤100 in practice).
    let mut t_data = t.to_vec::<f64>();
    let mut z_data = z.to_vec::<f64>();

    // Identify block structure and eigenvalues of the quasi-triangular T.
    // Diagonal: 1×1 blocks (real eigenvalues) or 2×2 blocks (complex conjugate pairs).
    let mut block_sizes = Vec::new();
    let mut i = 0;
    while i < n {
        if i + 1 < n && t_data[(i + 1) * n + i].abs() > 1e-10 {
            block_sizes.push(2);
            i += 2;
        } else {
            block_sizes.push(1);
            i += 1;
        }
    }

    // For each block, determine if its eigenvalue(s) are "selected".
    let is_selected = |t_data: &[f64], pos: usize, size: usize| -> bool {
        if size == 1 {
            let lambda = t_data[pos * n + pos];
            match selector {
                EigenvalueSelector::LeftHalfPlane => lambda < 0.0,
                EigenvalueSelector::InsideUnitCircle => lambda.abs() < 1.0,
            }
        } else {
            // 2×2 block: eigenvalues are a ± bi
            let a11 = t_data[pos * n + pos];
            let a12 = t_data[pos * n + pos + 1];
            let a21 = t_data[(pos + 1) * n + pos];
            let a22 = t_data[(pos + 1) * n + pos + 1];
            let real_part = (a11 + a22) / 2.0;
            let det = a11 * a22 - a12 * a21;
            match selector {
                EigenvalueSelector::LeftHalfPlane => real_part < 0.0,
                EigenvalueSelector::InsideUnitCircle => det.abs() < 1.0,
            }
        }
    };

    // Bubble sort: move selected blocks to top-left using orthogonal swaps.
    // Build list of (position, block_size, selected).
    let mut blocks: Vec<(usize, usize, bool)> = Vec::new();
    let mut pos = 0;
    for &sz in &block_sizes {
        let sel = is_selected(&t_data, pos, sz);
        blocks.push((pos, sz, sel));
        pos += sz;
    }

    // Repeatedly scan and swap adjacent blocks if an unselected block
    // precedes a selected block.
    let mut swapped = true;
    while swapped {
        swapped = false;
        let mut idx = 0;
        while idx + 1 < blocks.len() {
            if !blocks[idx].2 && blocks[idx + 1].2 {
                // Swap block[idx] (unselected) with block[idx+1] (selected).
                let p = blocks[idx].0;
                let s1 = blocks[idx].1;
                let s2 = blocks[idx + 1].1;
                swap_schur_blocks(&mut t_data, &mut z_data, n, p, s1, s2);
                // Update positions after swap.
                blocks[idx] = (p, s2, true);
                blocks[idx + 1] = (p + s2, s1, false);
                swapped = true;
            }
            idx += 1;
        }
    }

    let num_selected_size: usize = blocks.iter().filter(|b| b.2).map(|b| b.1).sum();

    let device = t.device();
    let t_out = Tensor::from_slice(&t_data, &[n, n], device);
    let z_out = Tensor::from_slice(&z_data, &[n, n], device);

    Ok(OrderedSchur {
        z: z_out,
        t: t_out,
        num_selected: num_selected_size,
    })
}

/// Swap two adjacent diagonal blocks in the Schur form.
///
/// Block 1 starts at position `p` with size `s1`.
/// Block 2 starts at position `p + s1` with size `s2`.
///
/// Uses the direct Givens/Householder method for 1×1/1×1, 1×1/2×2, 2×2/1×1, 2×2/2×2 swaps.
fn swap_schur_blocks(t: &mut [f64], z: &mut [f64], n: usize, p: usize, s1: usize, s2: usize) {
    match (s1, s2) {
        (1, 1) => swap_1x1_1x1(t, z, n, p),
        (1, 2) => swap_1x2(t, z, n, p),
        (2, 1) => swap_2x1(t, z, n, p),
        (2, 2) => swap_2x2(t, z, n, p),
        _ => unreachable!(),
    }
}

/// Swap two 1×1 blocks at position p and p+1.
fn swap_1x1_1x1(t: &mut [f64], z: &mut [f64], n: usize, p: usize) {
    let a = t[p * n + p];
    let b = t[p * n + p + 1];
    let d = t[(p + 1) * n + p + 1];

    // Solve the Sylvester equation: (a - d) * x = -b  =>  x = -b / (a - d)
    let diff = a - d;
    if diff.abs() < 1e-15 {
        return; // eigenvalues are equal, no swap needed
    }

    let tau = b / diff;
    let denom = (1.0 + tau * tau).sqrt();
    let cs = 1.0 / denom;
    let sn = tau / denom;

    // Apply Givens rotation: T <- G^T T G
    apply_givens_left(t, n, p, p + 1, cs, sn);
    apply_givens_right(t, n, p, p + 1, cs, sn);
    t[(p + 1) * n + p] = 0.0; // enforce exact zero

    // Update Z: Z <- Z G
    apply_givens_right(z, n, p, p + 1, cs, sn);
}

/// Swap 1×1 block at p with 2×2 block at p+1.
fn swap_1x2(t: &mut [f64], z: &mut [f64], n: usize, p: usize) {
    // Use Householder to move 1×1 block past 2×2 block.
    // Solve Sylvester: a11 * [x1, x2] - [x1, x2] * B22 = -[t12, t13]
    // where a11 = t[p,p], B22 = t[p+1..p+3, p+1..p+3]
    let a = t[p * n + p];
    let b11 = t[(p + 1) * n + p + 1];
    let b12 = t[(p + 1) * n + p + 2];
    let b21 = t[(p + 2) * n + p + 1];
    let b22 = t[(p + 2) * n + p + 2];

    // Solve 2×2 system: (aI - B^T) x = -[t(p,p+1), t(p,p+2)]^T
    let rhs1 = -t[p * n + p + 1];
    let rhs2 = -t[p * n + p + 2];

    let m11 = a - b11;
    let m12 = -b21;
    let m21 = -b12;
    let m22 = a - b22;

    let det = m11 * m22 - m12 * m21;
    if det.abs() < 1e-15 {
        return;
    }

    let x1 = (m22 * rhs1 - m12 * rhs2) / det;
    let x2 = (-m21 * rhs1 + m11 * rhs2) / det;

    // Build Householder reflector from [1, x1, x2]
    let norm = (1.0 + x1 * x1 + x2 * x2).sqrt();
    let v = [1.0 / norm, x1 / norm, x2 / norm];

    apply_householder_both(t, z, n, p, &v, 3);

    // Clean up: ensure proper quasi-triangular structure
    t[(p + 1) * n + p] = 0.0;
    t[(p + 2) * n + p] = 0.0;
    // The 2×2 block is now at top, 1×1 at bottom
    // Check if sub-diagonal of the 2×2 block is zero (split into two 1×1)
    if t[(p + 1) * n + p].abs() < 1e-14 {
        t[(p + 1) * n + p] = 0.0;
    }
}

/// Swap 2×2 block at p with 1×1 block at p+2.
fn swap_2x1(t: &mut [f64], z: &mut [f64], n: usize, p: usize) {
    // Solve Sylvester: B11 * x - x * a33 = -[t(p,p+2), t(p+1,p+2)]
    let b11 = t[p * n + p];
    let b12 = t[p * n + p + 1];
    let b21 = t[(p + 1) * n + p];
    let b22 = t[(p + 1) * n + p + 1];
    let a = t[(p + 2) * n + p + 2];

    let rhs1 = -t[p * n + p + 2];
    let rhs2 = -t[(p + 1) * n + p + 2];

    let m11 = b11 - a;
    let m12 = b12;
    let m21 = b21;
    let m22 = b22 - a;

    let det = m11 * m22 - m12 * m21;
    if det.abs() < 1e-15 {
        return;
    }

    let x1 = (m22 * rhs1 - m12 * rhs2) / det;
    let x2 = (-m21 * rhs1 + m11 * rhs2) / det;

    // Build Householder reflector from [x1, x2, 1]
    let norm = (x1 * x1 + x2 * x2 + 1.0).sqrt();
    let v = [x1 / norm, x2 / norm, 1.0 / norm];

    apply_householder_both(t, z, n, p, &v, 3);

    // Clean up
    t[(p + 2) * n + p] = 0.0;
    t[(p + 2) * n + p + 1] = 0.0;
}

/// Swap two 2×2 blocks at positions p and p+2.
#[allow(clippy::needless_range_loop)]
fn swap_2x2(t: &mut [f64], z: &mut [f64], n: usize, p: usize) {
    // Solve the Sylvester equation: T11 X - X T22 = -T12
    // where T11 = t[p..p+2, p..p+2], T22 = t[p+2..p+4, p+2..p+4], T12 = t[p..p+2, p+2..p+4]
    let a00 = t[p * n + p];
    let a01 = t[p * n + p + 1];
    let a10 = t[(p + 1) * n + p];
    let a11 = t[(p + 1) * n + p + 1];
    let b00 = t[(p + 2) * n + p + 2];
    let b01 = t[(p + 2) * n + p + 3];
    let b10 = t[(p + 3) * n + p + 2];
    let b11 = t[(p + 3) * n + p + 3];
    let c00 = t[p * n + p + 2];
    let c01 = t[p * n + p + 3];
    let c10 = t[(p + 1) * n + p + 2];
    let c11 = t[(p + 1) * n + p + 3];

    // Kronecker system for T11 X - X T22 = -T12
    // unknowns [x00, x01, x10, x11] (row-major vec(X))
    // Equations derived from element-wise expansion:
    let mut aug = [
        [(a00 - b00), -b10, a01, 0.0, -c00],
        [-b01, (a00 - b11), 0.0, a01, -c01],
        [a10, 0.0, (a11 - b00), -b10, -c10],
        [0.0, a10, -b01, (a11 - b11), -c11],
    ];

    // Gaussian elimination with partial pivoting
    for col in 0..4 {
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for row in col + 1..4 {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }
        if max_val < 1e-15 {
            return;
        }
        if max_row != col {
            aug.swap(col, max_row);
        }
        let pivot = aug[col][col];
        for row in col + 1..4 {
            let factor = aug[row][col] / pivot;
            for j in col..5 {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }

    let mut xv = [0.0f64; 4];
    for i in (0..4).rev() {
        let mut s = aug[i][4];
        for j in i + 1..4 {
            s -= aug[i][j] * xv[j];
        }
        xv[i] = s / aug[i][i];
    }

    // X = [[xv[0], xv[1]], [xv[2], xv[3]]] (row-major)
    // Build 4×4 orthogonal Q from QR of [[X]; [I_2]] (4×2 matrix).
    // Then apply T' = Q^T T Q, Z' = Z Q.
    // Compute Q explicitly as a 4×4 matrix using Householder QR.
    let mut mat = [[xv[0], xv[1]], [xv[2], xv[3]], [1.0, 0.0], [0.0, 1.0]];

    // Q = H1 * H2 where Hi are Householder reflectors
    // Store reflectors then build full Q
    let mut reflectors: Vec<(usize, Vec<f64>, f64)> = Vec::new();

    for col in 0..2 {
        let mut norm_sq = 0.0;
        for row in col..4 {
            norm_sq += mat[row][col] * mat[row][col];
        }
        let norm_val = norm_sq.sqrt();
        if norm_val < 1e-15 {
            continue;
        }
        let sign = if mat[col][col] >= 0.0 { 1.0 } else { -1.0 };
        let alpha = -sign * norm_val;
        let mut v = vec![0.0f64; 4];
        v[col] = mat[col][col] - alpha;
        for row in col + 1..4 {
            v[row] = mat[row][col];
        }
        let v_norm_sq: f64 = v[col..].iter().map(|&vi| vi * vi).sum();
        if v_norm_sq < 1e-30 {
            continue;
        }
        let beta = 2.0 / v_norm_sq;

        // Apply to mat for subsequent columns
        for j in col..2 {
            let dot: f64 = (col..4).map(|r| v[r] * mat[r][j]).sum();
            for r in col..4 {
                mat[r][j] -= beta * v[r] * dot;
            }
        }

        reflectors.push((col, v, beta));
    }

    // Build explicit 4×4 Q = H1 * H2 by applying reflectors to I_4
    let mut q = [[0.0f64; 4]; 4];
    for i in 0..4 {
        q[i][i] = 1.0;
    }
    for &(col, ref v, beta) in &reflectors {
        // Q <- Q * H = Q * (I - beta * v * v^T)
        // For each row of Q: row <- row - beta * (row · v) * v^T
        for row in 0..4 {
            let dot: f64 = (col..4).map(|k| q[row][k] * v[k]).sum();
            for k in col..4 {
                q[row][k] -= beta * dot * v[k];
            }
        }
    }

    // Apply Q^T T Q to the full n×n matrix T (rows/cols p..p+4 affected)
    // T <- Q^T T Q
    // Step 1: T <- Q^T T (apply Q^T from left to rows p..p+3)
    for col in 0..n {
        let mut vals = [0.0f64; 4];
        for i in 0..4 {
            vals[i] = t[(p + i) * n + col];
        }
        for i in 0..4 {
            let mut s = 0.0;
            for k in 0..4 {
                s += q[k][i] * vals[k]; // Q^T[i][k] = Q[k][i]
            }
            t[(p + i) * n + col] = s;
        }
    }
    // Step 2: T <- T Q (apply Q from right to cols p..p+3)
    for row in 0..n {
        let mut vals = [0.0f64; 4];
        for j in 0..4 {
            vals[j] = t[row * n + p + j];
        }
        for j in 0..4 {
            let mut s = 0.0;
            for k in 0..4 {
                s += vals[k] * q[k][j];
            }
            t[row * n + p + j] = s;
        }
    }

    // Apply Z <- Z Q (right multiply)
    for row in 0..n {
        let mut vals = [0.0f64; 4];
        for j in 0..4 {
            vals[j] = z[row * n + p + j];
        }
        for j in 0..4 {
            let mut s = 0.0;
            for k in 0..4 {
                s += vals[k] * q[k][j];
            }
            z[row * n + p + j] = s;
        }
    }

    // Clean up sub-diagonal entries below the new block structure
    t[(p + 2) * n + p] = 0.0;
    t[(p + 2) * n + p + 1] = 0.0;
    t[(p + 3) * n + p] = 0.0;
    t[(p + 3) * n + p + 1] = 0.0;
}

/// Apply Givens rotation G(i,j,cs,sn) from the left: T <- G^T * T
fn apply_givens_left(t: &mut [f64], n: usize, i: usize, j: usize, cs: f64, sn: f64) {
    for col in 0..n {
        let ti = t[i * n + col];
        let tj = t[j * n + col];
        t[i * n + col] = cs * ti + sn * tj;
        t[j * n + col] = -sn * ti + cs * tj;
    }
}

/// Apply Givens rotation G(i,j,cs,sn) from the right: T <- T * G
fn apply_givens_right(t: &mut [f64], n: usize, i: usize, j: usize, cs: f64, sn: f64) {
    for row in 0..n {
        let ti = t[row * n + i];
        let tj = t[row * n + j];
        t[row * n + i] = cs * ti + sn * tj;
        t[row * n + j] = -sn * ti + cs * tj;
    }
}

/// Apply 3×3 Householder reflector from both sides at position p.
fn apply_householder_both(
    t: &mut [f64],
    z: &mut [f64],
    n: usize,
    p: usize,
    v: &[f64; 3],
    _size: usize,
) {
    let beta = 2.0;
    // Apply H = I - beta * v * v^T from left: T <- H T
    for col in 0..n {
        let dot = v[0] * t[p * n + col] + v[1] * t[(p + 1) * n + col] + v[2] * t[(p + 2) * n + col];
        t[p * n + col] -= beta * v[0] * dot;
        t[(p + 1) * n + col] -= beta * v[1] * dot;
        t[(p + 2) * n + col] -= beta * v[2] * dot;
    }
    // Apply from right: T <- T H
    for row in 0..n {
        let dot = v[0] * t[row * n + p] + v[1] * t[row * n + p + 1] + v[2] * t[row * n + p + 2];
        t[row * n + p] -= beta * v[0] * dot;
        t[row * n + p + 1] -= beta * v[1] * dot;
        t[row * n + p + 2] -= beta * v[2] * dot;
    }
    // Apply to Z from right: Z <- Z H
    for row in 0..n {
        let dot = v[0] * z[row * n + p] + v[1] * z[row * n + p + 1] + v[2] * z[row * n + p + 2];
        z[row * n + p] -= beta * v[0] * dot;
        z[row * n + p + 1] -= beta * v[1] * dot;
        z[row * n + p + 2] -= beta * v[2] * dot;
    }
}

/// Reorder a generalized Schur (QZ) decomposition so that selected
/// generalized eigenvalues appear in the top-left block.
///
/// The generalized eigenvalues are α_i / β_i where α = diag(S) and β = diag(T).
pub struct OrderedQZ<R: Runtime<DType = DType>> {
    pub q: Tensor<R>,
    pub z: Tensor<R>,
    pub s: Tensor<R>,
    pub t: Tensor<R>,
    pub num_selected: usize,
}

/// Reorder generalized Schur form for stable eigenvalues.
pub fn ordqz_impl<R, C>(
    _client: &C,
    q: &Tensor<R>,
    z: &Tensor<R>,
    s: &Tensor<R>,
    t: &Tensor<R>,
    selector: EigenvalueSelector,
) -> Result<OrderedQZ<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + MatmulOps<R>
        + UtilityOps<R>
        + LinearAlgebraAlgorithms<R>
        + RuntimeClient<R>,
{
    let shape = s.shape();
    let n = shape[0];
    if n == 0 {
        return Ok(OrderedQZ {
            q: q.clone(),
            z: z.clone(),
            s: s.clone(),
            t: t.clone(),
            num_selected: 0,
        });
    }

    let mut s_data = s.to_vec::<f64>();
    let mut t_data = t.to_vec::<f64>();
    let mut q_data = q.to_vec::<f64>();
    let mut z_data = z.to_vec::<f64>();

    // Identify blocks and selection
    let mut block_sizes = Vec::new();
    let mut i = 0;
    while i < n {
        if i + 1 < n && s_data[(i + 1) * n + i].abs() > 1e-10 {
            block_sizes.push(2);
            i += 2;
        } else {
            block_sizes.push(1);
            i += 1;
        }
    }

    let is_selected = |s_data: &[f64], t_data: &[f64], pos: usize, size: usize| -> bool {
        if size == 1 {
            let alpha = s_data[pos * n + pos];
            let beta = t_data[pos * n + pos];
            if beta.abs() < 1e-15 {
                return false; // infinite eigenvalue
            }
            let lambda = alpha / beta;
            match selector {
                EigenvalueSelector::LeftHalfPlane => lambda < 0.0,
                EigenvalueSelector::InsideUnitCircle => lambda.abs() < 1.0,
            }
        } else {
            // 2×2 block: generalized eigenvalue from det
            let s11 = s_data[pos * n + pos];
            let s12 = s_data[pos * n + pos + 1];
            let s21 = s_data[(pos + 1) * n + pos];
            let s22 = s_data[(pos + 1) * n + pos + 1];
            let t11 = t_data[pos * n + pos];
            let t22 = t_data[(pos + 1) * n + pos + 1];

            let det_s = s11 * s22 - s12 * s21;
            let det_t = t11 * t22; // T is upper triangular
            if det_t.abs() < 1e-15 {
                return false;
            }
            match selector {
                EigenvalueSelector::LeftHalfPlane => {
                    // Real part of eigenvalue
                    let trace_s = s11 + s22;
                    let trace_t = t11 + t22;
                    if trace_t.abs() < 1e-15 {
                        return false;
                    }
                    (trace_s / trace_t) < 0.0
                }
                EigenvalueSelector::InsideUnitCircle => (det_s / det_t).abs() < 1.0,
            }
        }
    };

    // Build block list
    let mut blocks: Vec<(usize, usize, bool)> = Vec::new();
    let mut pos = 0;
    for &sz in &block_sizes {
        let sel = is_selected(&s_data, &t_data, pos, sz);
        blocks.push((pos, sz, sel));
        pos += sz;
    }

    // Bubble sort selected blocks to top
    let mut swapped = true;
    while swapped {
        swapped = false;
        let mut idx = 0;
        while idx + 1 < blocks.len() {
            if !blocks[idx].2 && blocks[idx + 1].2 {
                let p = blocks[idx].0;
                let s1 = blocks[idx].1;
                let s2 = blocks[idx + 1].1;
                // For QZ, swap generalized Schur blocks using Givens on both (S, T) pairs
                swap_qz_1x1(&mut s_data, &mut t_data, &mut q_data, &mut z_data, n, p);
                blocks[idx] = (p, s2, true);
                blocks[idx + 1] = (p + s2, s1, false);
                swapped = true;
            }
            idx += 1;
        }
    }

    let num_selected: usize = blocks.iter().filter(|b| b.2).map(|b| b.1).sum();
    let device = s.device();

    Ok(OrderedQZ {
        q: Tensor::from_slice(&q_data, &[n, n], device),
        z: Tensor::from_slice(&z_data, &[n, n], device),
        s: Tensor::from_slice(&s_data, &[n, n], device),
        t: Tensor::from_slice(&t_data, &[n, n], device),
        num_selected,
    })
}

/// Swap two adjacent 1×1 blocks in a generalized Schur form.
fn swap_qz_1x1(s: &mut [f64], t: &mut [f64], q: &mut [f64], z: &mut [f64], n: usize, p: usize) {
    // Compute Givens rotation to zero out s(p+1, p) after the swap.
    // We use the standard QZ swap: apply rotations to both (S, T).

    // Step 1: Right Givens on T to zero t(p+1,p) (it should be zero already for upper tri)
    // Step 2: Left Givens on S to move eigenvalues

    let a = s[p * n + p];
    let b = s[p * n + p + 1];
    let c = s[(p + 1) * n + p + 1];
    let ta = t[p * n + p];
    let tb = t[p * n + p + 1];
    let tc = t[(p + 1) * n + p + 1];

    // We want to swap eigenvalues a/ta and c/tc.
    // Use Givens on the right to create fill-in, then Givens on the left to restore.

    // Right Givens: zero the (p, p+1) element of S - but keep structure
    // Actually, for 1×1 swaps in QZ: compute rotation from the "cross" terms
    let x = a * tc - c * ta;
    let y = b * tc - c * tb;

    if x.abs() < 1e-15 && y.abs() < 1e-15 {
        return;
    }

    // Right Givens to zero y relative to x
    let r = (x * x + y * y).sqrt();
    let cs = x / r;
    let sn = y / r;

    // Apply right Givens to S and T
    apply_givens_right(s, n, p, p + 1, cs, sn);
    apply_givens_right(t, n, p, p + 1, cs, sn);
    apply_givens_right(z, n, p, p + 1, cs, sn);

    // Now apply left Givens to restore upper triangular structure in T
    let t_p0 = t[p * n + p];
    let t_p1 = t[(p + 1) * n + p];
    let r2 = (t_p0 * t_p0 + t_p1 * t_p1).sqrt();
    if r2 > 1e-15 {
        let cs2 = t_p0 / r2;
        let sn2 = t_p1 / r2;
        apply_givens_left(s, n, p, p + 1, cs2, sn2);
        apply_givens_left(t, n, p, p + 1, cs2, sn2);
        apply_givens_left(q, n, p, p + 1, cs2, sn2);
    }

    // Clean up
    s[(p + 1) * n + p] = 0.0;
    t[(p + 1) * n + p] = 0.0;
}
