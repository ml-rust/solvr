//! Bartels-Stewart algorithm for the Sylvester equation AX + XB = C.
//!
//! 1. Compute real Schur decompositions: A = U T_A U^T, B = V T_B V^T
//! 2. Transform: F = U^T C V
//! 3. Solve column-by-column: (T_A + t_bj I) y_j = f_j - Σ_{k<j} t_bkj y_k
//! 4. Back-transform: X = U Y V^T
//!
//! # Performance note
//!
//! Step 3 extracts the quasi-triangular factors to CPU for sequential
//! column back-substitution — each column depends on all previous columns.
//! Prefer `CpuRuntime` for this solver — GPU transfers add overhead with
//! no computational benefit for inherently sequential back-substitution.

use numr::algorithm::linalg::LinearAlgebraAlgorithms;
use numr::dtype::DType;
use numr::error::{Error, Result};
use numr::ops::{MatmulOps, ScalarOps, ShapeOps, TensorOps, UnaryOps, UtilityOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Solve the Sylvester equation AX + XB = C using Bartels-Stewart.
///
/// A is n×n, B is m×m, C is n×m. Returns X (n×m).
pub fn sylvester_impl<R, C>(
    client: &C,
    a: &Tensor<R>,
    b: &Tensor<R>,
    c: &Tensor<R>,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + UnaryOps<R>
        + MatmulOps<R>
        + UtilityOps<R>
        + LinearAlgebraAlgorithms<R>
        + RuntimeClient<R>,
{
    let a_shape = a.shape();
    let b_shape = b.shape();
    let c_shape = c.shape();

    // Validate dimensions
    if a_shape.len() != 2 || a_shape[0] != a_shape[1] {
        return Err(Error::InvalidArgument {
            arg: "a",
            reason: "Sylvester: A must be square".into(),
        });
    }
    if b_shape.len() != 2 || b_shape[0] != b_shape[1] {
        return Err(Error::InvalidArgument {
            arg: "b",
            reason: "Sylvester: B must be square".into(),
        });
    }
    let na = a_shape[0];
    let mb = b_shape[0];
    if c_shape != [na, mb] {
        return Err(Error::InvalidArgument {
            arg: "c",
            reason: format!("Sylvester: C must be {}×{}, got {:?}", na, mb, c_shape),
        });
    }

    let dtype = a.dtype();
    if dtype != DType::F32 && dtype != DType::F64 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "solve_sylvester",
        });
    }

    // Step 1: Schur decompositions
    let schur_a = client.schur_decompose(a)?;
    let schur_b = client.schur_decompose(b)?;

    let u = &schur_a.z; // n×n orthogonal
    let ta = &schur_a.t; // n×n upper quasi-triangular
    let v = &schur_b.z; // m×m orthogonal
    let tb = &schur_b.t; // m×m upper quasi-triangular

    // Step 2: F = U^T C V
    let ut = u.transpose(0, 1)?;
    let f = client.matmul(&client.matmul(&ut, c)?, v)?;

    // Step 3: Solve column by column in the transformed space.
    // (T_A + t_bj·I) y_j = f_j - Σ_{k<j} t_bkj · y_k
    //
    // T_B is upper quasi-triangular, so we process columns left-to-right.
    // For 2×2 blocks on diagonal of T_B, we solve a coupled 2n×2n system.
    //
    // We extract to CPU for this sequential column solve since it's inherently
    // serial and n is typically small for control problems.
    let ta_data = ta.to_vec::<f64>();
    let tb_data = tb.to_vec::<f64>();
    let f_data = f.to_vec::<f64>();
    let mut y_data = vec![0.0f64; na * mb];

    let mut j = 0;
    while j < mb {
        // Check if j is part of a 2×2 block on T_B diagonal
        let block_size = if j + 1 < mb && tb_data[(j + 1) * mb + j].abs() > 1e-10 {
            2
        } else {
            1
        };

        if block_size == 1 {
            // Single column solve: (T_A + t_bjj · I) y_j = rhs_j
            let tbjj = tb_data[j * mb + j];

            // Build RHS: f_j - Σ_{k<j} t_bkj · y_k
            let mut rhs = vec![0.0f64; na];
            for i in 0..na {
                rhs[i] = f_data[i * mb + j];
                for k in 0..j {
                    rhs[i] -= tb_data[k * mb + j] * y_data[i * mb + k];
                }
            }

            // Solve (T_A + tbjj · I) y_j = rhs
            // T_A + tbjj · I is upper quasi-triangular, solve by back-substitution
            solve_quasi_upper_shifted(&ta_data, na, tbjj, &rhs, &mut y_data, mb, j);
        } else {
            // 2×2 block: columns j and j+1 coupled
            let tb_jj = tb_data[j * mb + j];
            let tb_jj1 = tb_data[j * mb + j + 1];
            let tb_j1j = tb_data[(j + 1) * mb + j];
            let tb_j1j1 = tb_data[(j + 1) * mb + j + 1];

            // Build RHS for both columns
            let mut rhs0 = vec![0.0f64; na];
            let mut rhs1 = vec![0.0f64; na];
            for i in 0..na {
                rhs0[i] = f_data[i * mb + j];
                rhs1[i] = f_data[i * mb + j + 1];
                for k in 0..j {
                    rhs0[i] -= tb_data[k * mb + j] * y_data[i * mb + k];
                    rhs1[i] -= tb_data[k * mb + j + 1] * y_data[i * mb + k];
                }
            }

            // Solve coupled system row by row (back-substitution on 2×2 diagonal blocks)
            solve_quasi_upper_coupled(
                &ta_data,
                na,
                tb_jj,
                tb_jj1,
                tb_j1j,
                tb_j1j1,
                &rhs0,
                &rhs1,
                &mut y_data,
                mb,
                j,
            );
            j += 1; // extra increment for 2-column block
        }
        j += 1;
    }

    // Step 4: X = U Y V^T
    let device = a.device();
    let y = Tensor::from_slice(&y_data, &[na, mb], device);
    let vt = v.transpose(0, 1)?;
    client.matmul(&client.matmul(u, &y)?, &vt)
}

/// Back-substitution for (T_A + shift·I) x = rhs where T_A is upper quasi-triangular.
fn solve_quasi_upper_shifted(
    ta: &[f64],
    n: usize,
    shift: f64,
    rhs: &[f64],
    y: &mut [f64],
    y_cols: usize,
    col: usize,
) {
    let mut x = vec![0.0f64; n];

    let mut i = n;
    while i > 0 {
        i -= 1;
        // Check if i is part of a 2×2 block
        if i > 0 && ta[i * n + i - 1].abs() > 1e-10 {
            // 2×2 block at rows (i-1, i)
            let i0 = i - 1;
            let a00 = ta[i0 * n + i0] + shift;
            let a01 = ta[i0 * n + i];
            let a10 = ta[i * n + i0];
            let a11 = ta[i * n + i] + shift;

            let mut r0 = rhs[i0];
            let mut r1 = rhs[i];
            for k in i + 1..n {
                r0 -= ta[i0 * n + k] * x[k];
                r1 -= ta[i * n + k] * x[k];
            }

            let det = a00 * a11 - a01 * a10;
            if det.abs() > 1e-15 {
                x[i0] = (a11 * r0 - a01 * r1) / det;
                x[i] = (-a10 * r0 + a00 * r1) / det;
            }
            i -= 1; // skip the other row of the 2×2 block
        } else {
            // 1×1 block
            let diag = ta[i * n + i] + shift;
            let mut r = rhs[i];
            for k in i + 1..n {
                r -= ta[i * n + k] * x[k];
            }
            if diag.abs() > 1e-15 {
                x[i] = r / diag;
            }
        }
    }

    for i in 0..n {
        y[i * y_cols + col] = x[i];
    }
}

/// Solve coupled 2-column system for a 2×2 block on T_B diagonal.
/// (T_A ⊗ I + I ⊗ T_B_block) [y_j; y_{j+1}] = [rhs0; rhs1]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)]
fn solve_quasi_upper_coupled(
    ta: &[f64],
    n: usize,
    tb00: f64,
    tb01: f64,
    tb10: f64,
    tb11: f64,
    rhs0: &[f64],
    rhs1: &[f64],
    y: &mut [f64],
    y_cols: usize,
    col: usize,
) {
    let mut x0 = vec![0.0f64; n];
    let mut x1 = vec![0.0f64; n];

    let mut i = n;
    while i > 0 {
        i -= 1;
        // Check for 2×2 block on T_A
        if i > 0 && ta[i * n + i - 1].abs() > 1e-10 {
            // 4×4 system for rows (i-1, i) × columns (j, j+1)
            let i0 = i - 1;
            let a00 = ta[i0 * n + i0];
            let a01 = ta[i0 * n + i];
            let a10 = ta[i * n + i0];
            let a11 = ta[i * n + i];

            let mut r00 = rhs0[i0];
            let mut r10 = rhs0[i];
            let mut r01 = rhs1[i0];
            let mut r11 = rhs1[i];

            for k in i + 1..n {
                r00 -= ta[i0 * n + k] * x0[k];
                r10 -= ta[i * n + k] * x0[k];
                r01 -= ta[i0 * n + k] * x1[k];
                r11 -= ta[i * n + k] * x1[k];
            }

            // 4×4 system: M [x0_i0, x0_i, x1_i0, x1_i]^T = [r00, r10, r01, r11]^T
            // M = [[a00+tb00, a01, tb01, 0],
            //      [a10, a11+tb00, 0, tb01],
            //      [tb10, 0, a00+tb11, a01],
            //      [0, tb10, a10, a11+tb11]]
            let mut m = [
                [a00 + tb00, a01, tb01, 0.0],
                [a10, a11 + tb00, 0.0, tb01],
                [tb10, 0.0, a00 + tb11, a01],
                [0.0, tb10, a10, a11 + tb11],
            ];
            let mut rhs_4 = [r00, r10, r01, r11];

            // Gaussian elimination with partial pivoting
            for col_idx in 0..4 {
                let mut max_row = col_idx;
                let mut max_val = m[col_idx][col_idx].abs();
                for row in col_idx + 1..4 {
                    if m[row][col_idx].abs() > max_val {
                        max_val = m[row][col_idx].abs();
                        max_row = row;
                    }
                }
                if max_val < 1e-15 {
                    continue;
                }
                if max_row != col_idx {
                    m.swap(col_idx, max_row);
                    rhs_4.swap(col_idx, max_row);
                }
                let pivot = m[col_idx][col_idx];
                for row in col_idx + 1..4 {
                    let factor = m[row][col_idx] / pivot;
                    for jj in col_idx..4 {
                        m[row][jj] -= factor * m[col_idx][jj];
                    }
                    rhs_4[row] -= factor * rhs_4[col_idx];
                }
            }

            let mut sol = [0.0f64; 4];
            for ii in (0..4).rev() {
                let mut s = rhs_4[ii];
                for jj in ii + 1..4 {
                    s -= m[ii][jj] * sol[jj];
                }
                if m[ii][ii].abs() > 1e-15 {
                    sol[ii] = s / m[ii][ii];
                }
            }

            x0[i0] = sol[0];
            x0[i] = sol[1];
            x1[i0] = sol[2];
            x1[i] = sol[3];
            i -= 1;
        } else {
            // 2×2 system for row i × columns (j, j+1)
            let aii = ta[i * n + i];
            let mut r0 = rhs0[i];
            let mut r1 = rhs1[i];
            for k in i + 1..n {
                r0 -= ta[i * n + k] * x0[k];
                r1 -= ta[i * n + k] * x1[k];
            }

            // [[aii+tb00, tb01], [tb10, aii+tb11]] [x0_i, x1_i] = [r0, r1]
            let m00 = aii + tb00;
            let m01 = tb01;
            let m10 = tb10;
            let m11 = aii + tb11;
            let det = m00 * m11 - m01 * m10;
            if det.abs() > 1e-15 {
                x0[i] = (m11 * r0 - m01 * r1) / det;
                x1[i] = (-m10 * r0 + m00 * r1) / det;
            }
        }
    }

    for i in 0..n {
        y[i * y_cols + col] = x0[i];
        y[i * y_cols + col + 1] = x1[i];
    }
}
