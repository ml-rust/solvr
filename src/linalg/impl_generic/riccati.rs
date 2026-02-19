//! Algebraic Riccati equation solvers.
//!
//! - CARE: A^T X + X A - X B R^{-1} B^T X + Q = 0 (Hamiltonian Schur)
//! - DARE: A^T X A - X - A^T X B (R + B^T X B)^{-1} B^T X A + Q = 0 (Symplectic Schur)
//!
//! # Performance note
//!
//! The eigenvalue reordering step (via `ordschur_impl`) does a CPU
//! round-trip for inherently sequential Givens rotations. Prefer
//! `CpuRuntime` for these solvers to avoid unnecessary transfers.
//!
//! For GPU runtimes or large matrices, use the iterative alternatives
//! `solve_care_iterative` and `solve_dare_iterative` in the `iterative`
//! module, which use only matmul + inverse per iteration.
use crate::DType;

use super::ordschur::{EigenvalueSelector, ordschur_impl};
use numr::algorithm::linalg::LinearAlgebraAlgorithms;
use numr::error::{Error, Result};
use numr::ops::{MatmulOps, ScalarOps, ShapeOps, TensorOps, UnaryOps, UtilityOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Solve the continuous-time algebraic Riccati equation (CARE).
///
/// A^T X + X A - X B R^{-1} B^T X + Q = 0
///
/// Method: Form the 2n×2n Hamiltonian matrix H, compute ordered Schur
/// decomposition with stable eigenvalues (Re(λ) < 0) in top-left block,
/// extract X from the stable invariant subspace.
pub fn solve_care_impl<R, C>(
    client: &C,
    a: &Tensor<R>,
    b: &Tensor<R>,
    q: &Tensor<R>,
    r: &Tensor<R>,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
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
    if a_shape.len() != 2 || a_shape[0] != a_shape[1] {
        return Err(Error::InvalidArgument {
            arg: "a",
            reason: "CARE: A must be square".into(),
        });
    }
    let n = a_shape[0];
    let b_shape = b.shape();
    if b_shape.len() != 2 || b_shape[0] != n {
        return Err(Error::InvalidArgument {
            arg: "b",
            reason: format!("CARE: B must be {n}×m"),
        });
    }
    let m = b_shape[1];
    if q.shape() != [n, n] {
        return Err(Error::InvalidArgument {
            arg: "q",
            reason: format!("CARE: Q must be {n}×{n}"),
        });
    }
    if r.shape() != [m, m] {
        return Err(Error::InvalidArgument {
            arg: "r",
            reason: format!("CARE: R must be {m}×{m}"),
        });
    }

    // S = B R^{-1} B^T
    let r_inv = LinearAlgebraAlgorithms::inverse(client, r)?;
    let bt = b.transpose(0, 1)?.contiguous();
    let s = client.matmul(&client.matmul(b, &r_inv)?, &bt)?;

    // Build Hamiltonian: H = [[A, -S], [-Q, -A^T]]
    let at = a.transpose(0, 1)?.contiguous();
    let neg_s = client.neg(&s)?;
    let neg_q = client.neg(q)?;
    let neg_at = client.neg(&at)?;

    let top = client.cat(&[a, &neg_s], 1)?;
    let bottom = client.cat(&[&neg_q, &neg_at], 1)?;
    let h = client.cat(&[&top, &bottom], 0)?;

    // Schur decomposition of H, reorder stable eigenvalues to top-left
    let schur = client.schur_decompose(&h)?;
    let ordered = ordschur_impl(
        client,
        &schur.z,
        &schur.t,
        EigenvalueSelector::LeftHalfPlane,
    )?;

    if ordered.num_selected != n {
        return Err(Error::InvalidArgument {
            arg: "a",
            reason: format!(
                "CARE: expected {} stable eigenvalues, found {}",
                n, ordered.num_selected
            ),
        });
    }

    // First n columns of reordered Z span stable invariant subspace
    // X = U21 @ U11^{-1}
    let u11 = ordered.z.narrow(0, 0, n)?.narrow(1, 0, n)?.contiguous();
    let u21 = ordered.z.narrow(0, n, n)?.narrow(1, 0, n)?.contiguous();

    let u11_inv = LinearAlgebraAlgorithms::inverse(client, &u11)?;
    let x = client.matmul(&u21, &u11_inv)?;

    // Symmetrize: X = (X + X^T) / 2
    let xt = x.transpose(0, 1)?.contiguous();
    let x_sym = client.mul_scalar(&client.add(&x, &xt)?, 0.5)?;

    Ok(x_sym)
}

/// Solve the discrete-time algebraic Riccati equation (DARE).
///
/// A^T X A - X - A^T X B (R + B^T X B)^{-1} B^T X A + Q = 0
///
/// Method: Form the 2n×2n symplectic matrix Z, Schur decompose it,
/// reorder eigenvalues inside the unit circle to top-left, extract X.
///
/// The symplectic matrix is:
///   Z = [[A + S A^{-T} Q, -S A^{-T}], [-A^{-T} Q, A^{-T}]]
/// where S = B R^{-1} B^T.
///
/// Requires A to be invertible.
pub fn solve_dare_impl<R, C>(
    client: &C,
    a: &Tensor<R>,
    b: &Tensor<R>,
    q: &Tensor<R>,
    r: &Tensor<R>,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
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
    if a_shape.len() != 2 || a_shape[0] != a_shape[1] {
        return Err(Error::InvalidArgument {
            arg: "a",
            reason: "DARE: A must be square".into(),
        });
    }
    let n = a_shape[0];
    let b_shape = b.shape();
    if b_shape.len() != 2 || b_shape[0] != n {
        return Err(Error::InvalidArgument {
            arg: "b",
            reason: format!("DARE: B must be {n}×m"),
        });
    }
    let m = b_shape[1];
    if q.shape() != [n, n] {
        return Err(Error::InvalidArgument {
            arg: "q",
            reason: format!("DARE: Q must be {n}×{n}"),
        });
    }
    if r.shape() != [m, m] {
        return Err(Error::InvalidArgument {
            arg: "r",
            reason: format!("DARE: R must be {m}×{m}"),
        });
    }

    // S = B R^{-1} B^T
    let r_inv = LinearAlgebraAlgorithms::inverse(client, r)?;
    let bt = b.transpose(0, 1)?.contiguous();
    let s = client.matmul(&client.matmul(b, &r_inv)?, &bt)?;

    // A^{-T} = (A^{-1})^T = (A^T)^{-1}
    let a_inv = LinearAlgebraAlgorithms::inverse(client, a)?;
    let a_inv_t = a_inv.transpose(0, 1)?.contiguous();

    // Build symplectic matrix:
    // Z = [[A + S A^{-T} Q, -S A^{-T}], [-A^{-T} Q, A^{-T}]]
    let a_inv_t_q = client.matmul(&a_inv_t, q)?;
    let s_a_inv_t = client.matmul(&s, &a_inv_t)?;
    let s_a_inv_t_q = client.matmul(&s_a_inv_t, q)?;

    let z11 = client.add(a, &s_a_inv_t_q)?;
    let z12 = client.neg(&s_a_inv_t)?;
    let z21 = client.neg(&a_inv_t_q)?;

    let top = client.cat(&[&z11, &z12], 1)?;
    let bottom = client.cat(&[&z21, &a_inv_t], 1)?;
    let z_mat = client.cat(&[&top, &bottom], 0)?;

    // Schur decomposition, reorder eigenvalues inside unit circle to top-left
    let schur = client.schur_decompose(&z_mat)?;
    let ordered = ordschur_impl(
        client,
        &schur.z,
        &schur.t,
        EigenvalueSelector::InsideUnitCircle,
    )?;

    if ordered.num_selected != n {
        return Err(Error::InvalidArgument {
            arg: "a",
            reason: format!(
                "DARE: expected {} stable eigenvalues, found {}",
                n, ordered.num_selected
            ),
        });
    }

    // First n columns of reordered Z span stable invariant subspace
    let w11 = ordered.z.narrow(0, 0, n)?.narrow(1, 0, n)?.contiguous();
    let w21 = ordered.z.narrow(0, n, n)?.narrow(1, 0, n)?.contiguous();

    let w11_inv = LinearAlgebraAlgorithms::inverse(client, &w11)?;
    let x = client.matmul(&w21, &w11_inv)?;

    // Symmetrize
    let xt = x.transpose(0, 1)?.contiguous();
    let x_sym = client.mul_scalar(&client.add(&x, &xt)?, 0.5)?;

    Ok(x_sym)
}
