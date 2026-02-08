//! Lyapunov equation solvers.
//!
//! - Continuous: AX + XA^T = Q (delegates to Sylvester with B = A^T)
//! - Discrete: AXA^T - X + Q = 0 (bilinear transform → continuous Lyapunov)
//!
//! For GPU runtimes or large matrices, the discrete equation also has an
//! iterative alternative `solve_discrete_lyapunov_iterative` via Smith
//! doubling (see `iterative` module).

use super::sylvester::sylvester_impl;
use numr::algorithm::linalg::LinearAlgebraAlgorithms;
use numr::error::{Error, Result};
use numr::ops::{MatmulOps, ScalarOps, ShapeOps, TensorOps, UnaryOps, UtilityOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Solve the continuous Lyapunov equation AX + XA^T = Q.
///
/// This is AX + XB = C with B = A^T and C = Q.
pub fn continuous_lyapunov_impl<R, C>(client: &C, a: &Tensor<R>, q: &Tensor<R>) -> Result<Tensor<R>>
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
    if a_shape.len() != 2 || a_shape[0] != a_shape[1] {
        return Err(Error::InvalidArgument {
            arg: "a",
            reason: "continuous Lyapunov: A must be square".into(),
        });
    }
    let n = a_shape[0];
    let q_shape = q.shape();
    if q_shape != [n, n] {
        return Err(Error::InvalidArgument {
            arg: "q",
            reason: format!("continuous Lyapunov: Q must be {n}×{n}, got {:?}", q_shape),
        });
    }

    let at = a.transpose(0, 1)?.contiguous();
    sylvester_impl(client, a, &at, q)
}

/// Solve the discrete Lyapunov equation AXA^T - X + Q = 0.
///
/// Uses bilinear (Cayley) transform: A_c = (A - I)^{-1}(A + I)
/// transforms to continuous Lyapunov: A_c X + X A_c^T = Q_c
/// where Q_c = (A - I)^{-1} Q (A - I)^{-T}.
///
/// This is valid when A has no eigenvalue at +1.
pub fn discrete_lyapunov_impl<R, C>(client: &C, a: &Tensor<R>, q: &Tensor<R>) -> Result<Tensor<R>>
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
    if a_shape.len() != 2 || a_shape[0] != a_shape[1] {
        return Err(Error::InvalidArgument {
            arg: "a",
            reason: "discrete Lyapunov: A must be square".into(),
        });
    }
    let n = a_shape[0];
    let q_shape = q.shape();
    if q_shape != [n, n] {
        return Err(Error::InvalidArgument {
            arg: "q",
            reason: format!("discrete Lyapunov: Q must be {n}×{n}, got {:?}", q_shape),
        });
    }

    let dtype = a.dtype();

    // A_minus_I = A - I
    let eye = client.eye(n, None, dtype)?;
    let a_minus_i = client.sub(a, &eye)?;
    let a_plus_i = client.add(a, &eye)?;

    // A_c = (A - I)^{-1} (A + I)
    let a_minus_i_inv = LinearAlgebraAlgorithms::inverse(client, &a_minus_i)?;
    let a_c = client.matmul(&a_minus_i_inv, &a_plus_i)?;

    // Q_c = (A - I)^{-1} Q (A - I)^{-T}
    let a_minus_i_inv_t = a_minus_i_inv.transpose(0, 1)?.contiguous();
    let q_c = client.matmul(&client.matmul(&a_minus_i_inv, q)?, &a_minus_i_inv_t)?;

    // The bilinear transform gives: A_c X + X A_c^T = -2 Q_c
    // Our continuous solver solves A_c X + X A_c^T = Q_arg, so pass -2 Q_c.
    let rhs = client.mul_scalar(&client.neg(&q_c)?, 2.0)?;
    continuous_lyapunov_impl(client, &a_c, &rhs)
}
