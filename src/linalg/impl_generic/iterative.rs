//! GPU-friendly iterative matrix equation solvers.
//!
//! - CARE via matrix sign function (quadratic convergence)
//! - DARE via Cayley transform → CARE sign iteration
//! - Discrete Lyapunov via Smith doubling (linear convergence via squaring)
//!
//! These use only matmul + inverse per iteration, making them ideal for GPU
//! runtimes where Schur-based methods incur expensive CPU round-trips.
use crate::DType;

use numr::algorithm::linalg::{LinearAlgebraAlgorithms, MatrixNormOrder};
use numr::error::{Error, Result};
use numr::ops::{MatmulOps, ScalarOps, ShapeOps, TensorOps, UnaryOps, UtilityOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

const TOL: f64 = 1e-12;
const MAX_ITER: usize = 100;

/// Solve CARE via matrix sign function iteration.
///
/// Forms the 2n×2n Hamiltonian H = [[A, -S], [-Q, -A^T]] where S = B R⁻¹ B^T,
/// then iterates S_{k+1} = (S_k + S_k⁻¹) / 2 until convergence to sign(H).
/// Extracts X from the projector W = (I - sign(H)) / 2.
pub fn solve_care_iterative_impl<R, C>(
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
            reason: "CARE iterative: A must be square".into(),
        });
    }
    let n = a_shape[0];
    let b_shape = b.shape();
    if b_shape.len() != 2 || b_shape[0] != n {
        return Err(Error::InvalidArgument {
            arg: "b",
            reason: format!("CARE iterative: B must be {n}×m"),
        });
    }
    let m = b_shape[1];
    if q.shape() != [n, n] {
        return Err(Error::InvalidArgument {
            arg: "q",
            reason: format!("CARE iterative: Q must be {n}×{n}"),
        });
    }
    if r.shape() != [m, m] {
        return Err(Error::InvalidArgument {
            arg: "r",
            reason: format!("CARE iterative: R must be {m}×{m}"),
        });
    }

    // S = B R⁻¹ B^T
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
    let mut s_k = client.cat(&[&top, &bottom], 0)?;

    // Sign iteration: S_{k+1} = (S_k + S_k⁻¹) / 2
    for _ in 0..MAX_ITER {
        let s_inv = LinearAlgebraAlgorithms::inverse(client, &s_k)?;
        let s_next = client.mul_scalar(&client.add(&s_k, &s_inv)?, 0.5)?;

        // Convergence: ‖S_{k+1} - S_k‖_F / ‖S_k‖_F < tol
        let diff = client.sub(&s_next, &s_k)?;
        let diff_norm =
            LinearAlgebraAlgorithms::matrix_norm(client, &diff, MatrixNormOrder::Frobenius)?;
        let s_norm =
            LinearAlgebraAlgorithms::matrix_norm(client, &s_k, MatrixNormOrder::Frobenius)?;
        let diff_val: f64 = diff_norm.to_vec()[0];
        let s_val: f64 = s_norm.to_vec()[0];

        s_k = s_next;

        if s_val > 0.0 && diff_val / s_val < TOL {
            break;
        }
    }

    // W = (I_{2n} - S_∞) / 2
    let eye_2n = client.eye(2 * n, None, a.dtype())?;
    let w = client.mul_scalar(&client.sub(&eye_2n, &s_k)?, 0.5)?;

    // Extract blocks: W11 = W[0:n, 0:n], W21 = W[n:2n, 0:n]
    let w11 = w.narrow(0, 0, n)?.narrow(1, 0, n)?.contiguous();
    let w21 = w.narrow(0, n, n)?.narrow(1, 0, n)?.contiguous();

    // X = W21 @ W11⁻¹
    let w11_inv = LinearAlgebraAlgorithms::inverse(client, &w11)?;
    let x = client.matmul(&w21, &w11_inv)?;

    // Symmetrize: X = (X + X^T) / 2
    let xt = x.transpose(0, 1)?.contiguous();
    let x_sym = client.mul_scalar(&client.add(&x, &xt)?, 0.5)?;

    Ok(x_sym)
}

/// Solve DARE iteratively via matrix sign function on the symplectic pencil.
///
/// Forms the 2n×2n symplectic matrix Z (same as `solve_dare_impl`), then applies
/// sign iteration on M⁻¹L to extract the stable invariant subspace.
pub fn solve_dare_iterative_impl<R, C>(
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
            reason: "DARE iterative: A must be square".into(),
        });
    }
    let n = a_shape[0];
    let b_shape = b.shape();
    if b_shape.len() != 2 || b_shape[0] != n {
        return Err(Error::InvalidArgument {
            arg: "b",
            reason: format!("DARE iterative: B must be {n}×m"),
        });
    }
    let m = b_shape[1];
    if q.shape() != [n, n] {
        return Err(Error::InvalidArgument {
            arg: "q",
            reason: format!("DARE iterative: Q must be {n}×{n}"),
        });
    }
    if r.shape() != [m, m] {
        return Err(Error::InvalidArgument {
            arg: "r",
            reason: format!("DARE iterative: R must be {m}×{m}"),
        });
    }

    let dtype = a.dtype();

    // S = B R⁻¹ B^T
    let r_inv = LinearAlgebraAlgorithms::inverse(client, r)?;
    let bt = b.transpose(0, 1)?.contiguous();
    let s = client.matmul(&client.matmul(b, &r_inv)?, &bt)?;

    // Build symplectic pencil matrices L and M:
    //   L = [[A, 0], [-Q, I]]
    //   M = [[I, S], [0, A^T]]
    // The generalized eigenvalue problem is Lx = λMx
    // Sign iteration on M⁻¹L
    let eye = client.eye(n, None, dtype)?;
    let zeros = client.mul_scalar(&eye, 0.0)?;
    let at = a.transpose(0, 1)?.contiguous();
    let neg_q = client.neg(q)?;

    let l_top = client.cat(&[a, &zeros], 1)?;
    let l_bottom = client.cat(&[&neg_q, &eye], 1)?;
    let l = client.cat(&[&l_top, &l_bottom], 0)?;

    let m_top = client.cat(&[&eye, &s], 1)?;
    let m_bottom = client.cat(&[&zeros, &at], 1)?;
    let m_mat = client.cat(&[&m_top, &m_bottom], 0)?;

    // G = M⁻¹ L (the symplectic matrix)
    let m_inv = LinearAlgebraAlgorithms::inverse(client, &m_mat)?;
    let g_k = client.matmul(&m_inv, &l)?;

    // Sign iteration on G: G_{k+1} = (G_k + G_k⁻¹) / 2
    // This converges to sign(G), and eigenvalues of G are the generalized
    // eigenvalues of (L, M). For DARE, stable = inside unit circle.
    // However, sign function separates left/right half plane eigenvalues.
    // We need a Cayley-like transform to map unit disk to left half plane.
    //
    // Instead: apply Cayley transform to G itself.
    // Let C = (G - I)⁻¹(G + I), then sign(C) separates |λ| < 1 from |λ| > 1.
    let eye_2n = client.eye(2 * n, None, dtype)?;
    let g_minus_i = client.sub(&g_k, &eye_2n)?;
    let g_plus_i = client.add(&g_k, &eye_2n)?;
    let g_minus_i_inv = LinearAlgebraAlgorithms::inverse(client, &g_minus_i)?;
    let mut s_k = client.matmul(&g_minus_i_inv, &g_plus_i)?;

    // Sign iteration: S_{k+1} = (S_k + S_k⁻¹) / 2
    for _ in 0..MAX_ITER {
        let s_inv = LinearAlgebraAlgorithms::inverse(client, &s_k)?;
        let s_next = client.mul_scalar(&client.add(&s_k, &s_inv)?, 0.5)?;

        let diff = client.sub(&s_next, &s_k)?;
        let diff_norm =
            LinearAlgebraAlgorithms::matrix_norm(client, &diff, MatrixNormOrder::Frobenius)?;
        let s_norm =
            LinearAlgebraAlgorithms::matrix_norm(client, &s_k, MatrixNormOrder::Frobenius)?;
        let diff_val: f64 = diff_norm.to_vec()[0];
        let s_val: f64 = s_norm.to_vec()[0];

        s_k = s_next;

        if s_val > 0.0 && diff_val / s_val < TOL {
            break;
        }
    }

    // W = (I_{2n} - S_∞) / 2 projects onto the stable (|λ| < 1) subspace
    let w = client.mul_scalar(&client.sub(&eye_2n, &s_k)?, 0.5)?;

    // Extract blocks
    let w11 = w.narrow(0, 0, n)?.narrow(1, 0, n)?.contiguous();
    let w21 = w.narrow(0, n, n)?.narrow(1, 0, n)?.contiguous();

    let w11_inv = LinearAlgebraAlgorithms::inverse(client, &w11)?;
    let x = client.matmul(&w21, &w11_inv)?;

    // Symmetrize
    let xt = x.transpose(0, 1)?.contiguous();
    let x_sym = client.mul_scalar(&client.add(&x, &xt)?, 0.5)?;

    Ok(x_sym)
}

/// Solve discrete Lyapunov AXA^T - X + Q = 0 via Smith doubling.
///
/// X₀ = Q, A₀ = A
/// X_{k+1} = A_k X_k A_k^T + X_k
/// A_{k+1} = A_k @ A_k
/// Converges when ‖A_k‖_F → 0 (in O(log(1/ρ(A))) steps).
pub fn solve_discrete_lyapunov_iterative_impl<R, C>(
    client: &C,
    a: &Tensor<R>,
    q: &Tensor<R>,
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
            reason: "discrete Lyapunov iterative: A must be square".into(),
        });
    }
    let n = a_shape[0];
    if q.shape() != [n, n] {
        return Err(Error::InvalidArgument {
            arg: "q",
            reason: format!("discrete Lyapunov iterative: Q must be {n}×{n}"),
        });
    }

    let mut x_k = q.clone();
    let mut a_k = a.clone();

    for _ in 0..MAX_ITER {
        // X_{k+1} = A_k X_k A_k^T + X_k
        let a_k_t = a_k.transpose(0, 1)?.contiguous();
        let ax = client.matmul(&a_k, &x_k)?;
        let axat = client.matmul(&ax, &a_k_t)?;
        x_k = client.add(&axat, &x_k)?;

        // A_{k+1} = A_k @ A_k
        a_k = client.matmul(&a_k, &a_k)?;

        // Convergence: ‖A_k‖_F < tol
        let a_norm =
            LinearAlgebraAlgorithms::matrix_norm(client, &a_k, MatrixNormOrder::Frobenius)?;
        let a_val: f64 = a_norm.to_vec()[0];
        if a_val < TOL {
            break;
        }
    }

    // Symmetrize: X = (X + X^T) / 2
    let xt = x_k.transpose(0, 1)?.contiguous();
    let x_sym = client.mul_scalar(&client.add(&x_k, &xt)?, 0.5)?;

    Ok(x_sym)
}
