//! Matrix equation solver traits.
use crate::DType;

use numr::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Algorithms for solving matrix equations arising in control theory.
///
/// All methods require square matrix inputs. Matrices A, B, Q must have
/// compatible dimensions as specified per method.
pub trait MatrixEquationAlgorithms<R: Runtime<DType = DType>> {
    /// Solve the Sylvester equation AX + XB = C.
    ///
    /// Uses the Bartels-Stewart algorithm (real Schur decomposition).
    ///
    /// # Arguments
    /// - `a`: n×n matrix
    /// - `b`: m×m matrix
    /// - `c`: n×m matrix
    ///
    /// # Returns
    /// X such that AX + XB = C
    fn solve_sylvester(&self, a: &Tensor<R>, b: &Tensor<R>, c: &Tensor<R>) -> Result<Tensor<R>>;

    /// Solve the continuous Lyapunov equation AX + XA^T = Q.
    ///
    /// Special case of Sylvester with B = A^T.
    ///
    /// # Arguments
    /// - `a`: n×n matrix (must have eigenvalues with negative real parts for unique solution)
    /// - `q`: n×n symmetric matrix
    ///
    /// # Returns
    /// X such that AX + XA^T = Q
    fn solve_continuous_lyapunov(&self, a: &Tensor<R>, q: &Tensor<R>) -> Result<Tensor<R>>;

    /// Solve the discrete Lyapunov equation AXA^T - X + Q = 0.
    ///
    /// Uses bilinear transformation to convert to continuous form.
    ///
    /// # Arguments
    /// - `a`: n×n matrix (must have eigenvalues inside unit circle for unique solution)
    /// - `q`: n×n symmetric matrix
    ///
    /// # Returns
    /// X such that AXA^T - X + Q = 0
    fn solve_discrete_lyapunov(&self, a: &Tensor<R>, q: &Tensor<R>) -> Result<Tensor<R>>;

    /// Solve the continuous-time algebraic Riccati equation (CARE).
    ///
    /// A^T X + X A - X B R^{-1} B^T X + Q = 0
    ///
    /// Uses Hamiltonian Schur decomposition with eigenvalue reordering.
    ///
    /// # Arguments
    /// - `a`: n×n state matrix
    /// - `b`: n×m input matrix
    /// - `q`: n×n state cost (symmetric positive semidefinite)
    /// - `r`: m×m input cost (symmetric positive definite)
    ///
    /// # Returns
    /// X: n×n stabilizing solution
    fn solve_care(
        &self,
        a: &Tensor<R>,
        b: &Tensor<R>,
        q: &Tensor<R>,
        r: &Tensor<R>,
    ) -> Result<Tensor<R>>;

    /// Solve the discrete-time algebraic Riccati equation (DARE).
    ///
    /// A^T X A - X - A^T X B (R + B^T X B)^{-1} B^T X A + Q = 0
    ///
    /// Uses generalized Schur (QZ) decomposition with eigenvalue reordering.
    ///
    /// # Arguments
    /// - `a`: n×n state matrix
    /// - `b`: n×m input matrix
    /// - `q`: n×n state cost (symmetric positive semidefinite)
    /// - `r`: m×m input cost (symmetric positive definite)
    ///
    /// # Returns
    /// X: n×n stabilizing solution
    fn solve_dare(
        &self,
        a: &Tensor<R>,
        b: &Tensor<R>,
        q: &Tensor<R>,
        r: &Tensor<R>,
    ) -> Result<Tensor<R>>;

    /// Solve CARE iteratively via matrix sign function.
    ///
    /// A^T X + X A - X B R^{-1} B^T X + Q = 0
    ///
    /// GPU-friendly: uses only matmul and inverse per iteration (no CPU transfers
    /// except a single scalar for convergence check). Quadratic convergence,
    /// typically 15-25 iterations.
    ///
    /// Prefer this over `solve_care` on GPU runtimes or for large matrices.
    fn solve_care_iterative(
        &self,
        a: &Tensor<R>,
        b: &Tensor<R>,
        q: &Tensor<R>,
        r: &Tensor<R>,
    ) -> Result<Tensor<R>>;

    /// Solve DARE iteratively via Cayley transform + matrix sign function.
    ///
    /// A^T X A - X - A^T X B (R + B^T X B)^{-1} B^T X A + Q = 0
    ///
    /// GPU-friendly: uses only matmul and inverse per iteration (no CPU transfers
    /// except a single scalar for convergence check).
    ///
    /// Prefer this over `solve_dare` on GPU runtimes or for large matrices.
    fn solve_dare_iterative(
        &self,
        a: &Tensor<R>,
        b: &Tensor<R>,
        q: &Tensor<R>,
        r: &Tensor<R>,
    ) -> Result<Tensor<R>>;

    /// Solve discrete Lyapunov iteratively via Smith doubling.
    ///
    /// AXA^T - X + Q = 0
    ///
    /// GPU-friendly: uses only matmul per iteration (no CPU transfers except
    /// a single scalar for convergence check). Converges in O(log(1/ρ(A))) steps.
    ///
    /// Prefer this over `solve_discrete_lyapunov` on GPU runtimes or for large matrices.
    fn solve_discrete_lyapunov_iterative(&self, a: &Tensor<R>, q: &Tensor<R>) -> Result<Tensor<R>>;
}
