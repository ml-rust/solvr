//! CPU implementation of matrix equation solvers.

use crate::linalg::impl_generic::{
    continuous_lyapunov_impl, discrete_lyapunov_impl, solve_care_impl, solve_care_iterative_impl,
    solve_dare_impl, solve_dare_iterative_impl, solve_discrete_lyapunov_iterative_impl,
    sylvester_impl,
};
use crate::linalg::traits::matrix_equations::MatrixEquationAlgorithms;
use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl MatrixEquationAlgorithms<CpuRuntime> for CpuClient {
    fn solve_sylvester(
        &self,
        a: &Tensor<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
        c: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        sylvester_impl(self, a, b, c)
    }

    fn solve_continuous_lyapunov(
        &self,
        a: &Tensor<CpuRuntime>,
        q: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        continuous_lyapunov_impl(self, a, q)
    }

    fn solve_discrete_lyapunov(
        &self,
        a: &Tensor<CpuRuntime>,
        q: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        discrete_lyapunov_impl(self, a, q)
    }

    fn solve_care(
        &self,
        a: &Tensor<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
        q: &Tensor<CpuRuntime>,
        r: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        solve_care_impl(self, a, b, q, r)
    }

    fn solve_dare(
        &self,
        a: &Tensor<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
        q: &Tensor<CpuRuntime>,
        r: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        solve_dare_impl(self, a, b, q, r)
    }

    fn solve_care_iterative(
        &self,
        a: &Tensor<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
        q: &Tensor<CpuRuntime>,
        r: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        solve_care_iterative_impl(self, a, b, q, r)
    }

    fn solve_dare_iterative(
        &self,
        a: &Tensor<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
        q: &Tensor<CpuRuntime>,
        r: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        solve_dare_iterative_impl(self, a, b, q, r)
    }

    fn solve_discrete_lyapunov_iterative(
        &self,
        a: &Tensor<CpuRuntime>,
        q: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        solve_discrete_lyapunov_iterative_impl(self, a, q)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::algorithm::linalg::LinearAlgebraAlgorithms;
    use numr::ops::{BinaryOps, MatmulOps, ReduceOps};
    use numr::runtime::cpu::CpuDevice;

    fn setup() -> (CpuClient, CpuDevice) {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        (client, device)
    }

    /// Compute Frobenius norm of a matrix (as scalar).
    fn frob_norm(client: &CpuClient, m: &Tensor<CpuRuntime>) -> f64 {
        let sq = client.mul(m, m).unwrap();
        let sum = client.sum(&sq, &[0, 1], false).unwrap();
        sum.to_vec::<f64>()[0].sqrt()
    }

    // ========================================================================
    // Sylvester tests
    // ========================================================================

    #[test]
    fn test_sylvester_diagonal() {
        let (client, device) = setup();

        // A = diag(-1, -2), B = diag(-3, -4), C = [[1,2],[3,4]]
        // AX + XB = C => x_ij = c_ij / (a_ii + b_jj)
        let a = Tensor::from_slice(&[-1.0, 0.0, 0.0, -2.0], &[2, 2], &device);
        let b = Tensor::from_slice(&[-3.0, 0.0, 0.0, -4.0], &[2, 2], &device);
        let c = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &device);

        let x = client.solve_sylvester(&a, &b, &c).unwrap();
        let x_data = x.to_vec::<f64>();

        // x_11 = 1/(-1-3) = -0.25, x_12 = 2/(-1-4) = -0.4
        // x_21 = 3/(-2-3) = -0.6,  x_22 = 4/(-2-4) = -2/3
        assert!((x_data[0] - (-0.25)).abs() < 1e-10);
        assert!((x_data[1] - (-0.4)).abs() < 1e-10);
        assert!((x_data[2] - (-0.6)).abs() < 1e-10);
        assert!((x_data[3] - (-2.0 / 3.0)).abs() < 1e-10);
    }

    #[test]
    fn test_sylvester_residual() {
        let (client, device) = setup();

        // Random-ish stable matrices
        let a = Tensor::from_slice(
            &[-2.0, 1.0, 0.0, 0.5, -3.0, 0.5, 0.0, 1.0, -1.0],
            &[3, 3],
            &device,
        );
        let b = Tensor::from_slice(&[-1.0, 0.5, 0.5, -2.0], &[2, 2], &device);
        let c = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], &device);

        let x = client.solve_sylvester(&a, &b, &c).unwrap();

        // Verify: AX + XB - C ≈ 0
        let ax = client.matmul(&a, &x).unwrap();
        let xb = client.matmul(&x, &b).unwrap();
        let residual = client.sub(&client.add(&ax, &xb).unwrap(), &c).unwrap();

        let norm = frob_norm(&client, &residual);
        assert!(norm < 1e-10, "Sylvester residual norm = {}", norm);
    }

    // ========================================================================
    // Continuous Lyapunov tests
    // ========================================================================

    #[test]
    fn test_continuous_lyapunov() {
        let (client, device) = setup();

        // Stable A, identity Q
        let a = Tensor::from_slice(&[-1.0, 0.5, 0.0, -2.0], &[2, 2], &device);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2], &device);

        let x = client.solve_continuous_lyapunov(&a, &q).unwrap();

        // Verify: AX + XA^T - Q ≈ 0
        let at = a.transpose(0, 1).unwrap().contiguous();
        let ax = client.matmul(&a, &x).unwrap();
        let xat = client.matmul(&x, &at).unwrap();
        let residual = client.sub(&client.add(&ax, &xat).unwrap(), &q).unwrap();

        let norm = frob_norm(&client, &residual);
        assert!(norm < 1e-10, "Continuous Lyapunov residual = {}", norm);
    }

    // ========================================================================
    // Discrete Lyapunov tests
    // ========================================================================

    #[test]
    fn test_discrete_lyapunov() {
        let (client, device) = setup();

        // Stable A (eigenvalues inside unit circle)
        let a = Tensor::from_slice(&[0.5, 0.1, 0.0, 0.3], &[2, 2], &device);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2], &device);

        let x = client.solve_discrete_lyapunov(&a, &q).unwrap();

        // Verify: AXA^T - X + Q ≈ 0
        let at = a.transpose(0, 1).unwrap().contiguous();
        let axat = client.matmul(&client.matmul(&a, &x).unwrap(), &at).unwrap();
        let residual = client.add(&client.sub(&axat, &x).unwrap(), &q).unwrap();

        let norm = frob_norm(&client, &residual);
        assert!(norm < 1e-10, "Discrete Lyapunov residual = {}", norm);
    }

    // ========================================================================
    // CARE tests
    // ========================================================================

    #[test]
    fn test_care_debug() {
        use numr::algorithm::linalg::LinearAlgebraAlgorithms;
        use numr::ops::{MatmulOps, ShapeOps, UnaryOps};
        let (client, device) = setup();

        let a = Tensor::from_slice(&[0.0, 1.0, 0.0, 0.0], &[2, 2], &device);
        let b = Tensor::from_slice(&[0.0, 1.0], &[2, 1], &device);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2], &device);
        let r = Tensor::from_slice(&[1.0], &[1, 1], &device);

        let r_inv = LinearAlgebraAlgorithms::inverse(&client, &r).unwrap();
        let bt = b.transpose(0, 1).unwrap().contiguous();
        let s = client
            .matmul(&client.matmul(&b, &r_inv).unwrap(), &bt)
            .unwrap();
        eprintln!("S = {:?}", s.to_vec::<f64>());

        let at = a.transpose(0, 1).unwrap().contiguous();
        let neg_s = client.neg(&s).unwrap();
        let neg_q = client.neg(&q).unwrap();
        let neg_at = client.neg(&at).unwrap();

        let top = client.cat(&[&a, &neg_s], 1).unwrap();
        let bottom = client.cat(&[&neg_q, &neg_at], 1).unwrap();
        let h = client.cat(&[&top, &bottom], 0).unwrap();
        eprintln!("H = {:?}", h.to_vec::<f64>());

        let schur = client.schur_decompose(&h).unwrap();
        let t_data = schur.t.to_vec::<f64>();
        let z_data = schur.z.to_vec::<f64>();
        let n = 4;
        eprintln!("Schur T:");
        for i in 0..n {
            let row: Vec<f64> = (0..n).map(|j| t_data[i * n + j]).collect();
            eprintln!("  {:?}", row);
        }
        eprintln!("Schur Z:");
        for i in 0..n {
            let row: Vec<f64> = (0..n).map(|j| z_data[i * n + j]).collect();
            eprintln!("  {:?}", row);
        }

        // Verify Schur convention: is H = Z T Z^T or H = Z^T T Z?
        let zt = schur.z.transpose(0, 1).unwrap().contiguous();
        let ztzt = client
            .matmul(&client.matmul(&schur.z, &schur.t).unwrap(), &zt)
            .unwrap();
        let diff1 = client.sub(&ztzt, &h).unwrap();
        let norm1 = frob_norm(&client, &diff1);
        eprintln!("||Z T Z^T - H|| = {}", norm1);

        let zttzt = client
            .matmul(&client.matmul(&zt, &schur.t).unwrap(), &schur.z)
            .unwrap();
        let diff2 = client.sub(&zttzt, &h).unwrap();
        let norm2 = frob_norm(&client, &diff2);
        eprintln!("||Z^T T Z - H|| = {}", norm2);

        // Check eigenvalues
        let mut i = 0;
        while i < n {
            if i + 1 < n && t_data[(i + 1) * n + i].abs() > 1e-10 {
                let a11 = t_data[i * n + i];
                let a22 = t_data[(i + 1) * n + i + 1];
                let a12 = t_data[i * n + i + 1];
                let a21 = t_data[(i + 1) * n + i];
                let re = (a11 + a22) / 2.0;
                let det = a11 * a22 - a12 * a21;
                let disc = re * re - det;
                if disc < 0.0 {
                    eprintln!(
                        "2x2 block {},{}: eigenvalues = {} ± {}i",
                        i,
                        i + 1,
                        re,
                        (-disc).sqrt()
                    );
                }
                i += 2;
            } else {
                eprintln!("1x1 block {}: eigenvalue = {}", i, t_data[i * n + i]);
                i += 1;
            }
        }

        // Try extracting X from stable columns (2,3)
        let stable_z = schur.z.narrow(1, 2, 2).unwrap().contiguous();
        let u1 = stable_z.narrow(0, 0, 2).unwrap().contiguous();
        let u2 = stable_z.narrow(0, 2, 2).unwrap().contiguous();
        let u1_inv = LinearAlgebraAlgorithms::inverse(&client, &u1).unwrap();
        let x_stable = client.matmul(&u2, &u1_inv).unwrap();
        eprintln!("X from stable subspace: {:?}", x_stable.to_vec::<f64>());

        // Try unstable columns (0,1)
        let unstable_z = schur.z.narrow(1, 0, 2).unwrap().contiguous();
        let u1u = unstable_z.narrow(0, 0, 2).unwrap().contiguous();
        let u2u = unstable_z.narrow(0, 2, 2).unwrap().contiguous();
        let u1u_inv = LinearAlgebraAlgorithms::inverse(&client, &u1u).unwrap();
        let x_unstable = client.matmul(&u2u, &u1u_inv).unwrap();
        eprintln!("X from unstable subspace: {:?}", x_unstable.to_vec::<f64>());

        // Try: -U2 inv(U1)
        let neg_u2 = client.neg(&u2).unwrap();
        let x_neg = client.matmul(&neg_u2, &u1_inv).unwrap();
        eprintln!("X = -U2 inv(U1): {:?}", x_neg.to_vec::<f64>());

        // Check residual for each
        let s_mat = Tensor::from_slice(&[0.0, 0.0, 0.0, 1.0], &[2, 2], &device);
        for (name, x_candidate) in [
            ("stable", &x_stable),
            ("unstable", &x_unstable),
            ("neg_stable", &x_neg),
        ] {
            let at_loc = a.transpose(0, 1).unwrap().contiguous();
            let atx = client.matmul(&at_loc, x_candidate).unwrap();
            let xa = client.matmul(x_candidate, &a).unwrap();
            let xsx = client
                .matmul(&client.matmul(x_candidate, &s_mat).unwrap(), x_candidate)
                .unwrap();
            let res = client
                .add(
                    &client.sub(&client.add(&atx, &xa).unwrap(), &xsx).unwrap(),
                    &q,
                )
                .unwrap();
            eprintln!("CARE residual ({}) = {}", name, frob_norm(&client, &res));
        }

        // Check with known solution
        let x_true = Tensor::from_slice(&[1.732050808, 1.0, 1.0, 1.732050808], &[2, 2], &device);
        {
            let at_loc = a.transpose(0, 1).unwrap().contiguous();
            let atx = client.matmul(&at_loc, &x_true).unwrap();
            let xa = client.matmul(&x_true, &a).unwrap();
            let xsx = client
                .matmul(&client.matmul(&x_true, &s_mat).unwrap(), &x_true)
                .unwrap();
            let res = client
                .add(
                    &client.sub(&client.add(&atx, &xa).unwrap(), &xsx).unwrap(),
                    &q,
                )
                .unwrap();
            eprintln!("CARE residual (known X) = {}", frob_norm(&client, &res));
        }
    }

    #[test]
    fn test_care_scalar() {
        let (client, device) = setup();

        // Scalar CARE: A=-1, B=1, Q=2, R=1
        // Solution: X = sqrt(3) - 1
        let a = Tensor::from_slice(&[-1.0], &[1, 1], &device);
        let b = Tensor::from_slice(&[1.0], &[1, 1], &device);
        let q_mat = Tensor::from_slice(&[2.0], &[1, 1], &device);
        let r = Tensor::from_slice(&[1.0], &[1, 1], &device);

        let x = client.solve_care(&a, &b, &q_mat, &r).unwrap();
        let x_val = x.to_vec::<f64>()[0];
        let expected = 3.0f64.sqrt() - 1.0;
        eprintln!("Scalar CARE: X = {}, expected = {}", x_val, expected);
        assert!(
            (x_val - expected).abs() < 1e-10,
            "X = {}, expected {}",
            x_val,
            expected
        );
    }

    #[test]
    fn test_care_double_integrator() {
        let (client, device) = setup();

        // Double integrator: A = [[0,1],[0,0]], B = [[0],[1]]
        // Q = I, R = I
        let a = Tensor::from_slice(&[0.0, 1.0, 0.0, 0.0], &[2, 2], &device);
        let b = Tensor::from_slice(&[0.0, 1.0], &[2, 1], &device);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2], &device);
        let r = Tensor::from_slice(&[1.0], &[1, 1], &device);

        let x = client.solve_care(&a, &b, &q, &r).unwrap();
        eprintln!("CARE X = {:?}", x.to_vec::<f64>());

        // Verify CARE residual: A^T X + X A - X B R^{-1} B^T X + Q ≈ 0
        let at = a.transpose(0, 1).unwrap().contiguous();
        let bt = b.transpose(0, 1).unwrap().contiguous();
        let r_inv = Tensor::from_slice(&[1.0], &[1, 1], &device);

        let atx = client.matmul(&at, &x).unwrap();
        let xa = client.matmul(&x, &a).unwrap();
        let xb = client.matmul(&x, &b).unwrap();
        let xbr = client.matmul(&xb, &r_inv).unwrap();
        let xbrbt = client.matmul(&xbr, &bt).unwrap();
        let xbrbtx = client.matmul(&xbrbt, &x).unwrap();

        let residual = client
            .add(
                &client
                    .sub(&client.add(&atx, &xa).unwrap(), &xbrbtx)
                    .unwrap(),
                &q,
            )
            .unwrap();

        let norm = frob_norm(&client, &residual);
        assert!(norm < 1e-8, "CARE residual = {}", norm);

        // Known analytical: X should be [[sqrt(3), 1], [1, sqrt(3)]]
        let x_data = x.to_vec::<f64>();
        let sqrt3 = 3.0f64.sqrt();
        assert!(
            (x_data[0] - sqrt3).abs() < 1e-8,
            "X[0,0] = {}, expected {}",
            x_data[0],
            sqrt3
        );
        assert!(
            (x_data[3] - sqrt3).abs() < 1e-8,
            "X[1,1] = {}, expected {}",
            x_data[3],
            sqrt3
        );
    }

    // ========================================================================
    // DARE tests
    // ========================================================================

    // ========================================================================
    // Iterative solver tests
    // ========================================================================

    #[test]
    fn test_care_iterative_double_integrator() {
        let (client, device) = setup();

        let a = Tensor::from_slice(&[0.0, 1.0, 0.0, 0.0], &[2, 2], &device);
        let b = Tensor::from_slice(&[0.0, 1.0], &[2, 1], &device);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2], &device);
        let r = Tensor::from_slice(&[1.0], &[1, 1], &device);

        let x_iter = client.solve_care_iterative(&a, &b, &q, &r).unwrap();
        let x_schur = client.solve_care(&a, &b, &q, &r).unwrap();

        // Cross-check: iterative ≈ Schur-based
        let diff = client.sub(&x_iter, &x_schur).unwrap();
        let norm = frob_norm(&client, &diff);
        assert!(norm < 1e-8, "CARE iterative vs Schur diff = {}", norm);

        // Verify known solution: [[√3, 1], [1, √3]]
        let x_data = x_iter.to_vec::<f64>();
        let sqrt3 = 3.0f64.sqrt();
        assert!((x_data[0] - sqrt3).abs() < 1e-8);
        assert!((x_data[3] - sqrt3).abs() < 1e-8);
    }

    #[test]
    fn test_care_iterative_scalar() {
        let (client, device) = setup();

        let a = Tensor::from_slice(&[-1.0], &[1, 1], &device);
        let b = Tensor::from_slice(&[1.0], &[1, 1], &device);
        let q = Tensor::from_slice(&[2.0], &[1, 1], &device);
        let r = Tensor::from_slice(&[1.0], &[1, 1], &device);

        let x = client.solve_care_iterative(&a, &b, &q, &r).unwrap();
        let x_val = x.to_vec::<f64>()[0];
        let expected = 3.0f64.sqrt() - 1.0;
        assert!(
            (x_val - expected).abs() < 1e-10,
            "X = {}, expected {}",
            x_val,
            expected
        );
    }

    #[test]
    fn test_dare_iterative_discrete_lqr() {
        let (client, device) = setup();

        let a = Tensor::from_slice(&[1.0, 1.0, 0.0, 1.0], &[2, 2], &device);
        let b = Tensor::from_slice(&[0.5, 1.0], &[2, 1], &device);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2], &device);
        let r = Tensor::from_slice(&[1.0], &[1, 1], &device);

        let x_iter = client.solve_dare_iterative(&a, &b, &q, &r).unwrap();
        let x_schur = client.solve_dare(&a, &b, &q, &r).unwrap();

        // Cross-check
        let diff = client.sub(&x_iter, &x_schur).unwrap();
        let norm = frob_norm(&client, &diff);
        assert!(norm < 1e-8, "DARE iterative vs Schur diff = {}", norm);

        // Verify symmetric positive definite
        let x_data = x_iter.to_vec::<f64>();
        assert!(x_data[0] > 0.0);
        assert!(x_data[3] > 0.0);
        assert!((x_data[1] - x_data[2]).abs() < 1e-10);
    }

    #[test]
    fn test_discrete_lyapunov_iterative() {
        let (client, device) = setup();

        let a = Tensor::from_slice(&[0.5, 0.1, 0.0, 0.3], &[2, 2], &device);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2], &device);

        let x_iter = client.solve_discrete_lyapunov_iterative(&a, &q).unwrap();
        let x_bilinear = client.solve_discrete_lyapunov(&a, &q).unwrap();

        // Cross-check
        let diff = client.sub(&x_iter, &x_bilinear).unwrap();
        let norm = frob_norm(&client, &diff);
        assert!(
            norm < 1e-8,
            "Discrete Lyapunov iterative vs bilinear diff = {}",
            norm
        );

        // Verify residual: AXA^T - X + Q ≈ 0
        let at = a.transpose(0, 1).unwrap().contiguous();
        let axat = client
            .matmul(&client.matmul(&a, &x_iter).unwrap(), &at)
            .unwrap();
        let residual = client
            .add(&client.sub(&axat, &x_iter).unwrap(), &q)
            .unwrap();
        let res_norm = frob_norm(&client, &residual);
        assert!(
            res_norm < 1e-10,
            "Discrete Lyapunov iterative residual = {}",
            res_norm
        );
    }

    #[test]
    fn test_dare_debug() {
        use numr::algorithm::linalg::LinearAlgebraAlgorithms;
        use numr::ops::{MatmulOps, ShapeOps, UnaryOps, UtilityOps};
        let (client, device) = setup();

        let a = Tensor::from_slice(&[1.0, 1.0, 0.0, 1.0], &[2, 2], &device);
        let b = Tensor::from_slice(&[0.5, 1.0], &[2, 1], &device);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2], &device);
        let r = Tensor::from_slice(&[1.0], &[1, 1], &device);

        let r_inv = LinearAlgebraAlgorithms::inverse(&client, &r).unwrap();
        let bt = b.transpose(0, 1).unwrap().contiguous();
        let s = client
            .matmul(&client.matmul(&b, &r_inv).unwrap(), &bt)
            .unwrap();
        eprintln!("S = {:?}", s.to_vec::<f64>());

        let eye = client.eye(2, None, numr::dtype::DType::F64).unwrap();
        let at = a.transpose(0, 1).unwrap().contiguous();
        let neg_q = client.neg(&q).unwrap();
        let zeros = Tensor::zeros(&[2, 2], numr::dtype::DType::F64, &device);

        let l_top = client.cat(&[&a, &zeros], 1).unwrap();
        let l_bottom = client.cat(&[&neg_q, &eye], 1).unwrap();
        let l = client.cat(&[&l_top, &l_bottom], 0).unwrap();

        let m_top = client.cat(&[&eye, &s], 1).unwrap();
        let m_bottom = client.cat(&[&zeros, &at], 1).unwrap();
        let m_mat = client.cat(&[&m_top, &m_bottom], 0).unwrap();

        eprintln!("L = {:?}", l.to_vec::<f64>());
        eprintln!("M = {:?}", m_mat.to_vec::<f64>());

        let qz = client.qz_decompose(&l, &m_mat).unwrap();
        let n = 4;
        let s_data = qz.s.to_vec::<f64>();
        let t_data = qz.t.to_vec::<f64>();
        eprintln!("QZ S diag:");
        for i in 0..n {
            let row: Vec<f64> = (0..n).map(|j| s_data[i * n + j]).collect();
            eprintln!("  {:?}", row);
        }
        eprintln!("QZ T diag:");
        for i in 0..n {
            let row: Vec<f64> = (0..n).map(|j| t_data[i * n + j]).collect();
            eprintln!("  {:?}", row);
        }

        // Check eigenvalues
        let mut i = 0;
        while i < n {
            if i + 1 < n && s_data[(i + 1) * n + i].abs() > 1e-10 {
                let s11 = s_data[i * n + i];
                let s12 = s_data[i * n + i + 1];
                let s21 = s_data[(i + 1) * n + i];
                let s22 = s_data[(i + 1) * n + i + 1];
                let t11 = t_data[i * n + i];
                let t22 = t_data[(i + 1) * n + i + 1];
                let det_s = s11 * s22 - s12 * s21;
                let det_t = t11 * t22;
                eprintln!(
                    "2x2 block {},{}: det_s={}, det_t={}, |det_s/det_t|={}",
                    i,
                    i + 1,
                    det_s,
                    det_t,
                    (det_s / det_t).abs()
                );
                i += 2;
            } else {
                let alpha = s_data[i * n + i];
                let beta = t_data[i * n + i];
                if beta.abs() > 1e-15 {
                    eprintln!(
                        "1x1 block {}: lambda = {} / {} = {}",
                        i,
                        alpha,
                        beta,
                        alpha / beta
                    );
                } else {
                    eprintln!("1x1 block {}: lambda = inf (beta={})", i, beta);
                }
                i += 1;
            }
        }
    }

    #[test]
    fn test_dare_discrete_lqr() {
        let (client, device) = setup();

        // Discrete double integrator: A = [[1,1],[0,1]], B = [[0.5],[1]]
        // Q = I, R = I
        let a = Tensor::from_slice(&[1.0, 1.0, 0.0, 1.0], &[2, 2], &device);
        let b = Tensor::from_slice(&[0.5, 1.0], &[2, 1], &device);
        let q = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2], &device);
        let r = Tensor::from_slice(&[1.0], &[1, 1], &device);

        let x = client.solve_dare(&a, &b, &q, &r).unwrap();

        // Verify DARE residual: A^T X A - X - A^T X B (R + B^T X B)^{-1} B^T X A + Q ≈ 0
        let at = a.transpose(0, 1).unwrap().contiguous();
        let bt = b.transpose(0, 1).unwrap().contiguous();

        let atx = client.matmul(&at, &x).unwrap();
        let atxa = client.matmul(&atx, &a).unwrap();

        let xb = client.matmul(&x, &b).unwrap();
        let btxb = client.matmul(&bt, &xb).unwrap();
        let r_plus_btxb = client.add(&r, &btxb).unwrap();
        let inv_term = LinearAlgebraAlgorithms::inverse(&client, &r_plus_btxb).unwrap();

        let btxa = client.matmul(&bt, &client.matmul(&x, &a).unwrap()).unwrap();
        let middle = client
            .matmul(
                &client
                    .matmul(&client.matmul(&at, &xb).unwrap(), &inv_term)
                    .unwrap(),
                &btxa,
            )
            .unwrap();

        let residual = client
            .add(
                &client
                    .sub(&client.sub(&atxa, &x).unwrap(), &middle)
                    .unwrap(),
                &q,
            )
            .unwrap();

        let norm = frob_norm(&client, &residual);
        assert!(norm < 1e-8, "DARE residual = {}", norm);

        // X should be symmetric positive definite
        let x_data = x.to_vec::<f64>();
        assert!(x_data[0] > 0.0, "X[0,0] should be positive");
        assert!(x_data[3] > 0.0, "X[1,1] should be positive");
        assert!(
            (x_data[1] - x_data[2]).abs() < 1e-10,
            "X should be symmetric"
        );
    }
}
