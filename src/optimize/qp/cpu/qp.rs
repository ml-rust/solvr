//! CPU implementation of QP solver.

use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

use crate::optimize::error::OptimizeResult;
use crate::optimize::qp::impl_generic::{active_set_qp_impl, interior_point_qp_impl};
use crate::optimize::qp::traits::{QpAlgorithms, QpMethod, QpOptions, QpResult};

impl QpAlgorithms<CpuRuntime> for CpuClient {
    fn solve_qp(
        &self,
        q: &Tensor<CpuRuntime>,
        c: &Tensor<CpuRuntime>,
        a_eq: Option<&Tensor<CpuRuntime>>,
        b_eq: Option<&Tensor<CpuRuntime>>,
        a_ineq: Option<&Tensor<CpuRuntime>>,
        b_ineq: Option<&Tensor<CpuRuntime>>,
        options: &QpOptions,
    ) -> OptimizeResult<QpResult<CpuRuntime>> {
        match options.method {
            QpMethod::ActiveSet => {
                active_set_qp_impl(self, q, c, a_eq, b_eq, a_ineq, b_ineq, options)
            }
            QpMethod::InteriorPoint => {
                interior_point_qp_impl(self, q, c, a_eq, b_eq, a_ineq, b_ineq, options)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::CpuDevice;

    fn setup() -> (CpuDevice, CpuClient) {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        (device, client)
    }

    #[test]
    fn test_qp_unconstrained() {
        let (device, client) = setup();

        // min 0.5*x'*I*x + [-1,-1]'*x = min 0.5*(x1^2+x2^2) - x1 - x2
        // Solution: x = [1, 1], f = -1
        let q = Tensor::<CpuRuntime>::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2], &device).unwrap();
        let c = Tensor::<CpuRuntime>::from_slice(&[-1.0, -1.0], &[2], &device).unwrap();

        let result = client
            .solve_qp(&q, &c, None, None, None, None, &QpOptions::default())
            .unwrap();

        assert!(result.converged);
        let sol: Vec<f64> = result.x.to_vec();
        assert!((sol[0] - 1.0).abs() < 1e-6, "x[0] = {}", sol[0]);
        assert!((sol[1] - 1.0).abs() < 1e-6, "x[1] = {}", sol[1]);
        assert!((result.fun - (-1.0)).abs() < 1e-6, "fun = {}", result.fun);
    }

    #[test]
    fn test_qp_equality_constrained() {
        let (device, client) = setup();

        // min 0.5*(x1^2 + x2^2) s.t. x1 + x2 = 1
        // Solution: x = [0.5, 0.5], f = 0.25
        let q = Tensor::<CpuRuntime>::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2], &device).unwrap();
        let c = Tensor::<CpuRuntime>::from_slice(&[0.0, 0.0], &[2], &device).unwrap();
        let a_eq = Tensor::<CpuRuntime>::from_slice(&[1.0, 1.0], &[1, 2], &device).unwrap();
        let b_eq = Tensor::<CpuRuntime>::from_slice(&[1.0], &[1], &device).unwrap();

        let result = client
            .solve_qp(
                &q,
                &c,
                Some(&a_eq),
                Some(&b_eq),
                None,
                None,
                &QpOptions::default(),
            )
            .unwrap();

        assert!(result.converged);
        let sol: Vec<f64> = result.x.to_vec();
        assert!((sol[0] - 0.5).abs() < 1e-4, "x[0] = {}", sol[0]);
        assert!((sol[1] - 0.5).abs() < 1e-4, "x[1] = {}", sol[1]);
    }

    #[test]
    fn test_qp_equality_multiplier_sign() {
        let (device, client) = setup();

        // min 0.5*(x1^2 + x2^2) - x1 s.t. x1 + x2 = 2
        // Q*x + c + A_eq'*y = 0 gives x1 - 1 + y = 0 and x2 + y = 0.
        // With x1 + x2 = 2: 1 - 2y = 2, so y = -0.5 and x = [1.5, 0.5].
        let q = Tensor::<CpuRuntime>::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2], &device).unwrap();
        let c = Tensor::<CpuRuntime>::from_slice(&[-1.0, 0.0], &[2], &device).unwrap();
        let a_eq = Tensor::<CpuRuntime>::from_slice(&[1.0, 1.0], &[1, 2], &device).unwrap();
        let b_eq = Tensor::<CpuRuntime>::from_slice(&[2.0], &[1], &device).unwrap();

        for method in [QpMethod::ActiveSet, QpMethod::InteriorPoint] {
            let options = QpOptions {
                method,
                ..QpOptions::default()
            };
            let result = client
                .solve_qp(&q, &c, Some(&a_eq), Some(&b_eq), None, None, &options)
                .unwrap();

            assert!(result.converged, "{method:?}");
            let sol: Vec<f64> = result.x.to_vec();
            assert!((sol[0] - 1.5).abs() < 1e-8, "{method:?}: x = {:?}", sol);
            assert!((sol[1] - 0.5).abs() < 1e-8, "{method:?}: x = {:?}", sol);
            let y: Vec<f64> = result.dual_eq.unwrap().to_vec();
            assert_eq!(y.len(), 1, "{method:?}");
            assert!((y[0] + 0.5).abs() < 1e-8, "{method:?}: y = {:?}", y);
            assert!(result.dual_ineq.is_none(), "{method:?}");
        }
    }

    #[test]
    fn test_qp_equality_and_inequality_both_methods() {
        let (device, client) = setup();

        // min 0.5|x|^2 + c'x, c = (-1, -0.5, 1), over the simplex
        // x1 + x2 + x3 = 1, x >= 0. This projects -c = (1, 0.5, -1) onto it.
        // x_i = max(-c_i - t, 0) with sum 1 gives t = 0.25, x = (0.75, 0.25, 0).
        // Q x + c + A_eq' y - A' z = 0 row by row, with z1 = z2 = 0:
        //   0.75 - 1 + y = 0      -> y  = 0.25
        //   0.25 - 0.5 + y = 0    -> holds
        //   0 + 1 + y - z3 = 0    -> z3 = 1.25
        // Objective 0.5 * 0.625 - 0.875 = -0.5625.
        let q = Tensor::<CpuRuntime>::from_slice(
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            &[3, 3],
            &device,
        )
        .unwrap();
        let c = Tensor::<CpuRuntime>::from_slice(&[-1.0, -0.5, 1.0], &[3], &device).unwrap();
        let a_eq = Tensor::<CpuRuntime>::from_slice(&[1.0, 1.0, 1.0], &[1, 3], &device).unwrap();
        let b_eq = Tensor::<CpuRuntime>::from_slice(&[1.0], &[1], &device).unwrap();
        let a_ineq = Tensor::<CpuRuntime>::from_slice(
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            &[3, 3],
            &device,
        )
        .unwrap();
        let b_ineq = Tensor::<CpuRuntime>::from_slice(&[0.0, 0.0, 0.0], &[3], &device).unwrap();

        let want_x = [0.75, 0.25, 0.0];
        let want_z = [0.0, 0.0, 1.25];
        let mut solutions = Vec::new();
        for method in [QpMethod::ActiveSet, QpMethod::InteriorPoint] {
            let options = QpOptions {
                method,
                ..QpOptions::default()
            };
            let result = client
                .solve_qp(
                    &q,
                    &c,
                    Some(&a_eq),
                    Some(&b_eq),
                    Some(&a_ineq),
                    Some(&b_ineq),
                    &options,
                )
                .unwrap();
            assert!(result.converged, "{method:?}");

            let x: Vec<f64> = result.x.to_vec();
            let eq_res = x.iter().sum::<f64>() - 1.0;
            assert!(eq_res.abs() < 1e-8, "{method:?}: A_eq x - b_eq = {eq_res}");
            for (xi, wi) in x.iter().zip(want_x) {
                assert!((xi - wi).abs() < 1e-6, "{method:?}: x = {:?}", x);
                assert!(*xi >= -1e-8, "{method:?}: x = {:?}", x);
            }
            assert!(
                (result.fun + 0.5625).abs() < 1e-6,
                "{method:?}: fun = {}",
                result.fun
            );

            let y: Vec<f64> = result.dual_eq.unwrap().to_vec();
            assert!((y[0] - 0.25).abs() < 1e-6, "{method:?}: y = {:?}", y);
            let z: Vec<f64> = result.dual_ineq.unwrap().to_vec();
            for (zi, wi) in z.iter().zip(want_z) {
                assert!((zi - wi).abs() < 1e-6, "{method:?}: z = {:?}", z);
            }
            solutions.push((x, y, z));
        }

        let (xa, ya, za) = &solutions[0];
        let (xi, yi, zi) = &solutions[1];
        let close = |a: &[f64], b: &[f64]| a.iter().zip(b).all(|(u, v)| (u - v).abs() < 1e-6);
        assert!(
            close(xa, xi),
            "x: active set {:?}, interior point {:?}",
            xa,
            xi
        );
        assert!(
            close(ya, yi),
            "y: active set {:?}, interior point {:?}",
            ya,
            yi
        );
        assert!(
            close(za, zi),
            "z: active set {:?}, interior point {:?}",
            za,
            zi
        );
    }

    #[test]
    fn test_qp_inequality_constrained() {
        let (device, client) = setup();

        // min 0.5*(x1^2 + x2^2) - x1 - x2
        // s.t. x1 + x2 >= 0, x1 >= 0, x2 >= 0
        // Solution: x = [1, 1], f = -1
        let q = Tensor::<CpuRuntime>::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2], &device).unwrap();
        let c = Tensor::<CpuRuntime>::from_slice(&[-1.0, -1.0], &[2], &device).unwrap();
        let a_ineq =
            Tensor::<CpuRuntime>::from_slice(&[1.0, 1.0, 1.0, 0.0, 0.0, 1.0], &[3, 2], &device)
                .unwrap();
        let b_ineq = Tensor::<CpuRuntime>::from_slice(&[0.0, 0.0, 0.0], &[3], &device).unwrap();

        let result = client
            .solve_qp(
                &q,
                &c,
                None,
                None,
                Some(&a_ineq),
                Some(&b_ineq),
                &QpOptions::default(),
            )
            .unwrap();

        let sol: Vec<f64> = result.x.to_vec();
        assert!((sol[0] - 1.0).abs() < 0.1, "x[0] = {}", sol[0]);
        assert!((sol[1] - 1.0).abs() < 0.1, "x[1] = {}", sol[1]);
    }
}
