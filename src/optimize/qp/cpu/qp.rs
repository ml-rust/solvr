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
