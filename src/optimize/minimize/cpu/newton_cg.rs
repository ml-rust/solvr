//! CPU implementation of Newton-CG optimization.

use numr::autograd::Var;
use numr::error::Result as NumrResult;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

use crate::optimize::error::OptimizeResult;
use crate::optimize::minimize::impl_generic::newton_cg::newton_cg_impl;
use crate::optimize::minimize::traits::newton_cg::{
    NewtonCGAlgorithms, NewtonCGOptions, NewtonCGResult,
};

impl NewtonCGAlgorithms<CpuRuntime> for CpuClient {
    fn newton_cg<F>(
        &self,
        f: F,
        x0: &Tensor<CpuRuntime>,
        options: &NewtonCGOptions,
    ) -> OptimizeResult<NewtonCGResult<CpuRuntime>>
    where
        F: Fn(&Var<CpuRuntime>, &Self) -> NumrResult<Var<CpuRuntime>>,
    {
        newton_cg_impl(self, f, x0, options)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::autograd::{var_mul, var_sum};
    use numr::runtime::Runtime;
    use numr::runtime::cpu::CpuDevice;

    fn setup() -> (CpuDevice, CpuClient) {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        (device, client)
    }

    #[test]
    fn test_newton_cg_quadratic() {
        let (device, client) = setup();

        // f(x) = sum(x²), minimum at x = 0
        let x0 = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0], &[3], &device);

        let result = client
            .newton_cg(
                |x_var, c| {
                    let x_sq = var_mul(x_var, x_var, c)?;
                    var_sum(&x_sq, &[0], false, c)
                },
                &x0,
                &NewtonCGOptions::default(),
            )
            .unwrap();

        assert!(result.converged);
        assert!(result.fun < 1e-10);
        assert!(result.grad_norm < 1e-6);

        let x_final: Vec<f64> = result.x.to_vec();
        for xi in x_final {
            assert!(xi.abs() < 1e-5);
        }
    }

    #[test]
    fn test_newton_cg_shifted_quadratic() {
        let (device, client) = setup();

        // f(x) = sum((x - 1)²), minimum at x = [1, 1, 1]
        let x0 = Tensor::<CpuRuntime>::from_slice(&[0.0f64, 0.0, 0.0], &[3], &device);

        let result = client
            .newton_cg(
                |x_var, c| {
                    let one = Var::new(
                        Tensor::<CpuRuntime>::from_slice(&[1.0f64, 1.0, 1.0], &[3], &device),
                        false,
                    );
                    let diff = numr::autograd::var_sub(x_var, &one, c)?;
                    let diff_sq = var_mul(&diff, &diff, c)?;
                    var_sum(&diff_sq, &[0], false, c)
                },
                &x0,
                &NewtonCGOptions::default(),
            )
            .unwrap();

        assert!(result.converged);
        assert!(result.fun < 1e-10);

        let x_final: Vec<f64> = result.x.to_vec();
        for xi in x_final {
            assert!((xi - 1.0).abs() < 1e-5);
        }
    }
}
