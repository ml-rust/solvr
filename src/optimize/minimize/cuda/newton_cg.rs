//! CUDA implementation of Newton-CG optimization.

use numr::autograd::Var;
use numr::error::Result as NumrResult;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use crate::optimize::error::OptimizeResult;
use crate::optimize::minimize::impl_generic::newton_cg::newton_cg_impl;
use crate::optimize::minimize::traits::newton_cg::{
    NewtonCGAlgorithms, NewtonCGOptions, NewtonCGResult,
};

impl NewtonCGAlgorithms<CudaRuntime> for CudaClient {
    fn newton_cg<F>(
        &self,
        f: F,
        x0: &Tensor<CudaRuntime>,
        options: &NewtonCGOptions,
    ) -> OptimizeResult<NewtonCGResult<CudaRuntime>>
    where
        F: Fn(&Var<CudaRuntime>, &Self) -> NumrResult<Var<CudaRuntime>>,
    {
        newton_cg_impl(self, f, x0, options)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::autograd::{var_mul, var_sum};
    use numr::runtime::cuda::CudaDevice;

    fn setup() -> Option<(CudaDevice, CudaClient)> {
        let device = CudaDevice::new(0);
        let client = CudaClient::new(device.clone()).ok()?;
        Some((device, client))
    }

    #[test]
    fn test_newton_cg_cuda() {
        let Some((device, client)) = setup() else {
            eprintln!("Skipping CUDA test: no device");
            return;
        };

        // f(x) = sum(x²), minimum at x = 0
        let x0 = Tensor::<CudaRuntime>::from_slice(&[1.0f64, 2.0, 3.0], &[3], &device);

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
    }
}
