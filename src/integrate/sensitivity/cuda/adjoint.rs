//! CUDA implementation of AdjointSensitivityAlgorithms trait.
//!
//! Pure delegation to the generic `adjoint_sensitivity_impl` in `impl_generic/`.

use numr::autograd::Var;
use numr::error::Result;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use crate::integrate::ODEOptions;
use crate::integrate::error::IntegrateResult;
use crate::integrate::sensitivity::impl_generic::adjoint_sensitivity_impl;
use crate::integrate::sensitivity::traits::{
    AdjointSensitivityAlgorithms, SensitivityOptions, SensitivityResult,
};

impl AdjointSensitivityAlgorithms<CudaRuntime> for CudaClient {
    fn adjoint_sensitivity<F, G>(
        &self,
        f: F,
        g: G,
        t_span: [f64; 2],
        y0: &Tensor<CudaRuntime>,
        p: &Tensor<CudaRuntime>,
        ode_opts: &ODEOptions,
        sens_opts: &SensitivityOptions,
    ) -> IntegrateResult<SensitivityResult<CudaRuntime>>
    where
        F: Fn(
            &Var<CudaRuntime>,
            &Var<CudaRuntime>,
            &Var<CudaRuntime>,
            &Self,
        ) -> Result<Var<CudaRuntime>>,
        G: Fn(&Var<CudaRuntime>, &Self) -> Result<Var<CudaRuntime>>,
    {
        adjoint_sensitivity_impl(self, f, g, t_span, y0, p, ode_opts, sens_opts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integrate::sensitivity::traits::AdjointSensitivityAlgorithms;
    use numr::autograd::{var_mul, var_mul_scalar};
    use numr::runtime::cuda::CudaDevice;

    fn setup() -> Option<(CudaDevice, CudaClient)> {
        let device = CudaDevice::new(0);
        let client = CudaClient::new(device.clone()).ok()?;
        Some((device, client))
    }

    /// Adjoint sensitivity end-to-end on the GPU: dy/dt = -k*y, J = y(T)².
    /// Analytical gradient ∂J/∂k = -2T·exp(-2kT).
    #[test]
    fn test_cuda_adjoint_exponential_decay() {
        let Some((device, client)) = setup() else {
            eprintln!("skipping: no CUDA device");
            return;
        };

        let t_span = [0.0, 1.0];
        let y0 = Tensor::<CudaRuntime>::from_slice(&[1.0f64], &[1], &device);
        let k = Tensor::<CudaRuntime>::from_slice(&[0.5f64], &[1], &device);

        let f = |_t: &Var<CudaRuntime>,
                 y: &Var<CudaRuntime>,
                 p: &Var<CudaRuntime>,
                 c: &CudaClient|
         -> Result<Var<CudaRuntime>> {
            let ky = var_mul(p, y, c)?;
            var_mul_scalar(&ky, -1.0, c)
        };
        let g =
            |y: &Var<CudaRuntime>, c: &CudaClient| -> Result<Var<CudaRuntime>> { var_mul(y, y, c) };

        let ode_opts = ODEOptions::with_tolerances(1e-8, 1e-10);
        let sens_opts = SensitivityOptions::default()
            .with_checkpoints(10)
            .with_adjoint_tolerances(1e-6, 1e-8);

        let result = client
            .adjoint_sensitivity(f, g, t_span, &y0, &k, &ode_opts, &sens_opts)
            .expect("CUDA adjoint sensitivity should succeed");

        let k_val: f64 = 0.5;
        let t_final: f64 = 1.0;
        let y_analytical = (-k_val * t_final).exp();
        let cost_analytical = y_analytical * y_analytical;
        let grad_analytical = -2.0 * t_final * cost_analytical;

        let grad_val = result.gradient.to_vec::<f64>()[0];
        assert!(
            (grad_val - grad_analytical).abs() < 0.1 * grad_analytical.abs(),
            "CUDA adjoint gradient: expected {}, got {}",
            grad_analytical,
            grad_val
        );
    }
}
