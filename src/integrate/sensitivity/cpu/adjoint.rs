//! CPU implementation of AdjointSensitivityAlgorithms trait.

use numr::autograd::Var;
use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

use crate::integrate::ODEOptions;
use crate::integrate::error::IntegrateResult;
use crate::integrate::sensitivity::impl_generic::adjoint_sensitivity_impl;
use crate::integrate::sensitivity::traits::{
    AdjointSensitivityAlgorithms, SensitivityOptions, SensitivityResult,
};

impl AdjointSensitivityAlgorithms<CpuRuntime> for CpuClient {
    fn adjoint_sensitivity<F, G>(
        &self,
        f: F,
        g: G,
        t_span: [f64; 2],
        y0: &Tensor<CpuRuntime>,
        p: &Tensor<CpuRuntime>,
        ode_opts: &ODEOptions,
        sens_opts: &SensitivityOptions,
    ) -> IntegrateResult<SensitivityResult<CpuRuntime>>
    where
        F: Fn(
            &Var<CpuRuntime>,
            &Var<CpuRuntime>,
            &Var<CpuRuntime>,
            &Self,
        ) -> Result<Var<CpuRuntime>>,
        G: Fn(&Var<CpuRuntime>, &Self) -> Result<Var<CpuRuntime>>,
    {
        adjoint_sensitivity_impl(self, f, g, t_span, y0, p, ode_opts, sens_opts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::autograd::{var_mul, var_mul_scalar};
    use numr::runtime::cpu::CpuDevice;

    fn setup() -> (CpuDevice, CpuClient) {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        (device, client)
    }

    #[test]
    fn test_cpu_adjoint_exponential_decay() {
        // ODE: dy/dt = -k*y, y(0) = 1
        // Solution: y(t) = exp(-k*t)
        // Cost: J = y(T)² = exp(-2kT)
        // Analytical gradient: ∂J/∂k = -2T * exp(-2kT)
        let (device, client) = setup();

        let t_span = [0.0, 1.0];
        let y0 = Tensor::<CpuRuntime>::from_slice(&[1.0f64], &[1], &device);
        let k = Tensor::<CpuRuntime>::from_slice(&[0.5f64], &[1], &device);

        // ODE: dy/dt = -k * y
        let f = |_t: &Var<CpuRuntime>,
                 y: &Var<CpuRuntime>,
                 p: &Var<CpuRuntime>,
                 c: &CpuClient|
         -> Result<Var<CpuRuntime>> {
            let ky = var_mul(p, y, c)?;
            var_mul_scalar(&ky, -1.0, c)
        };

        // Cost: J = y²
        let g =
            |y: &Var<CpuRuntime>, c: &CpuClient| -> Result<Var<CpuRuntime>> { var_mul(y, y, c) };

        let ode_opts = ODEOptions::with_tolerances(1e-8, 1e-10);

        let sens_opts = SensitivityOptions::default()
            .with_checkpoints(10)
            .with_adjoint_tolerances(1e-6, 1e-8);

        let result = client
            .adjoint_sensitivity(f, g, t_span, &y0, &k, &ode_opts, &sens_opts)
            .unwrap();

        // Analytical values
        let k_val: f64 = 0.5;
        let t_final: f64 = 1.0;
        let y_analytical = (-k_val * t_final).exp();
        let cost_analytical = y_analytical * y_analytical;
        let grad_analytical = -2.0 * t_final * cost_analytical;

        // Check results
        let y_final_val = result.y_final.to_vec::<f64>()[0];
        let grad_val = result.gradient.to_vec::<f64>()[0];

        assert!(
            (y_final_val - y_analytical).abs() < 1e-5,
            "y_final: expected {}, got {}",
            y_analytical,
            y_final_val
        );

        assert!(
            (result.cost - cost_analytical).abs() < 1e-5,
            "cost: expected {}, got {}",
            cost_analytical,
            result.cost
        );

        // Gradient tolerance is looser due to numerical integration
        assert!(
            (grad_val - grad_analytical).abs() < 0.1 * grad_analytical.abs(),
            "gradient: expected {}, got {} (error = {}%)",
            grad_analytical,
            grad_val,
            100.0 * (grad_val - grad_analytical).abs() / grad_analytical.abs()
        );
    }
}
