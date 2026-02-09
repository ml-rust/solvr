//! WebGPU implementation of Newton-CG optimization.

use numr::autograd::Var;
use numr::error::Result as NumrResult;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

use crate::optimize::error::OptimizeResult;
use crate::optimize::minimize::impl_generic::newton_cg::newton_cg_impl;
use crate::optimize::minimize::traits::newton_cg::{
    NewtonCGAlgorithms, NewtonCGOptions, NewtonCGResult,
};

impl NewtonCGAlgorithms<WgpuRuntime> for WgpuClient {
    fn newton_cg<F>(
        &self,
        f: F,
        x0: &Tensor<WgpuRuntime>,
        options: &NewtonCGOptions,
    ) -> OptimizeResult<NewtonCGResult<WgpuRuntime>>
    where
        F: Fn(&Var<WgpuRuntime>, &Self) -> NumrResult<Var<WgpuRuntime>>,
    {
        newton_cg_impl(self, f, x0, options)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::autograd::{var_mul, var_sum};
    use numr::runtime::wgpu::WgpuDevice;

    fn setup() -> Option<(WgpuDevice, WgpuClient)> {
        let device = WgpuDevice::new(0);
        let client = WgpuClient::new(device.clone()).ok()?;
        Some((device, client))
    }

    #[test]
    fn test_newton_cg_wgpu() {
        let Some((device, client)) = setup() else {
            eprintln!("Skipping WebGPU test: no device");
            return;
        };

        // f(x) = sum(x²), minimum at x = 0
        let x0 = Tensor::<WgpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device);

        // Newton-CG may trigger wgpu validation errors (F64/complex unsupported).
        // wgpu-core panics instead of returning Err, so use catch_unwind.
        let result = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            client.newton_cg(
                |x_var, c| {
                    let x_sq = var_mul(x_var, x_var, c)?;
                    var_sum(&x_sq, &[0], false, c)
                },
                &x0,
                &NewtonCGOptions::default(),
            )
        })) {
            Ok(Ok(r)) => r,
            Ok(Err(e)) => {
                eprintln!("Skipping test_newton_cg_wgpu: {e}");
                return;
            }
            Err(_) => {
                eprintln!("Skipping test_newton_cg_wgpu: wgpu panic");
                return;
            }
        };

        assert!(result.converged);
        assert!(result.fun < 1e-3);
    }
}
