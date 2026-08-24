//! CPU implementation of trust-NCG optimization.

use numr::autograd::Var;
use numr::error::Result as NumrResult;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

use crate::optimize::error::OptimizeResult;
use crate::optimize::minimize::impl_generic::trust_ncg_impl;
use crate::optimize::minimize::traits::trust_ncg::TrustNcgAlgorithms;
use crate::optimize::minimize::traits::trust_region::{TrustRegionOptions, TrustRegionResult};

impl TrustNcgAlgorithms<CpuRuntime> for CpuClient {
    fn trust_ncg<F>(
        &self,
        f: F,
        x0: &Tensor<CpuRuntime>,
        options: &TrustRegionOptions,
    ) -> OptimizeResult<TrustRegionResult<CpuRuntime>>
    where
        F: Fn(&Var<CpuRuntime>, &Self) -> NumrResult<Var<CpuRuntime>>,
    {
        trust_ncg_impl(self, f, x0, options)
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
    fn test_trust_ncg_cpu() {
        let (device, client) = setup();
        let x0 = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0], &[3], &device).unwrap();

        let result = client
            .trust_ncg(
                |x_var, c| {
                    let x_sq = var_mul(x_var, x_var, c)?;
                    var_sum(&x_sq, &[0], false, c)
                },
                &x0,
                &TrustRegionOptions::default(),
            )
            .unwrap();

        assert!(result.converged);
        assert!(result.fun < 1e-10);
    }
}
