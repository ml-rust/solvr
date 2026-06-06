//! WebGPU implementation of AdjointSensitivityAlgorithms trait.
//!
//! Pure delegation to the generic `adjoint_sensitivity_impl` in `impl_generic/`.

use numr::autograd::Var;
use numr::error::Result;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

use crate::integrate::ODEOptions;
use crate::integrate::error::IntegrateResult;
use crate::integrate::sensitivity::impl_generic::adjoint_sensitivity_impl;
use crate::integrate::sensitivity::traits::{
    AdjointSensitivityAlgorithms, SensitivityOptions, SensitivityResult,
};

impl AdjointSensitivityAlgorithms<WgpuRuntime> for WgpuClient {
    fn adjoint_sensitivity<F, G>(
        &self,
        f: F,
        g: G,
        t_span: [f64; 2],
        y0: &Tensor<WgpuRuntime>,
        p: &Tensor<WgpuRuntime>,
        ode_opts: &ODEOptions,
        sens_opts: &SensitivityOptions,
    ) -> IntegrateResult<SensitivityResult<WgpuRuntime>>
    where
        F: Fn(
            &Var<WgpuRuntime>,
            &Var<WgpuRuntime>,
            &Var<WgpuRuntime>,
            &Self,
        ) -> Result<Var<WgpuRuntime>>,
        G: Fn(&Var<WgpuRuntime>, &Self) -> Result<Var<WgpuRuntime>>,
    {
        adjoint_sensitivity_impl(self, f, g, t_span, y0, p, ode_opts, sens_opts)
    }
}
