//! WebGPU implementation of matrix equation solvers.

#[cfg(feature = "wgpu")]
use crate::linalg::impl_generic::{
    continuous_lyapunov_impl, discrete_lyapunov_impl, solve_care_impl, solve_care_iterative_impl,
    solve_dare_impl, solve_dare_iterative_impl, solve_discrete_lyapunov_iterative_impl,
    sylvester_impl,
};
#[cfg(feature = "wgpu")]
use crate::linalg::traits::matrix_equations::MatrixEquationAlgorithms;
#[cfg(feature = "wgpu")]
use numr::error::Result;
#[cfg(feature = "wgpu")]
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
#[cfg(feature = "wgpu")]
use numr::tensor::Tensor;

#[cfg(feature = "wgpu")]
impl MatrixEquationAlgorithms<WgpuRuntime> for WgpuClient {
    fn solve_sylvester(
        &self,
        a: &Tensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
        c: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        sylvester_impl(self, a, b, c)
    }

    fn solve_continuous_lyapunov(
        &self,
        a: &Tensor<WgpuRuntime>,
        q: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        continuous_lyapunov_impl(self, a, q)
    }

    fn solve_discrete_lyapunov(
        &self,
        a: &Tensor<WgpuRuntime>,
        q: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        discrete_lyapunov_impl(self, a, q)
    }

    fn solve_care(
        &self,
        a: &Tensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
        q: &Tensor<WgpuRuntime>,
        r: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        solve_care_impl(self, a, b, q, r)
    }

    fn solve_dare(
        &self,
        a: &Tensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
        q: &Tensor<WgpuRuntime>,
        r: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        solve_dare_impl(self, a, b, q, r)
    }

    fn solve_care_iterative(
        &self,
        a: &Tensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
        q: &Tensor<WgpuRuntime>,
        r: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        solve_care_iterative_impl(self, a, b, q, r)
    }

    fn solve_dare_iterative(
        &self,
        a: &Tensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
        q: &Tensor<WgpuRuntime>,
        r: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        solve_dare_iterative_impl(self, a, b, q, r)
    }

    fn solve_discrete_lyapunov_iterative(
        &self,
        a: &Tensor<WgpuRuntime>,
        q: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        solve_discrete_lyapunov_iterative_impl(self, a, q)
    }
}
