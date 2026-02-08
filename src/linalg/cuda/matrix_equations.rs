//! CUDA implementation of matrix equation solvers.

#[cfg(feature = "cuda")]
use crate::linalg::impl_generic::{
    continuous_lyapunov_impl, discrete_lyapunov_impl, solve_care_impl, solve_care_iterative_impl,
    solve_dare_impl, solve_dare_iterative_impl, solve_discrete_lyapunov_iterative_impl,
    sylvester_impl,
};
#[cfg(feature = "cuda")]
use crate::linalg::traits::matrix_equations::MatrixEquationAlgorithms;
#[cfg(feature = "cuda")]
use numr::error::Result;
#[cfg(feature = "cuda")]
use numr::runtime::cuda::{CudaClient, CudaRuntime};
#[cfg(feature = "cuda")]
use numr::tensor::Tensor;

#[cfg(feature = "cuda")]
impl MatrixEquationAlgorithms<CudaRuntime> for CudaClient {
    fn solve_sylvester(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
        c: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        sylvester_impl(self, a, b, c)
    }

    fn solve_continuous_lyapunov(
        &self,
        a: &Tensor<CudaRuntime>,
        q: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        continuous_lyapunov_impl(self, a, q)
    }

    fn solve_discrete_lyapunov(
        &self,
        a: &Tensor<CudaRuntime>,
        q: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        discrete_lyapunov_impl(self, a, q)
    }

    fn solve_care(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
        q: &Tensor<CudaRuntime>,
        r: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        solve_care_impl(self, a, b, q, r)
    }

    fn solve_dare(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
        q: &Tensor<CudaRuntime>,
        r: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        solve_dare_impl(self, a, b, q, r)
    }

    fn solve_care_iterative(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
        q: &Tensor<CudaRuntime>,
        r: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        solve_care_iterative_impl(self, a, b, q, r)
    }

    fn solve_dare_iterative(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
        q: &Tensor<CudaRuntime>,
        r: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        solve_dare_iterative_impl(self, a, b, q, r)
    }

    fn solve_discrete_lyapunov_iterative(
        &self,
        a: &Tensor<CudaRuntime>,
        q: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        solve_discrete_lyapunov_iterative_impl(self, a, q)
    }
}
