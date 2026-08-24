//! CPU implementation of spectral PDE solvers.

use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

use crate::pde::error::PdeResult;
use crate::pde::impl_generic::{spectral_1d_impl, spectral_2d_impl};
use crate::pde::traits::SpectralAlgorithms;
use crate::pde::types::{BoundarySpec, SpectralResult};

impl SpectralAlgorithms<CpuRuntime> for CpuClient {
    fn spectral_1d(
        &self,
        f_rhs: &Tensor<CpuRuntime>,
        q: Option<&Tensor<CpuRuntime>>,
        n: usize,
        boundary: &[BoundarySpec<CpuRuntime>],
    ) -> PdeResult<SpectralResult<CpuRuntime>> {
        spectral_1d_impl(self, f_rhs, q, n, boundary)
    }

    fn spectral_2d(
        &self,
        f_rhs: &Tensor<CpuRuntime>,
        nx: usize,
        ny: usize,
        boundary: &[BoundarySpec<CpuRuntime>],
    ) -> PdeResult<SpectralResult<CpuRuntime>> {
        spectral_2d_impl(self, f_rhs, nx, ny, boundary)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pde::types::{BoundaryCondition, BoundarySide};
    use numr::runtime::cpu::CpuDevice;
    fn setup() -> (CpuClient, CpuDevice) {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        (client, device)
    }

    #[test]
    fn test_spectral_1d_polynomial() {
        let (client, device) = setup();

        // -u'' = 2 on [-1,1], u(-1) = u(1) = 0
        // Exact: u(x) = 1 - x^2
        let n = 16;
        let np1 = n + 1;

        let f_data = vec![2.0; np1];
        let f_rhs = Tensor::<CpuRuntime>::from_slice(&f_data, &[np1], &device).unwrap();

        let boundary = vec![
            BoundarySpec {
                side: BoundarySide::Left,
                condition: BoundaryCondition::Dirichlet(
                    Tensor::<CpuRuntime>::from_slice(&[0.0], &[1], &device).unwrap(),
                ),
            },
            BoundarySpec {
                side: BoundarySide::Right,
                condition: BoundaryCondition::Dirichlet(
                    Tensor::<CpuRuntime>::from_slice(&[0.0], &[1], &device).unwrap(),
                ),
            },
        ];

        let result = client
            .spectral_1d(&f_rhs, None, n, &boundary)
            .expect("Spectral 1D solve failed");

        let sol: Vec<f64> = result.solution.to_vec();
        let nodes: Vec<f64> = result.nodes.to_vec();

        // Check interior points against exact solution u(x) = 1 - x^2
        for i in 1..n {
            let x = nodes[i];
            let exact = 1.0 - x * x;
            assert!(
                (sol[i] - exact).abs() < 1e-8,
                "Spectral 1D at x={}: numerical={}, exact={}, error={}",
                x,
                sol[i],
                exact,
                (sol[i] - exact).abs()
            );
        }
    }

    #[test]
    fn test_spectral_2d_poisson() {
        let (client, device) = setup();

        // -nabla^2 u = f on [-1,1]^2, u=0 on boundary
        let nx = 8;
        let ny = 8;
        let npx = nx + 1;
        let npy = ny + 1;
        let total = npx * npy;

        // Simple constant RHS
        let f_data = vec![1.0; total];
        let f_rhs = Tensor::<CpuRuntime>::from_slice(&f_data, &[total], &device).unwrap();

        let boundary = vec![BoundarySpec {
            side: BoundarySide::All,
            condition: BoundaryCondition::Dirichlet(
                Tensor::<CpuRuntime>::from_slice(&[0.0], &[1], &device).unwrap(),
            ),
        }];

        let result = client
            .spectral_2d(&f_rhs, nx, ny, &boundary)
            .expect("Spectral 2D solve failed");

        let sol: Vec<f64> = result.solution.to_vec();

        // Boundary should be zero
        for j in 0..npy {
            assert!(sol[j].abs() < 1e-10, "Top boundary not zero");
            assert!(sol[nx * npy + j].abs() < 1e-10, "Bottom boundary not zero");
        }

        // Interior should be positive (Laplacian of positive function with zero BC)
        let center = (npx / 2) * npy + npy / 2;
        assert!(
            sol[center] > 0.0,
            "Center should be positive, got {}",
            sol[center]
        );
    }
}
