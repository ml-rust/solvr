//! CPU implementation of finite element PDE solvers.

use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

use crate::pde::error::PdeResult;
use crate::pde::impl_generic::{fem_1d_impl, fem_triangular_impl};
use crate::pde::traits::FiniteElementAlgorithms;
use crate::pde::types::{BoundarySpec, FdmOptions, FemResult};

impl FiniteElementAlgorithms<CpuRuntime> for CpuClient {
    fn fem_1d(
        &self,
        f_rhs: &Tensor<CpuRuntime>,
        x_nodes: &Tensor<CpuRuntime>,
        boundary: &[BoundarySpec<CpuRuntime>],
        options: &FdmOptions,
    ) -> PdeResult<FemResult<CpuRuntime>> {
        fem_1d_impl(self, f_rhs, x_nodes, boundary, options)
    }

    fn fem_triangular(
        &self,
        f_rhs: &Tensor<CpuRuntime>,
        nodes: &Tensor<CpuRuntime>,
        elements: &Tensor<CpuRuntime>,
        boundary_nodes: &Tensor<CpuRuntime>,
        boundary_values: &Tensor<CpuRuntime>,
        options: &FdmOptions,
    ) -> PdeResult<FemResult<CpuRuntime>> {
        fem_triangular_impl(
            self,
            f_rhs,
            nodes,
            elements,
            boundary_nodes,
            boundary_values,
            options,
        )
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
    fn test_fem_1d_known_solution() {
        let (client, device) = setup();

        // -u'' = 1 on [0,1], u(0) = u(1) = 0
        // Exact: u(x) = x(1-x)/2
        let n = 21;
        let x_data: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
        let f_data = vec![1.0; n];

        let x_nodes = Tensor::<CpuRuntime>::from_slice(&x_data, &[n], &device).unwrap();
        let f_rhs = Tensor::<CpuRuntime>::from_slice(&f_data, &[n], &device).unwrap();

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
            .fem_1d(&f_rhs, &x_nodes, &boundary, &FdmOptions::default())
            .expect("FEM 1D solve failed");

        let sol: Vec<f64> = result.solution.to_vec();

        // Check at midpoint: u(0.5) = 0.125
        let mid = n / 2;
        let x_mid = x_data[mid];
        let exact = x_mid * (1.0 - x_mid) / 2.0;
        assert!(
            (sol[mid] - exact).abs() < 0.005,
            "FEM 1D midpoint: numerical={}, exact={}, error={}",
            sol[mid],
            exact,
            (sol[mid] - exact).abs()
        );
    }

    #[test]
    fn test_fem_triangular_simple() {
        let (client, device) = setup();

        // Simple triangulation of unit square with 4 nodes, 2 triangles
        // Nodes: (0,0), (1,0), (1,1), (0,1)
        let nodes_data = vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        let elements_data: Vec<i64> = vec![0, 1, 2, 0, 2, 3];

        let nodes = Tensor::<CpuRuntime>::from_slice(&nodes_data, &[4, 2], &device).unwrap();
        let elements = Tensor::<CpuRuntime>::from_slice(&elements_data, &[2, 3], &device).unwrap();

        // All nodes are boundary nodes
        let bc_nodes = Tensor::<CpuRuntime>::from_slice(&[0i64, 1, 2, 3], &[4], &device).unwrap();
        let bc_vals =
            Tensor::<CpuRuntime>::from_slice(&[0.0, 0.0, 0.0, 0.0], &[4], &device).unwrap();

        let f_data = vec![1.0; 4];
        let f_rhs = Tensor::<CpuRuntime>::from_slice(&f_data, &[4], &device).unwrap();

        let result = client
            .fem_triangular(
                &f_rhs,
                &nodes,
                &elements,
                &bc_nodes,
                &bc_vals,
                &FdmOptions::default(),
            )
            .expect("FEM triangular solve failed");

        // All boundary -> solution should be all zeros
        let sol: Vec<f64> = result.solution.to_vec();
        for &v in &sol {
            assert!(
                v.abs() < 1e-10,
                "Boundary-only solution should be zero, got {}",
                v
            );
        }
    }
}
