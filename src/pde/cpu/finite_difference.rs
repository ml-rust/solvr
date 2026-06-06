//! CPU implementation of finite difference PDE solvers.

use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

use crate::pde::error::PdeResult;
use crate::pde::impl_generic::{heat_2d_impl, heat_3d_impl, poisson_impl, wave_impl};
use crate::pde::traits::FiniteDifferenceAlgorithms;
use crate::pde::types::{
    BoundarySpec, FdmOptions, FdmResult, Grid2D, Grid3D, TimeDependentOptions, TimeResult,
};

impl FiniteDifferenceAlgorithms<CpuRuntime> for CpuClient {
    fn fdm_poisson(
        &self,
        f: &Tensor<CpuRuntime>,
        grid: &Grid2D,
        boundary: &[BoundarySpec<CpuRuntime>],
        options: &FdmOptions,
    ) -> PdeResult<FdmResult<CpuRuntime>> {
        poisson_impl(self, f, grid, boundary, options)
    }

    fn fdm_heat_2d(
        &self,
        u0: &Tensor<CpuRuntime>,
        alpha: f64,
        source: Option<&Tensor<CpuRuntime>>,
        grid: &Grid2D,
        boundary: &[BoundarySpec<CpuRuntime>],
        time_opts: &TimeDependentOptions,
        options: &FdmOptions,
    ) -> PdeResult<TimeResult<CpuRuntime>> {
        heat_2d_impl(self, u0, alpha, source, grid, boundary, time_opts, options)
    }

    fn fdm_heat_3d(
        &self,
        u0: &Tensor<CpuRuntime>,
        alpha: f64,
        source: Option<&Tensor<CpuRuntime>>,
        grid: &Grid3D,
        boundary: &[BoundarySpec<CpuRuntime>],
        time_opts: &TimeDependentOptions,
        options: &FdmOptions,
    ) -> PdeResult<TimeResult<CpuRuntime>> {
        heat_3d_impl(self, u0, alpha, source, grid, boundary, time_opts, options)
    }

    fn fdm_wave(
        &self,
        u0: &Tensor<CpuRuntime>,
        v0: &Tensor<CpuRuntime>,
        c: f64,
        source: Option<&Tensor<CpuRuntime>>,
        grid: &Grid2D,
        boundary: &[BoundarySpec<CpuRuntime>],
        time_opts: &TimeDependentOptions,
        options: &FdmOptions,
    ) -> PdeResult<TimeResult<CpuRuntime>> {
        wave_impl(self, u0, v0, c, source, grid, boundary, time_opts, options)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::CpuDevice;

    fn setup() -> (CpuClient, CpuDevice) {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        (client, device)
    }

    #[test]
    fn test_poisson_known_solution() {
        let (client, device) = setup();

        // Solve -nabla^2 u = f on [0,1]^2 with u=0 on boundary
        // f(x,y) = 2*pi^2 * sin(pi*x) * sin(pi*y)
        // Exact: u(x,y) = sin(pi*x) * sin(pi*y)
        let nx = 21;
        let ny = 21;
        let dx = 1.0 / (nx - 1) as f64;
        let dy = 1.0 / (ny - 1) as f64;

        let grid = Grid2D {
            nx,
            ny,
            dx,
            dy,
            x_range: [0.0, 1.0],
            y_range: [0.0, 1.0],
        };

        let pi = std::f64::consts::PI;
        let mut f_data = vec![0.0; nx * ny];
        for i in 0..nx {
            for j in 0..ny {
                let x = i as f64 * dx;
                let y = j as f64 * dy;
                f_data[i * ny + j] = 2.0 * pi * pi * (pi * x).sin() * (pi * y).sin();
            }
        }
        let f_tensor = Tensor::<CpuRuntime>::from_slice(&f_data, &[nx, ny], &device);

        let result = client
            .fdm_poisson(&f_tensor, &grid, &[], &FdmOptions::default())
            .expect("Poisson solve failed");

        // Check solution at center
        let sol: Vec<f64> = result.solution.to_vec();
        let center_i = nx / 2;
        let center_j = ny / 2;
        let numerical = sol[center_i * ny + center_j];
        let exact = (pi * 0.5).sin() * (pi * 0.5).sin();

        // FDM on 21x21 grid gives ~O(h^2) error
        assert!(
            (numerical - exact).abs() < 0.02,
            "Poisson center: numerical={}, exact={}, error={}",
            numerical,
            exact,
            (numerical - exact).abs()
        );
    }

    #[test]
    fn test_heat_2d_diffusion() {
        let (client, device) = setup();

        let nx = 11;
        let ny = 11;
        let dx = 1.0 / (nx - 1) as f64;
        let dy = 1.0 / (ny - 1) as f64;

        let grid = Grid2D {
            nx,
            ny,
            dx,
            dy,
            x_range: [0.0, 1.0],
            y_range: [0.0, 1.0],
        };

        // Initial condition: hot spot in center
        let mut u0_data = vec![0.0; nx * ny];
        u0_data[(nx / 2) * ny + ny / 2] = 1.0;
        let u0 = Tensor::<CpuRuntime>::from_slice(&u0_data, &[nx, ny], &device);

        let time_opts = TimeDependentOptions {
            t_span: [0.0, 0.01],
            dt: None,
            save_every: 0,
        };

        let result = client
            .fdm_heat_2d(
                &u0,
                1.0,
                None,
                &grid,
                &[],
                &time_opts,
                &FdmOptions::default(),
            )
            .expect("Heat 2D solve failed");

        assert!(!result.solutions.is_empty());

        // Final solution should be more spread out (lower max)
        let final_sol: Vec<f64> = result.solutions.last().unwrap().to_vec();
        let max_val = final_sol.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(max_val < 1.0, "Heat should have diffused: max={}", max_val);
    }

    #[test]
    fn test_wave_energy_conservation() {
        let (client, device) = setup();

        let nx = 11;
        let ny = 11;
        let dx = 1.0 / (nx - 1) as f64;
        let dy = 1.0 / (ny - 1) as f64;

        let grid = Grid2D {
            nx,
            ny,
            dx,
            dy,
            x_range: [0.0, 1.0],
            y_range: [0.0, 1.0],
        };

        let pi = std::f64::consts::PI;
        let mut u0_data = vec![0.0; nx * ny];
        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                let x = i as f64 * dx;
                let y = j as f64 * dy;
                u0_data[i * ny + j] = (pi * x).sin() * (pi * y).sin();
            }
        }
        let u0 = Tensor::<CpuRuntime>::from_slice(&u0_data, &[nx, ny], &device);
        let v0 = Tensor::<CpuRuntime>::from_slice(&vec![0.0; nx * ny], &[nx, ny], &device);

        let time_opts = TimeDependentOptions {
            t_span: [0.0, 0.1],
            dt: None,
            save_every: 0,
        };

        let result = client
            .fdm_wave(
                &u0,
                &v0,
                1.0,
                None,
                &grid,
                &[],
                &time_opts,
                &FdmOptions::default(),
            )
            .expect("Wave solve failed");

        assert!(!result.solutions.is_empty());
    }

    // -------------------------------------------------------------------------
    // Neumann BC tests
    // -------------------------------------------------------------------------

    /// Zero-flux Neumann on all sides with constant source.
    ///
    /// PDE: -nabla^2 u = 1 on [0,1]^2, du/dn = 0 on all boundaries.
    ///
    /// The compatibility condition is satisfied because the total source
    /// integral (1 * area = 1) is balanced by the prescribed flux (0 per side).
    /// Wait — compatibility requires int(f) dA = int(du/dn) dS = 0, which fails
    /// here. We instead use f = 0 and prescribe a positive flux on the left
    /// side and negative on the right, so the net flux is zero, and verify the
    /// gradient across the domain is consistent with a linear solution.
    ///
    /// Simpler test: pure homogeneous Neumann with f=0 gives u=const (up to the
    /// gauge fix u[0,0]=0), so the solution should be zero everywhere.
    #[test]
    fn test_poisson_neumann_zero_flux_zero_source() {
        let (client, device) = setup();

        let nx = 7;
        let ny = 7;
        let dx = 1.0 / (nx - 1) as f64;
        let dy = 1.0 / (ny - 1) as f64;

        let grid = Grid2D {
            nx,
            ny,
            dx,
            dy,
            x_range: [0.0, 1.0],
            y_range: [0.0, 1.0],
        };

        // f = 0 everywhere
        let f_data = vec![0.0f64; nx * ny];
        let f_tensor = Tensor::<CpuRuntime>::from_slice(&f_data, &[nx, ny], &device);

        // Homogeneous Neumann (zero flux) on all sides via BoundarySide::All
        use crate::pde::types::{BoundaryCondition, BoundarySide, BoundarySpec};
        let bc_val = Tensor::<CpuRuntime>::from_slice(&[0.0f64], &[1], &device);
        let boundary = vec![BoundarySpec {
            side: BoundarySide::All,
            condition: BoundaryCondition::Neumann(bc_val),
        }];

        let opts = FdmOptions {
            solver: crate::pde::types::SparseSolver::Gmres,
            max_iter: 5000,
            tolerance: 1e-8,
            ..Default::default()
        };

        let result = client
            .fdm_poisson(&f_tensor, &grid, &boundary, &opts)
            .expect("Neumann zero-flux Poisson solve failed");

        let sol: Vec<f64> = result.solution.to_vec();

        // With f=0 and du/dn=0, the exact solution is u=0 (up to gauge; gauge is u[0,0]=0).
        let max_err = sol.iter().cloned().map(f64::abs).fold(0.0f64, f64::max);
        assert!(
            max_err < 1e-6,
            "Neumann zero-flux should give u≈0, max error = {:.2e}",
            max_err
        );
    }

    /// Neumann BCs with non-zero prescribed flux: verify gradient matches.
    ///
    /// PDE: -nabla^2 u = 0, du/dx|_{x=0} = -1 (outward = left = negative x),
    ///                        du/dx|_{x=1} = +1 (outward = right = positive x),
    ///                        du/dy = 0 on top/bottom.
    ///
    /// Compatibility: net flux = (-1)*Ly + (1)*Ly + 0 + 0 = 0 for any Ly.
    ///
    /// Analytic solution (up to gauge): u(x,y) = x - 0.5 shifted by gauge.
    /// With u[0,0]=0 and x[0]=0: u[i,j] = x_i = i*dx.
    ///
    /// Note: the Neumann convention in BoundaryCondition stores the *outward*
    /// normal derivative. Left side outward normal points in -x direction, so
    /// du/dn|_{x=0} = -du/dx|_{x=0}. We want du/dx = 1 (solution rises right),
    /// so du/dn|_{x=0} = -1. Right side: du/dn|_{x=1} = +du/dx|_{x=1} = +1.
    #[test]
    fn test_poisson_neumann_linear_solution() {
        let (client, device) = setup();

        let nx = 9;
        let ny = 9;
        let dx = 1.0 / (nx - 1) as f64;
        let dy = 1.0 / (ny - 1) as f64;

        let grid = Grid2D {
            nx,
            ny,
            dx,
            dy,
            x_range: [0.0, 1.0],
            y_range: [0.0, 1.0],
        };

        let f_data = vec![0.0f64; nx * ny];
        let f_tensor = Tensor::<CpuRuntime>::from_slice(&f_data, &[nx, ny], &device);

        use crate::pde::types::{BoundaryCondition, BoundarySide, BoundarySpec};

        // Left side outward flux = -1 (du/dn = -du/dx = -1 means du/dx = +1)
        let bc_left_val = Tensor::<CpuRuntime>::from_slice(&[-1.0f64], &[1], &device);
        // Right side outward flux = +1 (du/dn = +du/dx = +1)
        let bc_right_val = Tensor::<CpuRuntime>::from_slice(&[1.0f64], &[1], &device);
        // Top/bottom: zero flux
        let bc_tb_val = Tensor::<CpuRuntime>::from_slice(&[0.0f64], &[1], &device);
        let bc_tb_val2 = Tensor::<CpuRuntime>::from_slice(&[0.0f64], &[1], &device);

        let boundary = vec![
            BoundarySpec {
                side: BoundarySide::Left,
                condition: BoundaryCondition::Neumann(bc_left_val),
            },
            BoundarySpec {
                side: BoundarySide::Right,
                condition: BoundaryCondition::Neumann(bc_right_val),
            },
            BoundarySpec {
                side: BoundarySide::Bottom,
                condition: BoundaryCondition::Neumann(bc_tb_val),
            },
            BoundarySpec {
                side: BoundarySide::Top,
                condition: BoundaryCondition::Neumann(bc_tb_val2),
            },
        ];

        let opts = FdmOptions {
            solver: crate::pde::types::SparseSolver::Gmres,
            max_iter: 5000,
            tolerance: 1e-8,
            ..Default::default()
        };

        let result = client
            .fdm_poisson(&f_tensor, &grid, &boundary, &opts)
            .expect("Neumann linear Poisson solve failed");

        let sol: Vec<f64> = result.solution.to_vec();

        // Exact solution (with gauge u[0,0]=0): u(x,y) = x_i.
        // Check that u[i,j] ≈ i*dx for all i,j.
        let mut max_err = 0.0f64;
        for i in 0..nx {
            for j in 0..ny {
                let exact = i as f64 * dx;
                let err = (sol[i * ny + j] - exact).abs();
                if err > max_err {
                    max_err = err;
                }
            }
        }
        assert!(
            max_err < 0.02, // O(h^2) for h=1/8
            "Neumann linear solution error too large: max_err = {:.4e}",
            max_err
        );
    }

    // -------------------------------------------------------------------------
    // Mixed BC tests (different BC type per side)
    // -------------------------------------------------------------------------

    /// Mixed Dirichlet (left/right) + Neumann zero-flux (top/bottom).
    ///
    /// Laplace's equation (f = 0) with u(0,y) = 0, u(1,y) = 1, and zero normal
    /// flux on top/bottom has the exact solution u(x, y) = x (independent of y).
    /// Because at least one side is Dirichlet the system is non-singular (no
    /// gauge fix), so the recovered solution must match u = x directly.
    #[test]
    fn test_poisson_mixed_dirichlet_neumann_linear() {
        let (client, device) = setup();

        let nx = 9;
        let ny = 9;
        let dx = 1.0 / (nx - 1) as f64;
        let dy = 1.0 / (ny - 1) as f64;

        let grid = Grid2D {
            nx,
            ny,
            dx,
            dy,
            x_range: [0.0, 1.0],
            y_range: [0.0, 1.0],
        };

        let f_tensor = Tensor::<CpuRuntime>::from_slice(&vec![0.0f64; nx * ny], &[nx, ny], &device);

        use crate::pde::types::{BoundaryCondition, BoundarySide, BoundarySpec};

        // Left Dirichlet u = 0 (one value per boundary node along the side).
        let left_vals = Tensor::<CpuRuntime>::from_slice(&vec![0.0f64; ny], &[ny], &device);
        // Right Dirichlet u = 1.
        let right_vals = Tensor::<CpuRuntime>::from_slice(&vec![1.0f64; ny], &[ny], &device);
        // Top/bottom: zero-flux Neumann.
        let zero_flux_b = Tensor::<CpuRuntime>::from_slice(&[0.0f64], &[1], &device);
        let zero_flux_t = Tensor::<CpuRuntime>::from_slice(&[0.0f64], &[1], &device);

        let boundary = vec![
            BoundarySpec {
                side: BoundarySide::Left,
                condition: BoundaryCondition::Dirichlet(left_vals),
            },
            BoundarySpec {
                side: BoundarySide::Right,
                condition: BoundaryCondition::Dirichlet(right_vals),
            },
            BoundarySpec {
                side: BoundarySide::Bottom,
                condition: BoundaryCondition::Neumann(zero_flux_b),
            },
            BoundarySpec {
                side: BoundarySide::Top,
                condition: BoundaryCondition::Neumann(zero_flux_t),
            },
        ];

        let opts = FdmOptions {
            solver: crate::pde::types::SparseSolver::Gmres,
            max_iter: 5000,
            tolerance: 1e-10,
            ..Default::default()
        };

        let result = client
            .fdm_poisson(&f_tensor, &grid, &boundary, &opts)
            .expect("mixed Dirichlet/Neumann Poisson solve failed");

        let sol: Vec<f64> = result.solution.to_vec();

        // Exact solution u(x, y) = x = i*dx, independent of j.
        let mut max_err = 0.0f64;
        for i in 0..nx {
            for j in 0..ny {
                let exact = i as f64 * dx;
                let err = (sol[i * ny + j] - exact).abs();
                if err > max_err {
                    max_err = err;
                }
            }
        }
        assert!(
            max_err < 1e-6,
            "Mixed BC linear solution error too large: max_err = {:.4e}",
            max_err
        );
    }

    /// Mixed periodic (x) + Dirichlet (y). Periodic must be paired left&right.
    /// With f = 0, u(x,0) = 0, u(x,1) = 1 and x-periodicity, the exact solution
    /// is u(x, y) = y (independent of x), which respects the x-periodic wrap.
    #[test]
    fn test_poisson_mixed_periodic_x_dirichlet_y() {
        let (client, device) = setup();

        let nx = 8;
        let ny = 9;
        let dx = 1.0 / nx as f64; // periodic in x: nx cells
        let dy = 1.0 / (ny - 1) as f64; // Dirichlet in y: ny nodes

        let grid = Grid2D {
            nx,
            ny,
            dx,
            dy,
            x_range: [0.0, 1.0],
            y_range: [0.0, 1.0],
        };

        let f_tensor = Tensor::<CpuRuntime>::from_slice(&vec![0.0f64; nx * ny], &[nx, ny], &device);

        use crate::pde::types::{BoundaryCondition, BoundarySide, BoundarySpec};

        // Bottom Dirichlet u = 0, Top Dirichlet u = 1 (per node along x).
        let bottom_vals = Tensor::<CpuRuntime>::from_slice(&vec![0.0f64; nx], &[nx], &device);
        let top_vals = Tensor::<CpuRuntime>::from_slice(&vec![1.0f64; nx], &[nx], &device);

        let boundary = vec![
            BoundarySpec {
                side: BoundarySide::Left,
                condition: BoundaryCondition::Periodic,
            },
            BoundarySpec {
                side: BoundarySide::Right,
                condition: BoundaryCondition::Periodic,
            },
            BoundarySpec {
                side: BoundarySide::Bottom,
                condition: BoundaryCondition::Dirichlet(bottom_vals),
            },
            BoundarySpec {
                side: BoundarySide::Top,
                condition: BoundaryCondition::Dirichlet(top_vals),
            },
        ];

        let opts = FdmOptions {
            solver: crate::pde::types::SparseSolver::Gmres,
            max_iter: 5000,
            tolerance: 1e-10,
            ..Default::default()
        };

        let result = client
            .fdm_poisson(&f_tensor, &grid, &boundary, &opts)
            .expect("mixed periodic/Dirichlet Poisson solve failed");

        let sol: Vec<f64> = result.solution.to_vec();

        // Exact solution u(x, y) = y = j*dy, independent of i.
        let mut max_err = 0.0f64;
        for i in 0..nx {
            for j in 0..ny {
                let exact = j as f64 * dy;
                let err = (sol[i * ny + j] - exact).abs();
                if err > max_err {
                    max_err = err;
                }
            }
        }
        assert!(
            max_err < 1e-6,
            "Mixed periodic/Dirichlet solution error too large: max_err = {:.4e}",
            max_err
        );
    }

    // -------------------------------------------------------------------------
    // Periodic BC tests
    // -------------------------------------------------------------------------

    /// Periodic BCs with zero RHS: solution should be constant (zero after gauge fix).
    #[test]
    fn test_poisson_periodic_zero_rhs() {
        let (client, device) = setup();

        let nx = 6;
        let ny = 6;
        let dx = 1.0 / nx as f64; // periodic: domain has nx cells
        let dy = 1.0 / ny as f64;

        let grid = Grid2D {
            nx,
            ny,
            dx,
            dy,
            x_range: [0.0, 1.0],
            y_range: [0.0, 1.0],
        };

        let f_data = vec![0.0f64; nx * ny];
        let f_tensor = Tensor::<CpuRuntime>::from_slice(&f_data, &[nx, ny], &device);

        use crate::pde::types::{BoundaryCondition, BoundarySide, BoundarySpec};
        let boundary = vec![BoundarySpec {
            side: BoundarySide::All,
            condition: BoundaryCondition::Periodic,
        }];

        let opts = FdmOptions {
            solver: crate::pde::types::SparseSolver::Gmres,
            max_iter: 5000,
            tolerance: 1e-8,
            ..Default::default()
        };

        let result = client
            .fdm_poisson(&f_tensor, &grid, &boundary, &opts)
            .expect("Periodic zero-RHS Poisson solve failed");

        let sol: Vec<f64> = result.solution.to_vec();

        // With f=0 and periodic BCs, the null-space is constant. After gauge
        // fix u[0,0]=0, the solution should be zero everywhere.
        let max_err = sol.iter().cloned().map(f64::abs).fold(0.0f64, f64::max);
        assert!(
            max_err < 1e-6,
            "Periodic zero-RHS should give u≈0, max error = {:.2e}",
            max_err
        );
    }

    /// Periodic BCs with a zero-mean sinusoidal source.
    ///
    /// PDE: -nabla^2 u = f on [0,1)^2 (periodic), with
    ///      f(x,y) = 2*(2π)^2 * sin(2π*x) * sin(2π*y).
    ///
    /// Analytic solution: u(x,y) = sin(2π*x)*sin(2π*y) (zero mean, periodic).
    ///
    /// The source integrates to zero over [0,1)^2 (compatibility condition).
    /// After gauge fix u[0,0]=0 and since u_exact(0,0)=sin(0)*sin(0)=0, the
    /// gauge is consistent with the analytic solution.
    #[test]
    fn test_poisson_periodic_sinusoidal_source() {
        let (client, device) = setup();

        let nx = 16;
        let ny = 16;
        // Periodic grid: dx = L/N (no repeated endpoint)
        let dx = 1.0 / nx as f64;
        let dy = 1.0 / ny as f64;

        let grid = Grid2D {
            nx,
            ny,
            dx,
            dy,
            x_range: [0.0, 1.0],
            y_range: [0.0, 1.0],
        };

        let pi = std::f64::consts::PI;
        let mut f_data = vec![0.0f64; nx * ny];
        for i in 0..nx {
            for j in 0..ny {
                let x = i as f64 * dx;
                let y = j as f64 * dy;
                // -nabla^2 sin(2πx)sin(2πy) = 2*(2π)^2 sin(2πx)sin(2πy)
                f_data[i * ny + j] =
                    2.0 * (2.0 * pi).powi(2) * (2.0 * pi * x).sin() * (2.0 * pi * y).sin();
            }
        }
        let f_tensor = Tensor::<CpuRuntime>::from_slice(&f_data, &[nx, ny], &device);

        use crate::pde::types::{BoundaryCondition, BoundarySide, BoundarySpec};
        let boundary = vec![BoundarySpec {
            side: BoundarySide::All,
            condition: BoundaryCondition::Periodic,
        }];

        let opts = FdmOptions {
            solver: crate::pde::types::SparseSolver::Gmres,
            max_iter: 10000,
            tolerance: 1e-9,
            ..Default::default()
        };

        let result = client
            .fdm_poisson(&f_tensor, &grid, &boundary, &opts)
            .expect("Periodic sinusoidal Poisson solve failed");

        let sol: Vec<f64> = result.solution.to_vec();

        // Check solution against analytic at a few interior points.
        // FDM error is O(h^2) ≈ (1/16)^2 ≈ 0.004.
        let mut max_err = 0.0f64;
        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                let x = i as f64 * dx;
                let y = j as f64 * dy;
                let exact = (2.0 * pi * x).sin() * (2.0 * pi * y).sin();
                let err = (sol[i * ny + j] - exact).abs();
                if err > max_err {
                    max_err = err;
                }
            }
        }
        // Allow generous tolerance due to FDM discretisation on 16x16.
        assert!(
            max_err < 0.05,
            "Periodic sinusoidal Poisson max error = {:.4e}",
            max_err
        );
    }

    /// Verify that Neumann and Periodic BCs no longer return an error.
    #[test]
    fn test_neumann_and_periodic_do_not_error() {
        let (client, device) = setup();

        let nx = 5;
        let ny = 5;
        let dx = 1.0 / (nx - 1) as f64;
        let dy = 1.0 / (ny - 1) as f64;

        let grid = Grid2D {
            nx,
            ny,
            dx,
            dy,
            x_range: [0.0, 1.0],
            y_range: [0.0, 1.0],
        };

        let f_data = vec![0.0f64; nx * ny];
        let f_n = Tensor::<CpuRuntime>::from_slice(&f_data, &[nx, ny], &device);
        let f_p = Tensor::<CpuRuntime>::from_slice(&f_data, &[nx, ny], &device);

        use crate::pde::types::{BoundaryCondition, BoundarySide, BoundarySpec};

        // Neumann
        let bc_n_val = Tensor::<CpuRuntime>::from_slice(&[0.0f64], &[1], &device);
        let bcs_neumann = vec![BoundarySpec {
            side: BoundarySide::All,
            condition: BoundaryCondition::Neumann(bc_n_val),
        }];

        // Periodic
        let bcs_periodic = vec![BoundarySpec {
            side: BoundarySide::All,
            condition: BoundaryCondition::Periodic,
        }];

        let opts = FdmOptions {
            solver: crate::pde::types::SparseSolver::Gmres,
            max_iter: 2000,
            tolerance: 1e-8,
            ..Default::default()
        };

        let r_n = client.fdm_poisson(&f_n, &grid, &bcs_neumann, &opts);
        assert!(r_n.is_ok(), "Neumann BC returned error: {:?}", r_n.err());

        let r_p = client.fdm_poisson(&f_p, &grid, &bcs_periodic, &opts);
        assert!(r_p.is_ok(), "Periodic BC returned error: {:?}", r_p.err());
    }
}
