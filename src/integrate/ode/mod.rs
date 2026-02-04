//! Ordinary differential equation (ODE) solvers.
//!
//! This module provides types for configuring ODE solvers. The actual
//! implementations use tensor operations and are available via the
//! [`IntegrationAlgorithms`](crate::integrate::IntegrationAlgorithms) trait.
//!
//! # Available Methods
//!
//! ## Explicit Methods (Non-Stiff Problems)
//!
//! | Method | Order | Best For |
//! |--------|-------|----------|
//! | RK23 | 2(3) | Low accuracy, fast |
//! | RK45 | 4(5) | General purpose (default) |
//! | DOP853 | 8(5,3) | High accuracy |
//!
//! ## Implicit Methods (Stiff Problems)
//!
//! | Method | Order | Best For |
//! |--------|-------|----------|
//! | BDF | 1-5 | Stiff problems, chemical kinetics |
//! | Radau | 5 | Very stiff problems |
//! | LSODA | auto | Unknown stiffness |
//!
//! ## Symplectic Methods (Hamiltonian Systems)
//!
//! | Method | Order | Best For |
//! |--------|-------|----------|
//! | Verlet | 2 | Molecular dynamics |
//! | Leapfrog | 2 | N-body simulations |
//!
//! # Usage
//!
//! Use the `solve_ivp` method on a client implementing `IntegrationAlgorithms`:
//!
//! ```ignore
//! use solvr::integrate::{IntegrationAlgorithms, ODEOptions};
//! use numr::runtime::cpu::{CpuClient, CpuDevice};
//! use numr::tensor::Tensor;
//!
//! let device = CpuDevice::new();
//! let client = CpuClient::new(device.clone());
//!
//! // Solve dy/dt = -y, y(0) = 1
//! let y0 = Tensor::from_slice(&[1.0], &[1], &device);
//! let result = client.solve_ivp(
//!     |_t, y| client.mul_scalar(y, -1.0),
//!     [0.0, 5.0],
//!     &y0,
//!     &ODEOptions::default(),
//! )?;
//!
//! // y(5) ≈ exp(-5) ≈ 0.00674
//! let y_final = result.y_final_vec();
//! assert!((y_final[0] - (-5.0_f64).exp()).abs() < 1e-5);
//! ```

mod types;

pub use types::{
    BDFOptions, BVPOptions, LSODAOptions, ODEMethod, ODEOptions, RadauOptions, SymplecticOptions,
};
