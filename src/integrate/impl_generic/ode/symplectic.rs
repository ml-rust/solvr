//! Symplectic integrators for Hamiltonian systems.
//!
//! Provides Störmer-Verlet and Leapfrog integrators that preserve the
//! symplectic structure of Hamiltonian dynamics.

use numr::error::Result;
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use crate::integrate::error::{IntegrateError, IntegrateResult};
use crate::integrate::ode::SymplecticOptions;
use crate::integrate::traits::SymplecticResult;

/// Störmer-Verlet symplectic integrator.
///
/// For Hamiltonian H(q, p) = T(p) + V(q) with T(p) = p²/(2m),
/// the force function returns F(q) = -∂V/∂q.
///
/// The Verlet algorithm (velocity form):
/// 1. p_{n+1/2} = p_n + (dt/2) * F(q_n)
/// 2. q_{n+1} = q_n + dt * p_{n+1/2} / m
/// 3. p_{n+1} = p_{n+1/2} + (dt/2) * F(q_{n+1})
///
/// Assumes m = 1 (or that force includes mass factor).
pub fn verlet_impl<R, C, F>(
    client: &C,
    force: F,
    t_span: [f64; 2],
    q0: &Tensor<R>,
    p0: &Tensor<R>,
    options: &SymplecticOptions,
) -> IntegrateResult<SymplecticResult<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
    F: Fn(&Tensor<R>) -> Result<Tensor<R>>,
{
    let [t_start, t_end] = t_span;

    // Determine step size and number of steps
    let (dt, n_steps) = match options.n_steps {
        Some(n) => ((t_end - t_start) / n as f64, n),
        None => {
            let n = ((t_end - t_start) / options.dt).ceil() as usize;
            (options.dt, n.max(1))
        }
    };

    // Initialize
    let mut q = q0.clone();
    let mut p = p0.clone();

    // Storage
    let mut t_values = Vec::with_capacity(n_steps + 1);
    let mut q_values = Vec::with_capacity(n_steps + 1);
    let mut p_values = Vec::with_capacity(n_steps + 1);

    t_values.push(t_start);
    q_values.push(q.clone());
    p_values.push(p.clone());

    let dt_half = dt / 2.0;

    // Main loop
    for i in 0..n_steps {
        // Step 1: p_{n+1/2} = p_n + (dt/2) * F(q_n)
        let f_q = force(&q).map_err(to_integrate_err)?;
        let dp_half = client.mul_scalar(&f_q, dt_half).map_err(to_integrate_err)?;
        let p_half = client.add(&p, &dp_half).map_err(to_integrate_err)?;

        // Step 2: q_{n+1} = q_n + dt * p_{n+1/2}
        let dq = client.mul_scalar(&p_half, dt).map_err(to_integrate_err)?;
        q = client.add(&q, &dq).map_err(to_integrate_err)?;

        // Step 3: p_{n+1} = p_{n+1/2} + (dt/2) * F(q_{n+1})
        let f_q_new = force(&q).map_err(to_integrate_err)?;
        let dp_half_new = client
            .mul_scalar(&f_q_new, dt_half)
            .map_err(to_integrate_err)?;
        p = client
            .add(&p_half, &dp_half_new)
            .map_err(to_integrate_err)?;

        // Store
        let t_new = t_start + (i + 1) as f64 * dt;
        t_values.push(t_new);
        q_values.push(q.clone());
        p_values.push(p.clone());
    }

    build_symplectic_result(client, &t_values, &q_values, &p_values)
}

/// Leapfrog symplectic integrator.
///
/// Alternative arrangement of Verlet that updates p at half-steps:
/// 1. q_{n+1/2} = q_n + (dt/2) * p_n / m
/// 2. p_{n+1} = p_n + dt * F(q_{n+1/2})
/// 3. q_{n+1} = q_{n+1/2} + (dt/2) * p_{n+1} / m
///
/// Equivalent to Verlet but with different variable arrangement.
pub fn leapfrog_impl<R, C, F>(
    client: &C,
    force: F,
    t_span: [f64; 2],
    q0: &Tensor<R>,
    p0: &Tensor<R>,
    options: &SymplecticOptions,
) -> IntegrateResult<SymplecticResult<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
    F: Fn(&Tensor<R>) -> Result<Tensor<R>>,
{
    let [t_start, t_end] = t_span;

    // Determine step size and number of steps
    let (dt, n_steps) = match options.n_steps {
        Some(n) => ((t_end - t_start) / n as f64, n),
        None => {
            let n = ((t_end - t_start) / options.dt).ceil() as usize;
            (options.dt, n.max(1))
        }
    };

    // Initialize
    let mut q = q0.clone();
    let mut p = p0.clone();

    // Storage
    let mut t_values = Vec::with_capacity(n_steps + 1);
    let mut q_values = Vec::with_capacity(n_steps + 1);
    let mut p_values = Vec::with_capacity(n_steps + 1);

    t_values.push(t_start);
    q_values.push(q.clone());
    p_values.push(p.clone());

    let dt_half = dt / 2.0;

    // Main loop
    for i in 0..n_steps {
        // Step 1: q_{n+1/2} = q_n + (dt/2) * p_n
        let dq_half = client.mul_scalar(&p, dt_half).map_err(to_integrate_err)?;
        let q_half = client.add(&q, &dq_half).map_err(to_integrate_err)?;

        // Step 2: p_{n+1} = p_n + dt * F(q_{n+1/2})
        let f_q_half = force(&q_half).map_err(to_integrate_err)?;
        let dp = client.mul_scalar(&f_q_half, dt).map_err(to_integrate_err)?;
        p = client.add(&p, &dp).map_err(to_integrate_err)?;

        // Step 3: q_{n+1} = q_{n+1/2} + (dt/2) * p_{n+1}
        let dq_half_new = client.mul_scalar(&p, dt_half).map_err(to_integrate_err)?;
        q = client
            .add(&q_half, &dq_half_new)
            .map_err(to_integrate_err)?;

        // Store
        let t_new = t_start + (i + 1) as f64 * dt;
        t_values.push(t_new);
        q_values.push(q.clone());
        p_values.push(p.clone());
    }

    build_symplectic_result(client, &t_values, &q_values, &p_values)
}

/// Build the symplectic result struct.
fn build_symplectic_result<R, C>(
    client: &C,
    t_values: &[f64],
    q_values: &[Tensor<R>],
    p_values: &[Tensor<R>],
) -> IntegrateResult<SymplecticResult<R>>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
{
    let n_steps = t_values.len();
    let t_tensor = Tensor::<R>::from_slice(t_values, &[n_steps], client.device());

    let q_refs: Vec<&Tensor<R>> = q_values.iter().collect();
    let q_tensor = client
        .stack(&q_refs, 0)
        .map_err(|e| IntegrateError::InvalidInput {
            context: format!("Failed to stack q tensors: {}", e),
        })?;

    let p_refs: Vec<&Tensor<R>> = p_values.iter().collect();
    let p_tensor = client
        .stack(&p_refs, 0)
        .map_err(|e| IntegrateError::InvalidInput {
            context: format!("Failed to stack p tensors: {}", e),
        })?;

    Ok(SymplecticResult {
        t: t_tensor,
        q: q_tensor,
        p: p_tensor,
        energy: None,
        nsteps: n_steps - 1,
    })
}

fn to_integrate_err(e: numr::error::Error) -> IntegrateError {
    IntegrateError::InvalidInput {
        context: format!("Tensor operation error: {}", e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
    use std::f64::consts::PI;

    fn setup() -> (CpuDevice, CpuClient) {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        (device, client)
    }

    #[test]
    fn test_verlet_harmonic_oscillator() {
        let (device, client) = setup();

        // Harmonic oscillator: F = -k*q with k=1, m=1
        // H = p²/2 + q²/2
        // Period T = 2π
        let q0 = Tensor::<CpuRuntime>::from_slice(&[1.0], &[1], &device);
        let p0 = Tensor::<CpuRuntime>::from_slice(&[0.0], &[1], &device);

        // Force: F(q) = -q (spring constant k=1)
        let result = verlet_impl(
            &client,
            |q| client.mul_scalar(q, -1.0),
            [0.0, 2.0 * PI],
            &q0,
            &p0,
            &SymplecticOptions {
                dt: 0.01,
                n_steps: None,
            },
        )
        .unwrap();

        // After one period, should return to initial state
        let q_final: Vec<f64> = result.q.to_vec();
        let p_final: Vec<f64> = result.p.to_vec();
        let n = q_final.len() / result.nsteps.max(1);

        let q_last = q_final[q_final.len() - n];
        let p_last = p_final[p_final.len() - n];

        assert!(
            (q_last - 1.0).abs() < 0.01,
            "q_final = {}, expected ~1.0",
            q_last
        );
        assert!(p_last.abs() < 0.01, "p_final = {}, expected ~0.0", p_last);
    }

    #[test]
    fn test_leapfrog_harmonic_oscillator() {
        let (device, client) = setup();

        let q0 = Tensor::<CpuRuntime>::from_slice(&[1.0], &[1], &device);
        let p0 = Tensor::<CpuRuntime>::from_slice(&[0.0], &[1], &device);

        let result = leapfrog_impl(
            &client,
            |q| client.mul_scalar(q, -1.0),
            [0.0, 2.0 * PI],
            &q0,
            &p0,
            &SymplecticOptions {
                dt: 0.01,
                n_steps: None,
            },
        )
        .unwrap();

        let q_final: Vec<f64> = result.q.to_vec();
        let p_final: Vec<f64> = result.p.to_vec();
        let n = q_final.len() / result.nsteps.max(1);

        let q_last = q_final[q_final.len() - n];
        let p_last = p_final[p_final.len() - n];

        assert!(
            (q_last - 1.0).abs() < 0.01,
            "q_final = {}, expected ~1.0",
            q_last
        );
        assert!(p_last.abs() < 0.01, "p_final = {}, expected ~0.0", p_last);
    }

    #[test]
    fn test_verlet_energy_conservation() {
        let (device, client) = setup();

        // Check that energy is approximately conserved over many periods
        let q0 = Tensor::<CpuRuntime>::from_slice(&[1.0], &[1], &device);
        let p0 = Tensor::<CpuRuntime>::from_slice(&[0.0], &[1], &device);

        let result = verlet_impl(
            &client,
            |q| client.mul_scalar(q, -1.0),
            [0.0, 20.0 * PI], // 10 periods
            &q0,
            &p0,
            &SymplecticOptions {
                dt: 0.01,
                n_steps: None,
            },
        )
        .unwrap();

        // Initial energy: E = p²/2 + q²/2 = 0 + 0.5 = 0.5
        let e_initial = 0.5;

        let q_final: Vec<f64> = result.q.to_vec();
        let p_final: Vec<f64> = result.p.to_vec();
        let n = q_final.len() / (result.nsteps + 1);

        let q_last = q_final[q_final.len() - n];
        let p_last = p_final[p_final.len() - n];
        let e_final = p_last * p_last / 2.0 + q_last * q_last / 2.0;

        // Energy should be conserved within about 1%
        let energy_drift = (e_final - e_initial).abs() / e_initial;
        assert!(
            energy_drift < 0.01,
            "Energy drift = {:.4}%, E_initial = {}, E_final = {}",
            energy_drift * 100.0,
            e_initial,
            e_final
        );
    }

    #[test]
    fn test_verlet_2d_kepler() {
        let (device, client) = setup();

        // 2D Kepler problem: F = -r/|r|³ (gravitational force with GM=1)
        // Circular orbit at r=1 has v=1, period T=2π
        let q0 = Tensor::<CpuRuntime>::from_slice(&[1.0, 0.0], &[2], &device);
        let p0 = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0], &[2], &device);

        let result = verlet_impl(
            &client,
            |q| {
                // F = -q / |q|³
                let q_data: Vec<f64> = q.to_vec();
                let r = (q_data[0] * q_data[0] + q_data[1] * q_data[1]).sqrt();
                let r3 = r * r * r;
                let fx = -q_data[0] / r3;
                let fy = -q_data[1] / r3;
                Ok(Tensor::<CpuRuntime>::from_slice(&[fx, fy], &[2], &device))
            },
            [0.0, 2.0 * PI],
            &q0,
            &p0,
            &SymplecticOptions {
                dt: 0.01,
                n_steps: None,
            },
        )
        .unwrap();

        // After one orbit, should return close to initial position
        let q_final: Vec<f64> = result.q.to_vec();
        let n_dof = 2;
        let n_steps = q_final.len() / n_dof;

        let qx_last = q_final[(n_steps - 1) * n_dof];
        let qy_last = q_final[(n_steps - 1) * n_dof + 1];

        assert!(
            (qx_last - 1.0).abs() < 0.1,
            "qx_final = {}, expected ~1.0",
            qx_last
        );
        assert!(qy_last.abs() < 0.1, "qy_final = {}, expected ~0.0", qy_last);
    }
}
