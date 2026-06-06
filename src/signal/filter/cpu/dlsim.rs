//! CPU-only implementation of discrete-time LTI simulation.
//!
//! # Why CPU-Only?
//!
//! State-space simulation is **inherently sequential** due to the state update:
//!
//! ```text
//! x[k+1] = A·x[k] + B·u[k]
//! y[k]   = C·x[k] + D·u[k]
//! ```
//!
//! Each state x[k+1] depends on the previous state x[k]. This data dependency
//! makes parallelization impossible - GPU acceleration provides ZERO benefit.

use crate::signal::filter::traits::conversions::FilterConversions;
use crate::signal::filter::traits::lti_system::{DiscreteTimeLtiAlgorithms, DlsimResult};
use crate::signal::filter::traits::state_space::StateSpaceConversions;
use crate::signal::filter::types::{DiscreteTimeSystem, StateSpace, SystemRepresentation};
use numr::error::{Error, Result};
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl DiscreteTimeLtiAlgorithms<CpuRuntime> for CpuClient {
    fn dlsim(
        &self,
        system: &DiscreteTimeSystem<CpuRuntime>,
        u: &Tensor<CpuRuntime>,
        x0: Option<&Tensor<CpuRuntime>>,
        device: &<CpuRuntime as numr::runtime::Runtime>::Device,
    ) -> Result<DlsimResult<CpuRuntime>> {
        dlsim_impl(self, system, u, x0, device)
    }

    fn dlsim_ss(
        &self,
        ss: &StateSpace<CpuRuntime>,
        u: &Tensor<CpuRuntime>,
        x0: Option<&Tensor<CpuRuntime>>,
        device: &<CpuRuntime as numr::runtime::Runtime>::Device,
    ) -> Result<DlsimResult<CpuRuntime>> {
        dlsim_ss_impl(ss, u, x0, device)
    }
}

// ============================================================================
// Implementation Functions (CPU-only, not generic)
// ============================================================================

/// Simulate discrete-time LTI system.
fn dlsim_impl(
    client: &CpuClient,
    system: &DiscreteTimeSystem<CpuRuntime>,
    u: &Tensor<CpuRuntime>,
    x0: Option<&Tensor<CpuRuntime>>,
    device: &<CpuRuntime as numr::runtime::Runtime>::Device,
) -> Result<DlsimResult<CpuRuntime>> {
    // Convert to state-space if needed
    let ss = match &system.system {
        SystemRepresentation::StateSpace(ss) => ss.clone(),
        SystemRepresentation::TransferFunction(tf) => client.tf2ss(tf)?,
        SystemRepresentation::ZeroPoleGain(zpk) => {
            let tf = client.zpk2tf(zpk)?;
            client.tf2ss(&tf)?
        }
    };

    dlsim_ss_impl(&ss, u, x0, device)
}

/// Simulate discrete-time LTI system in state-space form.
fn dlsim_ss_impl(
    ss: &StateSpace<CpuRuntime>,
    u: &Tensor<CpuRuntime>,
    x0: Option<&Tensor<CpuRuntime>>,
    device: &<CpuRuntime as numr::runtime::Runtime>::Device,
) -> Result<DlsimResult<CpuRuntime>> {
    let n_states = ss.num_states();
    let u_data: Vec<f64> = u.to_vec();
    let n_samples = u_data.len();

    if n_samples == 0 {
        return Err(Error::InvalidArgument {
            arg: "u",
            reason: "Input signal cannot be empty".to_string(),
        });
    }

    // Handle zero-state system (just feedthrough)
    if n_states == 0 {
        let d_data: Vec<f64> = ss.d.to_vec();
        let d_val = if d_data.is_empty() { 0.0 } else { d_data[0] };

        let y: Vec<f64> = u_data.iter().map(|&ui| d_val * ui).collect();
        let t: Vec<f64> = (0..n_samples).map(|i| i as f64).collect();

        return Ok(DlsimResult {
            t: Tensor::from_slice(&t, &[n_samples], device),
            y: Tensor::from_slice(&y, &[n_samples], device),
            x: Tensor::zeros(&[n_samples, 0], u.dtype(), device),
        });
    }

    // Get state-space matrices (CPU memory - no transfer)
    let a_data: Vec<f64> = ss.a.to_vec();
    let b_data: Vec<f64> = ss.b.to_vec();
    let c_data: Vec<f64> = ss.c.to_vec();
    let d_data: Vec<f64> = ss.d.to_vec();

    // Initialize state
    let mut x = if let Some(x0_tensor) = x0 {
        let x0_data: Vec<f64> = x0_tensor.to_vec();
        if x0_data.len() != n_states {
            return Err(Error::InvalidArgument {
                arg: "x0",
                reason: format!("Initial state must have {} elements", n_states),
            });
        }
        x0_data
    } else {
        vec![0.0; n_states]
    };

    // Allocate output arrays
    let mut y_out = Vec::with_capacity(n_samples);
    let mut x_out = Vec::with_capacity(n_samples * n_states);

    // Simulate system (sequential by necessity)
    for &uk in &u_data {
        // Output: y[k] = C * x[k] + D * u[k]
        let mut yk = 0.0;
        for i in 0..n_states {
            yk += c_data[i] * x[i];
        }
        if !d_data.is_empty() {
            yk += d_data[0] * uk;
        }
        y_out.push(yk);

        // Store current state
        x_out.extend_from_slice(&x);

        // State update: x[k+1] = A * x[k] + B * u[k]
        let mut x_new = vec![0.0; n_states];
        for i in 0..n_states {
            for j in 0..n_states {
                x_new[i] += a_data[i * n_states + j] * x[j];
            }
            x_new[i] += b_data[i] * uk;
        }
        x = x_new;
    }

    // Create time indices
    let t: Vec<f64> = (0..n_samples).map(|i| i as f64).collect();

    Ok(DlsimResult {
        t: Tensor::from_slice(&t, &[n_samples], device),
        y: Tensor::from_slice(&y_out, &[n_samples], device),
        x: Tensor::from_slice(&x_out, &[n_samples, n_states], device),
    })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signal::filter::types::TransferFunction;
    use numr::runtime::cpu::CpuDevice;

    fn setup() -> (CpuClient, CpuDevice) {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        (client, device)
    }

    #[test]
    fn test_dlsim_ss_first_order() {
        let (client, device) = setup();

        // First-order system: x[k+1] = 0.5*x[k] + u[k], y[k] = x[k]
        let a = Tensor::<CpuRuntime>::from_slice(&[0.5f64], &[1, 1], &device);
        let b = Tensor::<CpuRuntime>::from_slice(&[1.0f64], &[1, 1], &device);
        let c = Tensor::<CpuRuntime>::from_slice(&[1.0f64], &[1, 1], &device);
        let d = Tensor::<CpuRuntime>::from_slice(&[0.0f64], &[1, 1], &device);
        let ss = StateSpace::new(a, b, c, d);

        // Step input
        let u = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 1.0, 1.0, 1.0, 1.0], &[5], &device);

        let result = client.dlsim_ss(&ss, &u, None, &device).unwrap();
        let y: Vec<f64> = result.y.to_vec();

        // Output: 0, 1, 1.5, 1.75, 1.875 (accumulating to 2)
        assert!((y[0] - 0.0).abs() < 1e-10);
        assert!((y[1] - 1.0).abs() < 1e-10);
        assert!((y[2] - 1.5).abs() < 1e-10);
        assert!((y[3] - 1.75).abs() < 1e-10);
        assert!((y[4] - 1.875).abs() < 1e-10);
    }

    #[test]
    fn test_dlsim_ss_impulse() {
        let (client, device) = setup();

        // Same first-order system
        let a = Tensor::<CpuRuntime>::from_slice(&[0.5f64], &[1, 1], &device);
        let b = Tensor::<CpuRuntime>::from_slice(&[1.0f64], &[1, 1], &device);
        let c = Tensor::<CpuRuntime>::from_slice(&[1.0f64], &[1, 1], &device);
        let d = Tensor::<CpuRuntime>::from_slice(&[0.0f64], &[1, 1], &device);
        let ss = StateSpace::new(a, b, c, d);

        // Impulse input
        let u = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 0.0, 0.0, 0.0, 0.0], &[5], &device);

        let result = client.dlsim_ss(&ss, &u, None, &device).unwrap();
        let y: Vec<f64> = result.y.to_vec();

        // Impulse response: 0, 1, 0.5, 0.25, 0.125
        assert!((y[0] - 0.0).abs() < 1e-10);
        assert!((y[1] - 1.0).abs() < 1e-10);
        assert!((y[2] - 0.5).abs() < 1e-10);
        assert!((y[3] - 0.25).abs() < 1e-10);
        assert!((y[4] - 0.125).abs() < 1e-10);
    }

    #[test]
    fn test_dlsim_ss_with_feedthrough() {
        let (client, device) = setup();

        // System with direct feedthrough
        let a = Tensor::<CpuRuntime>::from_slice(&[0.5f64], &[1, 1], &device);
        let b = Tensor::<CpuRuntime>::from_slice(&[1.0f64], &[1, 1], &device);
        let c = Tensor::<CpuRuntime>::from_slice(&[1.0f64], &[1, 1], &device);
        let d = Tensor::<CpuRuntime>::from_slice(&[0.5f64], &[1, 1], &device);
        let ss = StateSpace::new(a, b, c, d);

        // Impulse input
        let u = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 0.0, 0.0], &[3], &device);

        let result = client.dlsim_ss(&ss, &u, None, &device).unwrap();
        let y: Vec<f64> = result.y.to_vec();

        // With D=0.5: y[0] = 0.5*1 = 0.5
        // y[1] = 1 + 0 = 1
        // y[2] = 0.5 + 0 = 0.5
        assert!((y[0] - 0.5).abs() < 1e-10);
        assert!((y[1] - 1.0).abs() < 1e-10);
        assert!((y[2] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_dlsim_ss_second_order() {
        let (client, device) = setup();

        // Second-order oscillator
        let a = Tensor::<CpuRuntime>::from_slice(&[0.0f64, 1.0, -0.5, 1.0], &[2, 2], &device);
        let b = Tensor::<CpuRuntime>::from_slice(&[0.0f64, 1.0], &[2, 1], &device);
        let c = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 0.0], &[1, 2], &device);
        let d = Tensor::<CpuRuntime>::from_slice(&[0.0f64], &[1, 1], &device);
        let ss = StateSpace::new(a, b, c, d);

        // Impulse input
        let u = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 0.0, 0.0, 0.0, 0.0], &[5], &device);

        let result = client.dlsim_ss(&ss, &u, None, &device).unwrap();
        let y: Vec<f64> = result.y.to_vec();
        let x: Vec<f64> = result.x.to_vec();

        // Check we have state trajectory
        assert_eq!(x.len(), 10); // 5 samples * 2 states
        assert_eq!(y.len(), 5);
    }

    #[test]
    fn test_dlsim_ss_with_initial_state() {
        let (client, device) = setup();

        // First-order system
        let a = Tensor::<CpuRuntime>::from_slice(&[0.5f64], &[1, 1], &device);
        let b = Tensor::<CpuRuntime>::from_slice(&[1.0f64], &[1, 1], &device);
        let c = Tensor::<CpuRuntime>::from_slice(&[1.0f64], &[1, 1], &device);
        let d = Tensor::<CpuRuntime>::from_slice(&[0.0f64], &[1, 1], &device);
        let ss = StateSpace::new(a, b, c, d);

        // Zero input
        let u = Tensor::<CpuRuntime>::from_slice(&[0.0f64, 0.0, 0.0, 0.0], &[4], &device);

        // Initial state x0 = 2
        let x0 = Tensor::<CpuRuntime>::from_slice(&[2.0f64], &[1], &device);

        let result = client.dlsim_ss(&ss, &u, Some(&x0), &device).unwrap();
        let y: Vec<f64> = result.y.to_vec();

        // With x0=2 and u=0: y[0]=2, y[1]=1, y[2]=0.5, y[3]=0.25
        assert!((y[0] - 2.0).abs() < 1e-10);
        assert!((y[1] - 1.0).abs() < 1e-10);
        assert!((y[2] - 0.5).abs() < 1e-10);
        assert!((y[3] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_dlsim_from_transfer_function() {
        let (client, device) = setup();

        // H(z) = 1 / (1 - 0.5z^-1)
        let b = Tensor::<CpuRuntime>::from_slice(&[1.0f64], &[1], &device);
        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f64, -0.5], &[2], &device);
        let tf = TransferFunction::new(b, a);

        let system = DiscreteTimeSystem::new(SystemRepresentation::TransferFunction(tf), Some(1.0));

        // Impulse input
        let u = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 0.0, 0.0, 0.0], &[4], &device);

        let result = client.dlsim(&system, &u, None, &device).unwrap();
        let y: Vec<f64> = result.y.to_vec();

        // Impulse response should match the direct implementation
        // However, there's a one-sample delay due to state-space representation
        assert_eq!(y.len(), 4);
    }

    #[test]
    fn test_dlsim_from_zpk_matches_tf() {
        // Verify that simulating a ZPK system produces the same output as the
        // equivalent TF system. System: zero at z=0, pole at z=0.5, gain=1.
        // TF: H(z) = z / (z - 0.5) = 1 / (1 - 0.5z^-1)  [num=[1,0], den=[1,-0.5]]
        // ZPK: zeros=[], poles=[0.5], gain=1 (no zero needed for H(z) = 1/(z-0.5))
        // Use a simple pole-only system: ZPK zeros=[], poles=[0.5], gain=1
        // corresponds to TF b=[1], a=[1, -0.5].
        let (client, device) = setup();
        use crate::signal::filter::types::ZpkFilter;

        // ZPK: no zeros, one real pole at 0.5, gain=1
        let zeros_re = Tensor::<CpuRuntime>::from_slice(&[] as &[f64], &[0], &device);
        let zeros_im = Tensor::<CpuRuntime>::from_slice(&[] as &[f64], &[0], &device);
        let poles_re = Tensor::<CpuRuntime>::from_slice(&[0.5f64], &[1], &device);
        let poles_im = Tensor::<CpuRuntime>::from_slice(&[0.0f64], &[1], &device);
        let zpk = ZpkFilter::new(zeros_re, zeros_im, poles_re, poles_im, 1.0);

        let system_zpk =
            DiscreteTimeSystem::new(SystemRepresentation::ZeroPoleGain(zpk), Some(1.0));

        // Equivalent TF: b=[1], a=[1, -0.5]
        let b_tf = Tensor::<CpuRuntime>::from_slice(&[1.0f64], &[1], &device);
        let a_tf = Tensor::<CpuRuntime>::from_slice(&[1.0f64, -0.5], &[2], &device);
        let tf = TransferFunction::new(b_tf, a_tf);
        let system_tf =
            DiscreteTimeSystem::new(SystemRepresentation::TransferFunction(tf), Some(1.0));

        // Impulse input
        let u = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 0.0, 0.0, 0.0, 0.0], &[5], &device);

        let result_zpk = client.dlsim(&system_zpk, &u, None, &device).unwrap();
        let result_tf = client.dlsim(&system_tf, &u, None, &device).unwrap();

        let y_zpk: Vec<f64> = result_zpk.y.to_vec();
        let y_tf: Vec<f64> = result_tf.y.to_vec();

        assert_eq!(y_zpk.len(), y_tf.len());
        for (a, b) in y_zpk.iter().zip(y_tf.iter()) {
            assert!((a - b).abs() < 1e-10, "ZPK y={a} != TF y={b}");
        }
    }

    #[test]
    fn test_dlsim_from_zpk_complex_conjugate_poles() {
        // System with a complex-conjugate pole pair: poles at 0.5 ± 0.5i.
        // These must produce real polynomial coefficients.
        // H(z) = gain / ((z - (0.5+0.5i))(z - (0.5-0.5i)))
        //      = gain / (z^2 - z + 0.5)
        // In z^-1 form: den = [1, -1, 0.5], num = [gain]
        //
        // Verify dlsim doesn't error and produces real (finite) output.
        let (client, device) = setup();
        use crate::signal::filter::types::ZpkFilter;

        let gain = 0.5_f64;
        let zeros_re = Tensor::<CpuRuntime>::from_slice(&[] as &[f64], &[0], &device);
        let zeros_im = Tensor::<CpuRuntime>::from_slice(&[] as &[f64], &[0], &device);
        // Complex conjugate pair: 0.5 + 0.5i, 0.5 - 0.5i
        let poles_re = Tensor::<CpuRuntime>::from_slice(&[0.5f64, 0.5], &[2], &device);
        let poles_im = Tensor::<CpuRuntime>::from_slice(&[0.5f64, -0.5], &[2], &device);
        let zpk = ZpkFilter::new(zeros_re, zeros_im, poles_re, poles_im, gain);

        let system_zpk =
            DiscreteTimeSystem::new(SystemRepresentation::ZeroPoleGain(zpk), Some(1.0));

        // Impulse input
        let u = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 0.0, 0.0, 0.0, 0.0, 0.0], &[6], &device);

        let result = client.dlsim(&system_zpk, &u, None, &device).unwrap();
        let y: Vec<f64> = result.y.to_vec();

        // All outputs must be finite and real (no NaN/inf)
        for (k, &yk) in y.iter().enumerate() {
            assert!(yk.is_finite(), "y[{k}] = {yk} is not finite");
        }

        // Cross-check against equivalent TF: H(z) = 0.5 / (z^2 - z + 0.5)
        // Numerator: 0.5 (strictly proper — no direct feedthrough), denominator: z^2 - z + 0.5.
        // In descending coefficient form: b=[0.5], a=[1, -1, 0.5].
        // NOTE: b=[0.5, 0, 0] would be WRONG — that encodes 0.5*z^2/(z^2-z+0.5) which
        // is improper and has D=0.5 feedthrough. The ZPK system has no zeros, so no feedthrough.
        let b_tf = Tensor::<CpuRuntime>::from_slice(&[gain], &[1], &device);
        let a_tf = Tensor::<CpuRuntime>::from_slice(&[1.0f64, -1.0, 0.5], &[3], &device);
        let tf = TransferFunction::new(b_tf, a_tf);
        let system_tf =
            DiscreteTimeSystem::new(SystemRepresentation::TransferFunction(tf), Some(1.0));

        let u2 =
            Tensor::<CpuRuntime>::from_slice(&[1.0f64, 0.0, 0.0, 0.0, 0.0, 0.0], &[6], &device);
        let result_tf = client.dlsim(&system_tf, &u2, None, &device).unwrap();
        let y_tf: Vec<f64> = result_tf.y.to_vec();

        for (k, (&yz, &yt)) in y.iter().zip(y_tf.iter()).enumerate() {
            assert!(
                (yz - yt).abs() < 1e-9,
                "sample {k}: ZPK y={yz}, TF y={yt}, diff={}",
                (yz - yt).abs()
            );
        }
    }

    #[test]
    fn test_dlsim_time_indices() {
        let (client, device) = setup();

        let a = Tensor::<CpuRuntime>::from_slice(&[0.5f64], &[1, 1], &device);
        let b = Tensor::<CpuRuntime>::from_slice(&[1.0f64], &[1, 1], &device);
        let c = Tensor::<CpuRuntime>::from_slice(&[1.0f64], &[1, 1], &device);
        let d = Tensor::<CpuRuntime>::from_slice(&[0.0f64], &[1, 1], &device);
        let ss = StateSpace::new(a, b, c, d);

        let u = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 1.0, 1.0], &[3], &device);

        let result = client.dlsim_ss(&ss, &u, None, &device).unwrap();
        let t: Vec<f64> = result.t.to_vec();

        assert_eq!(t, vec![0.0, 1.0, 2.0]);
    }
}
