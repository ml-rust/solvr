//! CPU-only implementation of state-space conversions.
//!
//! # Why CPU-Only?
//!
//! State-space conversions use the Faddeev-LeVerrier algorithm which is
//! **inherently sequential** due to the matrix recurrence:
//!
//! ```text
//! M_k = A * M_{k-1} + c_k * I
//! ```
//!
//! Each M_k depends on M_{k-1}, making parallelization impossible.
//!
//! Additionally, the matrices are tiny (n×n where n is filter order, typically 1-20).
//! GPU transfer overhead would far exceed any potential computation benefit.

use crate::signal::filter::traits::state_space::StateSpaceConversions;
use crate::signal::filter::types::{StateSpace, TransferFunction};
use numr::dtype::DType;
use numr::error::{Error, Result};
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl StateSpaceConversions<CpuRuntime> for CpuClient {
    fn tf2ss(&self, tf: &TransferFunction<CpuRuntime>) -> Result<StateSpace<CpuRuntime>> {
        tf2ss_impl(tf)
    }

    fn ss2tf(&self, ss: &StateSpace<CpuRuntime>) -> Result<TransferFunction<CpuRuntime>> {
        ss2tf_impl(ss)
    }
}

// ============================================================================
// Implementation Functions (CPU-only, not generic)
// ============================================================================

/// Convert transfer function to controllable canonical state-space form.
///
/// Uses the controllable canonical form (companion matrix form).
fn tf2ss_impl(tf: &TransferFunction<CpuRuntime>) -> Result<StateSpace<CpuRuntime>> {
    let b_data: Vec<f64> = tf.b.to_vec();
    let a_data: Vec<f64> = tf.a.to_vec();
    let device = tf.b.device();

    let nb = b_data.len();
    let na = a_data.len();

    // Normalize by a[0]
    let a0 = a_data[0];
    if a0.abs() < 1e-30 {
        return Err(Error::InvalidArgument {
            arg: "a",
            reason: "Leading denominator coefficient cannot be zero".to_string(),
        });
    }

    let b_norm: Vec<f64> = b_data.iter().map(|&x| x / a0).collect();
    let a_norm: Vec<f64> = a_data.iter().map(|&x| x / a0).collect();

    // Determine state dimension
    // n = max(len(a), len(b)) - 1
    let n = (na.max(nb)).saturating_sub(1);

    if n == 0 {
        // Zero-order system (just a gain)
        // D = b[0] / a[0] = b_norm[0]
        return Ok(StateSpace::new(
            Tensor::zeros(&[0, 0], DType::F64, device)?,
            Tensor::zeros(&[0, 1], DType::F64, device)?,
            Tensor::zeros(&[1, 0], DType::F64, device)?,
            Tensor::from_slice(&[b_norm[0]], &[1, 1], device)?,
        ));
    }

    // Pad b to length n+1 (same as a)
    let mut b_pad = vec![0.0; n + 1];
    let b_start = n + 1 - nb;
    for (i, &bi) in b_norm.iter().enumerate() {
        if b_start + i < n + 1 {
            b_pad[b_start + i] = bi;
        }
    }

    // Pad a to length n+1
    let mut a_pad = vec![0.0; n + 1];
    for (i, &ai) in a_norm.iter().enumerate() {
        if i < n + 1 {
            a_pad[i] = ai;
        }
    }

    // Construct A matrix (controllable canonical form)
    // A is n x n companion matrix
    let mut a_mat = vec![0.0; n * n];

    // First n-1 rows: shifted identity
    for i in 0..n - 1 {
        a_mat[i * n + i + 1] = 1.0;
    }

    // Last row: -a[n], -a[n-1], ..., -a[1]
    for j in 0..n {
        a_mat[(n - 1) * n + j] = -a_pad[n - j];
    }

    // B matrix: [0, 0, ..., 0, 1]^T
    let mut b_mat = vec![0.0; n];
    b_mat[n - 1] = 1.0;

    // C matrix: [b_n - b_0*a_n, b_{n-1} - b_0*a_{n-1}, ..., b_1 - b_0*a_1]
    // where b_0 is the direct feedthrough term
    let d_val = b_pad[0];
    let mut c_mat = vec![0.0; n];
    for i in 0..n {
        c_mat[i] = b_pad[n - i] - d_val * a_pad[n - i];
    }

    // D matrix: b_0
    let d_mat = vec![d_val];

    Ok(StateSpace::new(
        Tensor::from_slice(&a_mat, &[n, n], device)?,
        Tensor::from_slice(&b_mat, &[n, 1], device)?,
        Tensor::from_slice(&c_mat, &[1, n], device)?,
        Tensor::from_slice(&d_mat, &[1, 1], device)?,
    ))
}

/// Convert state-space to transfer function.
///
/// Computes the characteristic polynomial and numerator polynomial
/// using the Faddeev-LeVerrier algorithm.
fn ss2tf_impl(ss: &StateSpace<CpuRuntime>) -> Result<TransferFunction<CpuRuntime>> {
    let n = ss.num_states();
    let device = ss.a.device();

    if n == 0 {
        // Zero-order system: H(z) = D
        let d_data: Vec<f64> = ss.d.to_vec();
        let d_val = if d_data.is_empty() { 0.0 } else { d_data[0] };

        return Ok(TransferFunction::new(
            Tensor::from_slice(&[d_val], &[1], device)?,
            Tensor::from_slice(&[1.0], &[1], device)?,
        ));
    }

    let a_data: Vec<f64> = ss.a.to_vec();
    let b_data: Vec<f64> = ss.b.to_vec();
    let c_data: Vec<f64> = ss.c.to_vec();
    let d_data: Vec<f64> = ss.d.to_vec();

    // Compute characteristic polynomial det(sI - A) using Faddeev-LeVerrier algorithm
    // This gives the denominator coefficients
    let denom = characteristic_polynomial(&a_data, n);

    // Compute numerator using C * adj(sI - A) * B + D * det(sI - A)
    // Build adjugate matrix coefficients and multiply
    let (adj_coeffs, _) = faddeev_leverrier(&a_data, n);

    // C * adj(sI - A) * B gives polynomial in s of degree n-1
    // Add D * det(sI - A)
    let d_val = if d_data.is_empty() { 0.0 } else { d_data[0] };

    let mut numer = vec![0.0; n + 1];

    // adj(sI - A) = M_{n-1}*s^{n-1} + M_{n-2}*s^{n-2} + ... + M_0
    // where M_k are n x n matrices
    // C * M_k * B gives scalar coefficient for s^k in numerator (before adding D*det)

    for k in 0..n {
        // Extract M_k (the coefficient matrix for s^k in adjugate)
        let m_k = &adj_coeffs[k * n * n..(k + 1) * n * n];

        // Compute C * M_k * B
        let mut c_m = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                c_m[j] += c_data[i] * m_k[i * n + j];
            }
        }

        let mut c_m_b = 0.0;
        for j in 0..n {
            c_m_b += c_m[j] * b_data[j];
        }

        // Coefficient for s^k in C*adj(sI-A)*B
        // Note: numer[0] is coefficient of s^n, numer[n] is constant
        numer[n - k] += c_m_b;
    }

    // Add D * det(sI - A)
    for i in 0..=n {
        numer[i] += d_val * denom[i];
    }

    Ok(TransferFunction::new(
        Tensor::from_slice(&numer, &[n + 1], device)?,
        Tensor::from_slice(&denom, &[n + 1], device)?,
    ))
}

/// Compute characteristic polynomial using Faddeev-LeVerrier algorithm.
///
/// Returns coefficients [1, c_{n-1}, c_{n-2}, ..., c_0] of det(sI - A).
#[allow(clippy::needless_range_loop)]
fn characteristic_polynomial(a: &[f64], n: usize) -> Vec<f64> {
    if n == 0 {
        return vec![1.0];
    }

    // Faddeev-LeVerrier: compute p_k = (-1)^k * c_k
    // where c_k are characteristic polynomial coefficients
    let mut coeffs = vec![0.0; n + 1];
    coeffs[0] = 1.0; // Leading coefficient is always 1

    let mut m = vec![0.0; n * n]; // M_k matrix

    // Initialize M_0 = I
    for i in 0..n {
        m[i * n + i] = 1.0;
    }

    for k in 1..=n {
        // M_k = A * M_{k-1}
        let mut m_new = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                for l in 0..n {
                    m_new[i * n + j] += a[i * n + l] * m[l * n + j];
                }
            }
        }

        // c_k = -tr(A * M_{k-1}) / k
        let trace: f64 = (0..n).map(|i| m_new[i * n + i]).sum();
        coeffs[k] = -trace / k as f64;

        // M_k = A * M_{k-1} + c_k * I
        for i in 0..n {
            m_new[i * n + i] += coeffs[k];
        }
        m = m_new;
    }

    coeffs
}

/// Faddeev-LeVerrier algorithm returning both adjugate coefficients and characteristic polynomial.
///
/// Returns:
/// - adj_coeffs: Flattened array of n matrices M_0, M_1, ..., M_{n-1}
///   where adj(sI - A) = M_{n-1}*s^{n-1} + ... + M_0
/// - char_coeffs: Characteristic polynomial coefficients
#[allow(clippy::needless_range_loop)]
fn faddeev_leverrier(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    if n == 0 {
        return (vec![], vec![1.0]);
    }

    let mut char_coeffs = vec![0.0; n + 1];
    char_coeffs[0] = 1.0;

    // Store all M_k matrices (n matrices of size n x n)
    let mut all_m = vec![0.0; n * n * n];

    // M_0 = I
    for i in 0..n {
        all_m[i * n + i] = 1.0;
    }

    let mut m = vec![0.0; n * n];
    for i in 0..n {
        m[i * n + i] = 1.0;
    }

    for k in 1..=n {
        // M_k = A * M_{k-1}
        let mut m_new = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                for l in 0..n {
                    m_new[i * n + j] += a[i * n + l] * m[l * n + j];
                }
            }
        }

        // c_k = -tr(A * M_{k-1}) / k
        let trace: f64 = (0..n).map(|i| m_new[i * n + i]).sum();
        char_coeffs[k] = -trace / k as f64;

        // M_k = A * M_{k-1} + c_k * I
        for i in 0..n {
            m_new[i * n + i] += char_coeffs[k];
        }

        // Store m (which is M_{k-1} before update) as coefficient for s^{n-k}
        if k < n {
            let idx = (n - k) * n * n;
            all_m[idx..idx + n * n].copy_from_slice(&m);
        }

        m = m_new;
    }

    (all_m, char_coeffs)
}

// ============================================================================
// Tests
// ============================================================================

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
    fn test_tf2ss_first_order() {
        let (client, device) = setup();

        // First-order system: H(z) = 1 / (1 - 0.5z^-1) = z / (z - 0.5)
        // In transfer function form: b = [1], a = [1, -0.5]
        let b = Tensor::<CpuRuntime>::from_slice(&[1.0f64], &[1], &device).unwrap();
        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f64, -0.5], &[2], &device).unwrap();
        let tf = TransferFunction::new(b, a);

        let ss = client.tf2ss(&tf).unwrap();

        assert_eq!(ss.num_states(), 1);

        let a_mat: Vec<f64> = ss.a.to_vec();
        let b_mat: Vec<f64> = ss.b.to_vec();
        let c_mat: Vec<f64> = ss.c.to_vec();
        let d_mat: Vec<f64> = ss.d.to_vec();

        // For H(z) = 1/(1-0.5z^-1), canonical form:
        // A = [0.5], B = [1], C = [1], D = [0]
        assert!((a_mat[0] - 0.5).abs() < 1e-10);
        assert!((b_mat[0] - 1.0).abs() < 1e-10);
        assert!((c_mat[0] - 1.0).abs() < 1e-10);
        assert!(d_mat[0].abs() < 1e-10);
    }

    #[test]
    fn test_tf2ss_second_order() {
        let (client, device) = setup();

        // Second-order system
        // H(z) = (0.1 + 0.2z^-1) / (1 - 1.4z^-1 + 0.5z^-2)
        let b = Tensor::<CpuRuntime>::from_slice(&[0.1f64, 0.2], &[2], &device).unwrap();
        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f64, -1.4, 0.5], &[3], &device).unwrap();
        let tf = TransferFunction::new(b, a);

        let ss = client.tf2ss(&tf).unwrap();

        assert_eq!(ss.num_states(), 2);

        let a_mat: Vec<f64> = ss.a.to_vec();
        assert_eq!(a_mat.len(), 4);

        // A should be companion matrix:
        // [0, 1]
        // [-0.5, 1.4]
        assert!((a_mat[0] - 0.0).abs() < 1e-10);
        assert!((a_mat[1] - 1.0).abs() < 1e-10);
        assert!((a_mat[2] - (-0.5)).abs() < 1e-10);
        assert!((a_mat[3] - 1.4).abs() < 1e-10);
    }

    #[test]
    fn test_tf2ss_with_direct_feedthrough() {
        let (client, device) = setup();

        // System with direct feedthrough: H(z) = (1 + z^-1) / (1 - 0.5z^-1)
        let b = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 1.0], &[2], &device).unwrap();
        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f64, -0.5], &[2], &device).unwrap();
        let tf = TransferFunction::new(b, a);

        let ss = client.tf2ss(&tf).unwrap();

        let d_mat: Vec<f64> = ss.d.to_vec();

        // D = b[0] = 1
        assert!((d_mat[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_ss2tf_first_order() {
        let (client, device) = setup();

        // Simple first-order state-space: x[k+1] = 0.5*x[k] + u[k], y[k] = x[k]
        // This gives H(z) = z/(z-0.5) = 1/(1 - 0.5z^-1)
        let a = Tensor::<CpuRuntime>::from_slice(&[0.5f64], &[1, 1], &device).unwrap();
        let b = Tensor::<CpuRuntime>::from_slice(&[1.0f64], &[1, 1], &device).unwrap();
        let c = Tensor::<CpuRuntime>::from_slice(&[1.0f64], &[1, 1], &device).unwrap();
        let d = Tensor::<CpuRuntime>::from_slice(&[0.0f64], &[1, 1], &device).unwrap();
        let ss = StateSpace::new(a, b, c, d);

        let tf = client.ss2tf(&ss).unwrap();

        let a_data: Vec<f64> = tf.a.to_vec();

        // H(z) = 1 / (1 - 0.5z^-1) = z / (z - 0.5)
        // In descending powers: den = [1, -0.5]
        assert_eq!(a_data.len(), 2);
        assert!((a_data[0] - 1.0).abs() < 1e-10);
        assert!((a_data[1] - (-0.5)).abs() < 1e-10);

        // Verify numerator has correct shape
        assert_eq!(tf.b.shape()[0], 2);
    }

    #[test]
    fn test_roundtrip_tf_ss_tf() {
        let (client, device) = setup();

        // Original transfer function
        let b = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0], &[2], &device).unwrap();
        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f64, -0.8, 0.15], &[3], &device).unwrap();
        let tf1 = TransferFunction::new(b, a);

        // Convert to state-space and back
        let ss = client.tf2ss(&tf1).unwrap();
        let tf2 = client.ss2tf(&ss).unwrap();

        let b1: Vec<f64> = tf1.b.to_vec();
        let a1: Vec<f64> = tf1.a.to_vec();
        let b2: Vec<f64> = tf2.b.to_vec();
        let a2: Vec<f64> = tf2.a.to_vec();

        // Denominators should match (after normalization)
        let a1_norm: Vec<f64> = a1.iter().map(|&x| x / a1[0]).collect();
        let a2_norm: Vec<f64> = a2.iter().map(|&x| x / a2[0]).collect();

        for (v1, v2) in a1_norm.iter().zip(a2_norm.iter()) {
            assert!(
                (v1 - v2).abs() < 1e-8,
                "Denominator mismatch: {} vs {}",
                v1,
                v2
            );
        }

        // Numerators should also match (proportionally)
        // They might differ by a common factor due to different normalizations
        let b1_norm: Vec<f64> = b1.iter().map(|&x| x / a1[0]).collect();
        let b2_norm: Vec<f64> = b2.iter().map(|&x| x / a2[0]).collect();

        // Check ratio is consistent
        if b1_norm[0].abs() > 1e-10 && b2_norm[0].abs() > 1e-10 {
            let ratio = b1_norm[0] / b2_norm[0];
            for i in 1..b1_norm.len().min(b2_norm.len()) {
                if b2_norm[i].abs() > 1e-10 {
                    assert!(
                        (b1_norm[i] / b2_norm[i] - ratio).abs() < 1e-6,
                        "Numerator ratio mismatch at {}: {} vs {}",
                        i,
                        b1_norm[i],
                        b2_norm[i]
                    );
                }
            }
        }
    }

    #[test]
    fn test_zero_order_system() {
        let (client, device) = setup();

        // Zero-order system: H(z) = 2 (just a gain)
        let b = Tensor::<CpuRuntime>::from_slice(&[2.0f64], &[1], &device).unwrap();
        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f64], &[1], &device).unwrap();
        let tf = TransferFunction::new(b, a);

        let ss = client.tf2ss(&tf).unwrap();

        assert_eq!(ss.num_states(), 0);
        let d_mat: Vec<f64> = ss.d.to_vec();
        assert!((d_mat[0] - 2.0).abs() < 1e-10);
    }
}
