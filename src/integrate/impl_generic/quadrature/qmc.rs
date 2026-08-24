//! Quasi-Monte Carlo integration using low-discrepancy sequences.
//!
//! Provides Sobol and Halton sequences for better convergence than
//! standard Monte Carlo: O(1/n) vs O(1/√n).
//!
//! Uses numr's QuasiRandomOps for GPU-accelerated sequence generation.

use numr::dtype::DType;
use numr::error::Result;
use numr::ops::{QuasiRandomOps, ReduceOps, ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use crate::integrate::traits::{QMCOptions, QMCSequence, QuadResult};

/// Quasi-Monte Carlo integration implementation.
///
/// All computation stays on device:
/// 1. Generate low-discrepancy sequence on device via QuasiRandomOps
/// 2. Transform to integration domain via tensor ops
/// 3. Batch evaluate function on device
/// 4. Compute mean and error estimate via tensor ops
/// 5. Only transfer final scalar results
pub fn qmc_impl<R, C, F>(
    client: &C,
    f: F,
    bounds: &[(f64, f64)],
    options: &QMCOptions,
) -> Result<QuadResult<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + QuasiRandomOps<R> + ReduceOps<R> + RuntimeClient<R>,
    F: Fn(&Tensor<R>) -> Result<Tensor<R>>,
{
    if bounds.is_empty() {
        return Err(numr::error::Error::InvalidArgument {
            arg: "bounds",
            reason: "Bounds cannot be empty".to_string(),
        });
    }

    let device = client.device();
    let n_dims = bounds.len();
    let n_samples = options.n_samples;

    // Compute domain volume
    let volume: f64 = bounds.iter().map(|(a, b)| b - a).product();

    // Generate low-discrepancy sequence using numr's QuasiRandomOps (GPU-accelerated)
    let samples = match options.sequence {
        QMCSequence::Sobol => client.sobol(n_samples, n_dims, options.skip, DType::F64)?,
        QMCSequence::Halton => client.halton(n_samples, n_dims, options.skip, DType::F64)?,
    };

    // Transform to integration domain using tensor operations (stays on device)
    let x = transform_to_bounds_tensor(client, &samples, bounds)?;

    // Batch evaluate function (stays on device)
    let f_values = f(&x)?;

    // Compute mean using tensor operations (stays on device)
    let sum_tensor = client.sum(&f_values, &[0], false)?;
    let mean_tensor = client.div_scalar(&sum_tensor, n_samples as f64)?;

    // Get scalar mean
    let mean_scalar: f64 = mean_tensor.item()?;

    // Integral = volume * mean
    let integral = volume * mean_scalar;

    // QMC error estimation: Use randomized QMC or compare with halved sample
    // For deterministic QMC, estimate error by comparing full vs half estimate
    let error = if n_samples >= 2 {
        let half_n = n_samples / 2;
        let f_half = f_values.narrow(0, 0, half_n)?;
        let sum_half = client.sum(&f_half, &[0], false)?;
        let mean_half_tensor = client.div_scalar(&sum_half, half_n as f64)?;
        let mean_half: f64 = mean_half_tensor.item()?;
        let integral_half = volume * mean_half;
        (integral - integral_half).abs()
    } else {
        0.0
    };

    let integral_tensor = Tensor::<R>::from_slice(&[integral], &[1], device)?;

    Ok(QuadResult {
        integral: integral_tensor,
        error,
        neval: n_samples,
        converged: true, // QMC always produces an estimate
    })
}

/// Transform samples from [0,1]^d to integration domain using tensor operations.
///
/// For each dimension d: x_d = a_d + (b_d - a_d) * u_d
/// All operations stay on device.
fn transform_to_bounds_tensor<R, C>(
    client: &C,
    samples: &Tensor<R>,
    bounds: &[(f64, f64)],
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
    let device = client.device();
    let n_dims = bounds.len();

    // Create scale and offset tensors [1, n_dims] for broadcasting
    let scales: Vec<f64> = bounds.iter().map(|(a, b)| b - a).collect();
    let offsets: Vec<f64> = bounds.iter().map(|(a, _)| *a).collect();

    let scale_tensor = Tensor::<R>::from_slice(&scales, &[1, n_dims], device)?;
    let offset_tensor = Tensor::<R>::from_slice(&offsets, &[1, n_dims], device)?;

    // x = offset + scale * samples (broadcasting handles [1, n_dims] with [n_samples, n_dims])
    let scaled = client.mul(samples, &scale_tensor)?;
    client.add(&scaled, &offset_tensor)
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
    fn test_qmc_unit_cube() {
        let (device, client) = setup();

        // Integrate f(x,y) = 1 over [0,1]^2, exact = 1
        let result = qmc_impl(
            &client,
            |_x| {
                let shape = _x.shape();
                let n = shape[0];
                let ones = vec![1.0; n];
                Tensor::<CpuRuntime>::from_slice(&ones, &[n], &device)
            },
            &[(0.0, 1.0), (0.0, 1.0)],
            &QMCOptions::with_samples(4096),
        )
        .unwrap();

        let integral: Vec<f64> = result.integral.to_vec();
        assert!(
            (integral[0] - 1.0).abs() < 0.01,
            "integral = {}, expected 1.0",
            integral[0]
        );
    }

    #[test]
    fn test_qmc_sobol_1d() {
        let (device, client) = setup();

        // Integrate x^2 from 0 to 1, exact = 1/3
        let result = qmc_impl(
            &client,
            |x| {
                let data: Vec<f64> = x.to_vec();
                let sq: Vec<f64> = data.iter().map(|&xi| xi * xi).collect();
                Tensor::<CpuRuntime>::from_slice(&sq, &[sq.len()], &device)
            },
            &[(0.0, 1.0)],
            &QMCOptions::with_samples(4096).sequence(QMCSequence::Sobol),
        )
        .unwrap();

        let integral: Vec<f64> = result.integral.to_vec();
        assert!(
            (integral[0] - 1.0 / 3.0).abs() < 0.01,
            "integral = {}, expected 1/3",
            integral[0]
        );
    }

    #[test]
    fn test_qmc_halton_circle() {
        let (device, client) = setup();

        // Integrate indicator of unit circle over [-1,1]^2, exact = π
        let result = qmc_impl(
            &client,
            |x| {
                let data: Vec<f64> = x.to_vec();
                let n = data.len() / 2;
                let mut vals = Vec::with_capacity(n);
                for i in 0..n {
                    let xi = data[i * 2];
                    let yi = data[i * 2 + 1];
                    vals.push(if xi * xi + yi * yi <= 1.0 { 1.0 } else { 0.0 });
                }
                Tensor::<CpuRuntime>::from_slice(&vals, &[n], &device)
            },
            &[(-1.0, 1.0), (-1.0, 1.0)],
            &QMCOptions::with_samples(16384).sequence(QMCSequence::Halton),
        )
        .unwrap();

        let integral: Vec<f64> = result.integral.to_vec();
        assert!(
            (integral[0] - PI).abs() < 0.1,
            "integral = {}, expected π ≈ {}",
            integral[0],
            PI
        );
    }

    #[test]
    fn test_qmc_high_dimension() {
        let (device, client) = setup();

        // Test with 10 dimensions - numr's Sobol supports up to 21,201 dimensions
        let result = qmc_impl(
            &client,
            |x| {
                let shape = x.shape();
                let n = shape[0];
                // f(x) = 1, integral over [0,1]^10 = 1
                let ones = vec![1.0; n];
                Tensor::<CpuRuntime>::from_slice(&ones, &[n], &device)
            },
            &[(0.0, 1.0); 10],
            &QMCOptions::with_samples(8192).sequence(QMCSequence::Sobol),
        )
        .unwrap();

        let integral: Vec<f64> = result.integral.to_vec();
        assert!(
            (integral[0] - 1.0).abs() < 0.01,
            "integral = {}, expected 1.0",
            integral[0]
        );
    }

    #[test]
    fn test_qmc_deterministic() {
        let (device, client) = setup();

        // Same parameters should produce identical results (Sobol is deterministic)
        let opts = QMCOptions::with_samples(1000).sequence(QMCSequence::Sobol);

        let result1 = qmc_impl(
            &client,
            |x| {
                let shape = x.shape();
                let n = shape[0];
                let ones = vec![1.0; n];
                Tensor::<CpuRuntime>::from_slice(&ones, &[n], &device)
            },
            &[(0.0, 1.0)],
            &opts,
        )
        .unwrap();

        let result2 = qmc_impl(
            &client,
            |x| {
                let shape = x.shape();
                let n = shape[0];
                let ones = vec![1.0; n];
                Tensor::<CpuRuntime>::from_slice(&ones, &[n], &device)
            },
            &[(0.0, 1.0)],
            &opts,
        )
        .unwrap();

        let v1: Vec<f64> = result1.integral.to_vec();
        let v2: Vec<f64> = result2.integral.to_vec();
        assert_eq!(v1[0], v2[0], "QMC should be deterministic");
    }
}
