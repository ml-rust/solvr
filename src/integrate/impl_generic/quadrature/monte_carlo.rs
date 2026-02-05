//! Monte Carlo integration for multi-dimensional integrals.
//!
//! Provides plain, stratified, and antithetic variance reduction methods.
//! GPU-accelerated: all sample generation and evaluation stays on device
//! using numr's RandomOps for device-native random number generation.
//!
//! Supports reproducibility via optional seed parameter using AdvancedRandomOps
//! (Philox PRNG for deterministic parallel random generation).

use numr::dtype::DType;
use numr::error::Result;
use numr::ops::{AdvancedRandomOps, RandomOps, ReduceOps, ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use crate::integrate::traits::{MonteCarloMethod, MonteCarloOptions, MonteCarloResult};

/// Monte Carlo integration implementation.
///
/// All computation stays on device:
/// 1. Generate random samples on device via RandomOps
/// 2. Transform to integration domain via tensor ops
/// 3. Batch evaluate function on device
/// 4. Compute mean and variance via tensor ops
/// 5. Only transfer final scalar results
pub fn monte_carlo_impl<R, C, F>(
    client: &C,
    f: F,
    bounds: &[(f64, f64)],
    options: &MonteCarloOptions,
) -> Result<MonteCarloResult<R>>
where
    R: Runtime,
    C: TensorOps<R>
        + ScalarOps<R>
        + RandomOps<R>
        + AdvancedRandomOps<R>
        + ReduceOps<R>
        + RuntimeClient<R>,
    F: Fn(&Tensor<R>) -> Result<Tensor<R>>,
{
    if bounds.is_empty() {
        return Err(numr::error::Error::InvalidArgument {
            arg: "bounds",
            reason: "Bounds cannot be empty".to_string(),
        });
    }

    let n_samples = options.n_samples;

    // Compute domain volume
    let volume: f64 = bounds.iter().map(|(a, b)| b - a).product();

    match options.method {
        MonteCarloMethod::Plain => {
            plain_monte_carlo(client, &f, bounds, n_samples, volume, options.seed)
        }
        MonteCarloMethod::Stratified { n_strata } => stratified_monte_carlo(
            client,
            &f,
            bounds,
            n_samples,
            volume,
            n_strata,
            options.seed,
        ),
        MonteCarloMethod::Antithetic => {
            antithetic_monte_carlo(client, &f, bounds, n_samples, volume, options.seed)
        }
    }
}

/// Plain Monte Carlo with uniform random sampling.
///
/// Uses numr's RandomOps for GPU-accelerated random generation.
/// When seed is provided, uses AdvancedRandomOps (Philox) for reproducibility.
fn plain_monte_carlo<R, C, F>(
    client: &C,
    f: &F,
    bounds: &[(f64, f64)],
    n_samples: usize,
    volume: f64,
    seed: Option<u64>,
) -> Result<MonteCarloResult<R>>
where
    R: Runtime,
    C: TensorOps<R>
        + ScalarOps<R>
        + RandomOps<R>
        + AdvancedRandomOps<R>
        + ReduceOps<R>
        + RuntimeClient<R>,
    F: Fn(&Tensor<R>) -> Result<Tensor<R>>,
{
    let device = client.device();
    let n_dims = bounds.len();

    // Generate uniform samples in [0, 1]^d
    // Use seeded Philox PRNG for reproducibility if seed is provided
    let samples = match seed {
        Some(s) => client.philox_uniform(&[n_samples, n_dims], s, 0, DType::F64)?,
        None => client.rand(&[n_samples, n_dims], DType::F64)?,
    };

    // Transform to integration domain using tensor operations (stays on device)
    let x = transform_to_bounds_tensor(client, &samples, bounds)?;

    // Batch evaluate function (stays on device)
    let f_values = f(&x)?;

    // Compute mean and variance using tensor operations (stays on device)
    let (mean_scalar, variance_scalar) = compute_mean_variance_tensor(client, &f_values)?;

    // Standard error = sqrt(variance / n) * volume
    let std_error = (variance_scalar / n_samples as f64).sqrt() * volume;

    // Integral = volume * mean
    let integral = volume * mean_scalar;
    let integral_tensor = Tensor::<R>::from_slice(&[integral], &[1], device);

    Ok(MonteCarloResult {
        integral: integral_tensor,
        std_error,
        n_samples,
    })
}

/// Stratified Monte Carlo sampling.
///
/// Divides the domain into strata and samples uniformly within each stratum.
/// Uses numr's RandomOps for GPU-accelerated random generation.
/// When seed is provided, uses AdvancedRandomOps (Philox) for reproducibility.
fn stratified_monte_carlo<R, C, F>(
    client: &C,
    f: &F,
    bounds: &[(f64, f64)],
    n_samples: usize,
    volume: f64,
    n_strata_per_dim: usize,
    seed: Option<u64>,
) -> Result<MonteCarloResult<R>>
where
    R: Runtime,
    C: TensorOps<R>
        + ScalarOps<R>
        + RandomOps<R>
        + AdvancedRandomOps<R>
        + ReduceOps<R>
        + RuntimeClient<R>,
    F: Fn(&Tensor<R>) -> Result<Tensor<R>>,
{
    let device = client.device();
    let n_dims = bounds.len();

    // Total number of strata
    let total_strata = n_strata_per_dim.pow(n_dims as u32);

    // Samples per stratum (at least 1)
    let samples_per_stratum = (n_samples / total_strata).max(1);
    let actual_samples = samples_per_stratum * total_strata;

    // Generate all random samples at once using RandomOps (GPU-accelerated)
    // Use seeded Philox PRNG for reproducibility if seed is provided
    // Shape: [actual_samples, n_dims] with values in [0, 1)
    let rand_samples = match seed {
        Some(s) => client.philox_uniform(&[actual_samples, n_dims], s, 0, DType::F64)?,
        None => client.rand(&[actual_samples, n_dims], DType::F64)?,
    };

    // Build stratum offsets and scales
    // For stratified sampling: x = (stratum_idx + u) / n_strata where u ~ U(0,1)
    let stratum_size = 1.0 / n_strata_per_dim as f64;

    // Create offset tensor for each sample based on its stratum
    let mut offsets = Vec::with_capacity(actual_samples * n_dims);
    for stratum_idx in 0..total_strata {
        // Convert stratum index to multi-index
        let mut idx = stratum_idx;
        for _ in 0..n_dims {
            let stratum_d = idx % n_strata_per_dim;
            idx /= n_strata_per_dim;
            let offset = stratum_d as f64 * stratum_size;
            // Repeat for each sample in this stratum
            for _ in 0..samples_per_stratum {
                offsets.push(offset);
            }
        }
    }

    // Reshape offsets to match sample layout [actual_samples, n_dims]
    // The offsets vec is currently laid out wrong, let me fix this
    let mut offsets_correct = Vec::with_capacity(actual_samples * n_dims);
    for sample_idx in 0..actual_samples {
        let stratum_idx = sample_idx / samples_per_stratum;
        let mut idx = stratum_idx;
        for _ in 0..n_dims {
            let stratum_d = idx % n_strata_per_dim;
            idx /= n_strata_per_dim;
            offsets_correct.push(stratum_d as f64 * stratum_size);
        }
    }

    let offset_tensor =
        Tensor::<R>::from_slice(&offsets_correct, &[actual_samples, n_dims], device);

    // Stratified samples in [0,1]^d: offset + rand * stratum_size
    let scaled_rand = client.mul_scalar(&rand_samples, stratum_size)?;
    let stratified_samples = client.add(&offset_tensor, &scaled_rand)?;

    // Transform to integration domain
    let x = transform_to_bounds_tensor(client, &stratified_samples, bounds)?;

    // Evaluate function
    let f_values = f(&x)?;

    // Compute mean and variance
    let (mean_scalar, variance_scalar) = compute_mean_variance_tensor(client, &f_values)?;

    let std_error = (variance_scalar / actual_samples as f64).sqrt() * volume;
    let integral = volume * mean_scalar;
    let integral_tensor = Tensor::<R>::from_slice(&[integral], &[1], device);

    Ok(MonteCarloResult {
        integral: integral_tensor,
        std_error,
        n_samples: actual_samples,
    })
}

/// Antithetic variates Monte Carlo.
///
/// Uses pairs (x, 1-x) to reduce variance for monotonic functions.
/// All operations stay on device using tensor ops.
/// When seed is provided, uses AdvancedRandomOps (Philox) for reproducibility.
fn antithetic_monte_carlo<R, C, F>(
    client: &C,
    f: &F,
    bounds: &[(f64, f64)],
    n_samples: usize,
    volume: f64,
    seed: Option<u64>,
) -> Result<MonteCarloResult<R>>
where
    R: Runtime,
    C: TensorOps<R>
        + ScalarOps<R>
        + RandomOps<R>
        + AdvancedRandomOps<R>
        + ReduceOps<R>
        + RuntimeClient<R>,
    F: Fn(&Tensor<R>) -> Result<Tensor<R>>,
{
    let device = client.device();
    let n_dims = bounds.len();

    // Use n_samples/2 base points, each generates 2 samples
    let n_base = n_samples / 2;
    let actual_samples = n_base * 2;

    // Generate base samples using RandomOps (GPU-accelerated)
    // Use seeded Philox PRNG for reproducibility if seed is provided
    let base_samples = match seed {
        Some(s) => client.philox_uniform(&[n_base, n_dims], s, 0, DType::F64)?,
        None => client.rand(&[n_base, n_dims], DType::F64)?,
    };

    // Transform base samples to integration domain
    let x = transform_to_bounds_tensor(client, &base_samples, bounds)?;

    // Compute antithetic samples: x' = a + b - x (reflection about midpoint)
    // This stays entirely on device
    let x_anti = transform_antithetic_tensor(client, &x, bounds)?;

    // Evaluate both sets (stays on device)
    let f_x = f(&x)?;
    let f_x_anti = f(&x_anti)?;

    // Compute paired averages: (f(x) + f(x')) / 2
    // This stays on device
    let sum = client.add(&f_x, &f_x_anti)?;
    let paired_avg = client.div_scalar(&sum, 2.0)?;

    // Compute mean and variance of paired values
    let (mean_scalar, variance_scalar) = compute_mean_variance_tensor(client, &paired_avg)?;

    // Standard error (accounts for variance reduction from pairing)
    let std_error = (variance_scalar / n_base as f64).sqrt() * volume;

    let integral = volume * mean_scalar;
    let integral_tensor = Tensor::<R>::from_slice(&[integral], &[1], device);

    Ok(MonteCarloResult {
        integral: integral_tensor,
        std_error,
        n_samples: actual_samples,
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
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
    let device = client.device();
    let n_dims = bounds.len();

    // Create scale and offset tensors [1, n_dims] for broadcasting
    let scales: Vec<f64> = bounds.iter().map(|(a, b)| b - a).collect();
    let offsets: Vec<f64> = bounds.iter().map(|(a, _)| *a).collect();

    let scale_tensor = Tensor::<R>::from_slice(&scales, &[1, n_dims], device);
    let offset_tensor = Tensor::<R>::from_slice(&offsets, &[1, n_dims], device);

    // x = offset + scale * samples (broadcasting handles [1, n_dims] with [n_samples, n_dims])
    let scaled = client.mul(samples, &scale_tensor)?;
    client.add(&scaled, &offset_tensor)
}

/// Transform samples to antithetic pairs using tensor operations.
///
/// x' = a + b - x (reflection about midpoint of each dimension)
/// All operations stay on device.
fn transform_antithetic_tensor<R, C>(
    client: &C,
    x: &Tensor<R>,
    bounds: &[(f64, f64)],
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
    let device = client.device();
    let n_dims = bounds.len();

    // Create (a + b) tensor [1, n_dims] for broadcasting
    let sums: Vec<f64> = bounds.iter().map(|(a, b)| a + b).collect();
    let sum_tensor = Tensor::<R>::from_slice(&sums, &[1, n_dims], device);

    // x' = (a + b) - x
    client.sub(&sum_tensor, x)
}

/// Compute mean and variance using tensor operations.
///
/// Returns scalar values (requires one device-to-host transfer at the end).
fn compute_mean_variance_tensor<R, C>(client: &C, values: &Tensor<R>) -> Result<(f64, f64)>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + ReduceOps<R> + RuntimeClient<R>,
{
    let n = values.numel();
    if n == 0 {
        return Ok((0.0, 0.0));
    }

    // Compute mean: sum(values) / n
    let sum_tensor = client.sum(values, &[0], false)?;
    let mean_tensor = client.div_scalar(&sum_tensor, n as f64)?;

    // Get scalar mean (one transfer)
    let mean_scalar: f64 = mean_tensor.to_vec()[0];

    if n == 1 {
        return Ok((mean_scalar, 0.0));
    }

    // Compute variance: sum((values - mean)^2) / (n - 1)
    let mean_broadcast = client.add_scalar(
        &Tensor::<R>::from_slice(&[0.0], &[1], client.device()),
        mean_scalar,
    )?;
    let centered = client.sub(values, &mean_broadcast)?;
    let squared = client.mul(&centered, &centered)?;
    let sum_sq = client.sum(&squared, &[0], false)?;
    let variance_tensor = client.div_scalar(&sum_sq, (n - 1) as f64)?;

    // Get scalar variance (one more transfer)
    let variance_scalar: f64 = variance_tensor.to_vec()[0];

    Ok((mean_scalar, variance_scalar))
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
    fn test_monte_carlo_unit_square() {
        let (device, client) = setup();

        // Integrate f(x,y) = 1 over [0,1]^2, exact = 1
        let result = monte_carlo_impl(
            &client,
            |_x| {
                let shape = _x.shape();
                let n = shape[0];
                let ones = vec![1.0; n];
                Ok(Tensor::<CpuRuntime>::from_slice(&ones, &[n], &device))
            },
            &[(0.0, 1.0), (0.0, 1.0)],
            &MonteCarloOptions::with_samples(10000),
        )
        .unwrap();

        let integral: Vec<f64> = result.integral.to_vec();
        assert!(
            (integral[0] - 1.0).abs() < 0.1,
            "integral = {}, expected 1.0",
            integral[0]
        );
    }

    #[test]
    fn test_monte_carlo_circle() {
        let (device, client) = setup();

        // Integrate indicator function of unit circle over [-1,1]^2
        // Area = π
        let result = monte_carlo_impl(
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
                Ok(Tensor::<CpuRuntime>::from_slice(&vals, &[n], &device))
            },
            &[(-1.0, 1.0), (-1.0, 1.0)],
            &MonteCarloOptions::with_samples(50000),
        )
        .unwrap();

        let integral: Vec<f64> = result.integral.to_vec();
        assert!(
            (integral[0] - PI).abs() < 0.2,
            "integral = {}, expected π ≈ {}",
            integral[0],
            PI
        );
    }

    #[test]
    fn test_stratified_monte_carlo() {
        let (device, client) = setup();

        // Same unit square test with stratified sampling
        let result = monte_carlo_impl(
            &client,
            |_x| {
                let shape = _x.shape();
                let n = shape[0];
                let ones = vec![1.0; n];
                Ok(Tensor::<CpuRuntime>::from_slice(&ones, &[n], &device))
            },
            &[(0.0, 1.0), (0.0, 1.0)],
            &MonteCarloOptions::with_samples(10000)
                .method(MonteCarloMethod::Stratified { n_strata: 10 }),
        )
        .unwrap();

        let integral: Vec<f64> = result.integral.to_vec();
        assert!(
            (integral[0] - 1.0).abs() < 0.1,
            "integral = {}, expected 1.0",
            integral[0]
        );
    }

    #[test]
    fn test_antithetic_monte_carlo() {
        let (device, client) = setup();

        // Integrate x over [0,1], exact = 0.5
        // Antithetic should help because f(x) = x is monotonic
        let result = monte_carlo_impl(
            &client,
            |x| {
                let data: Vec<f64> = x.to_vec();
                let n = data.len();
                Ok(Tensor::<CpuRuntime>::from_slice(&data, &[n], &device))
            },
            &[(0.0, 1.0)],
            &MonteCarloOptions::with_samples(10000).method(MonteCarloMethod::Antithetic),
        )
        .unwrap();

        let integral: Vec<f64> = result.integral.to_vec();
        assert!(
            (integral[0] - 0.5).abs() < 0.05,
            "integral = {}, expected 0.5",
            integral[0]
        );
    }

    #[test]
    fn test_monte_carlo_reproducibility_plain() {
        let (device, client) = setup();

        let seed = 42u64;

        // Run twice with same seed - should get identical results
        let f = |x: &Tensor<CpuRuntime>| {
            let data: Vec<f64> = x.to_vec();
            let n = data.len() / 2;
            let mut vals = Vec::with_capacity(n);
            for i in 0..n {
                let xi = data[i * 2];
                let yi = data[i * 2 + 1];
                vals.push(xi * xi + yi * yi);
            }
            Ok(Tensor::<CpuRuntime>::from_slice(&vals, &[n], &device))
        };

        let result1 = monte_carlo_impl(
            &client,
            f,
            &[(0.0, 1.0), (0.0, 1.0)],
            &MonteCarloOptions::with_samples(1000).seed(seed),
        )
        .unwrap();

        let result2 = monte_carlo_impl(
            &client,
            f,
            &[(0.0, 1.0), (0.0, 1.0)],
            &MonteCarloOptions::with_samples(1000).seed(seed),
        )
        .unwrap();

        let integral1: f64 = result1.integral.to_vec()[0];
        let integral2: f64 = result2.integral.to_vec()[0];

        assert_eq!(
            integral1, integral2,
            "Same seed should produce identical results"
        );
    }

    #[test]
    fn test_monte_carlo_reproducibility_stratified() {
        let (device, client) = setup();

        let seed = 123u64;

        let f = |x: &Tensor<CpuRuntime>| {
            let data: Vec<f64> = x.to_vec();
            let n = data.len() / 2;
            let mut vals = Vec::with_capacity(n);
            for i in 0..n {
                let xi = data[i * 2];
                let yi = data[i * 2 + 1];
                vals.push(xi + yi);
            }
            Ok(Tensor::<CpuRuntime>::from_slice(&vals, &[n], &device))
        };

        let options = MonteCarloOptions::with_samples(1000)
            .method(MonteCarloMethod::Stratified { n_strata: 5 })
            .seed(seed);

        let result1 = monte_carlo_impl(&client, f, &[(0.0, 1.0), (0.0, 1.0)], &options).unwrap();
        let result2 = monte_carlo_impl(&client, f, &[(0.0, 1.0), (0.0, 1.0)], &options).unwrap();

        let integral1: f64 = result1.integral.to_vec()[0];
        let integral2: f64 = result2.integral.to_vec()[0];

        assert_eq!(
            integral1, integral2,
            "Same seed should produce identical results for stratified"
        );
    }

    #[test]
    fn test_monte_carlo_reproducibility_antithetic() {
        let (device, client) = setup();

        let seed = 999u64;

        let f = |x: &Tensor<CpuRuntime>| {
            let data: Vec<f64> = x.to_vec();
            let n = data.len();
            let vals: Vec<f64> = data.iter().map(|&xi| xi.exp()).collect();
            Ok(Tensor::<CpuRuntime>::from_slice(&vals, &[n], &device))
        };

        let options = MonteCarloOptions::with_samples(1000)
            .method(MonteCarloMethod::Antithetic)
            .seed(seed);

        let result1 = monte_carlo_impl(&client, f, &[(0.0, 1.0)], &options).unwrap();
        let result2 = monte_carlo_impl(&client, f, &[(0.0, 1.0)], &options).unwrap();

        let integral1: f64 = result1.integral.to_vec()[0];
        let integral2: f64 = result2.integral.to_vec()[0];

        assert_eq!(
            integral1, integral2,
            "Same seed should produce identical results for antithetic"
        );
    }
}
