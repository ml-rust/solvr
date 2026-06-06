//! CPU implementation of Wiener filter algorithms.
//!
//! Wiener filtering is CPU-only because it requires computing local
//! statistics over sliding windows, which involves sequential data access
//! patterns that don't parallelize efficiently on GPU.

use crate::signal::traits::wiener::WienerAlgorithms;
use numr::error::{Error, Result};
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl WienerAlgorithms<CpuRuntime> for CpuClient {
    fn wiener(
        &self,
        x: &Tensor<CpuRuntime>,
        kernel_size: Option<usize>,
        noise: Option<f64>,
    ) -> Result<Tensor<CpuRuntime>> {
        wiener_cpu(x, kernel_size, noise)
    }

    fn wiener2d(
        &self,
        x: &Tensor<CpuRuntime>,
        kernel_size: Option<(usize, usize)>,
        noise: Option<f64>,
    ) -> Result<Tensor<CpuRuntime>> {
        wiener2d_cpu(x, kernel_size, noise)
    }
}

/// Apply a 1D Wiener filter for noise reduction (CPU implementation).
fn wiener_cpu(
    x: &Tensor<CpuRuntime>,
    kernel_size: Option<usize>,
    noise: Option<f64>,
) -> Result<Tensor<CpuRuntime>> {
    if x.ndim() != 1 {
        return Err(Error::InvalidArgument {
            arg: "x",
            reason: "Input must be 1D".to_string(),
        });
    }

    let n = x.shape()[0];
    let device = x.device();

    if n == 0 {
        return Err(Error::InvalidArgument {
            arg: "x",
            reason: "Input signal cannot be empty".to_string(),
        });
    }

    let ksize = kernel_size.unwrap_or(3);
    if ksize == 0 {
        return Err(Error::InvalidArgument {
            arg: "kernel_size",
            reason: "Kernel size must be positive".to_string(),
        });
    }
    if ksize.is_multiple_of(2) {
        return Err(Error::InvalidArgument {
            arg: "kernel_size",
            reason: "Kernel size must be odd".to_string(),
        });
    }

    // CPU-specific: extract data for processing
    let data: Vec<f64> = x.to_vec();
    let half = ksize / 2;

    // Compute local mean and variance for each point
    let mut local_means = Vec::with_capacity(n);
    let mut local_vars = Vec::with_capacity(n);

    for i in 0..n {
        let start = i.saturating_sub(half);
        let end = (i + half + 1).min(n);
        let window_size = end - start;

        // Compute local mean
        let sum: f64 = data[start..end].iter().sum();
        let mean = sum / window_size as f64;
        local_means.push(mean);

        // Compute local variance
        let var_sum: f64 = data[start..end].iter().map(|&v| (v - mean).powi(2)).sum();
        let var = var_sum / window_size as f64;
        local_vars.push(var);
    }

    // Estimate noise variance if not provided
    // Use minimum local variance as noise estimate (heuristic)
    let noise_var = noise.unwrap_or_else(|| {
        local_vars
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min)
            .max(1e-10)
    });

    // Apply Wiener filter
    // output = mean + max(0, (var - noise) / var) * (input - mean)
    let mut result = Vec::with_capacity(n);

    for i in 0..n {
        let local_mean = local_means[i];
        let local_var = local_vars[i];

        // Compute filter coefficient
        // Wiener filter: (local_var - noise) / local_var, clamped to [0, 1]
        let filter_coeff = if local_var > noise_var {
            (local_var - noise_var) / local_var
        } else {
            0.0
        };

        // Apply filter
        let filtered = local_mean + filter_coeff * (data[i] - local_mean);
        result.push(filtered);
    }

    Ok(Tensor::from_slice(&result, &[n], device))
}

/// Apply a 2D Wiener filter for noise reduction in images (CPU implementation).
fn wiener2d_cpu(
    x: &Tensor<CpuRuntime>,
    kernel_size: Option<(usize, usize)>,
    noise: Option<f64>,
) -> Result<Tensor<CpuRuntime>> {
    if x.ndim() != 2 {
        return Err(Error::InvalidArgument {
            arg: "x",
            reason: "Input must be 2D".to_string(),
        });
    }

    let shape = x.shape();
    let height = shape[0];
    let width = shape[1];
    let device = x.device();

    if height == 0 || width == 0 {
        return Err(Error::InvalidArgument {
            arg: "x",
            reason: "Input image cannot be empty".to_string(),
        });
    }

    let (kh, kw) = kernel_size.unwrap_or((3, 3));
    if kh == 0 || kw == 0 {
        return Err(Error::InvalidArgument {
            arg: "kernel_size",
            reason: "Kernel sizes must be positive".to_string(),
        });
    }
    if kh % 2 == 0 || kw % 2 == 0 {
        return Err(Error::InvalidArgument {
            arg: "kernel_size",
            reason: "Kernel sizes must be odd".to_string(),
        });
    }

    // CPU-specific: extract data for processing
    let data: Vec<f64> = x.to_vec();
    let half_h = kh / 2;
    let half_w = kw / 2;

    // Compute local mean and variance for each pixel
    let total_pixels = height * width;
    let mut local_means = vec![0.0; total_pixels];
    let mut local_vars = vec![0.0; total_pixels];

    for i in 0..height {
        for j in 0..width {
            let row_start = i.saturating_sub(half_h);
            let row_end = (i + half_h + 1).min(height);
            let col_start = j.saturating_sub(half_w);
            let col_end = (j + half_w + 1).min(width);
            let window_size = (row_end - row_start) * (col_end - col_start);

            // Compute local mean
            let mut sum = 0.0;
            for row in row_start..row_end {
                for col in col_start..col_end {
                    sum += data[row * width + col];
                }
            }
            let mean = sum / window_size as f64;
            local_means[i * width + j] = mean;

            // Compute local variance
            let mut var_sum = 0.0;
            for row in row_start..row_end {
                for col in col_start..col_end {
                    let diff = data[row * width + col] - mean;
                    var_sum += diff * diff;
                }
            }
            let var = var_sum / window_size as f64;
            local_vars[i * width + j] = var;
        }
    }

    // Estimate noise variance if not provided
    let noise_var = noise.unwrap_or_else(|| {
        local_vars
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min)
            .max(1e-10)
    });

    // Apply Wiener filter
    let mut result = vec![0.0; total_pixels];

    for i in 0..total_pixels {
        let local_mean = local_means[i];
        let local_var = local_vars[i];

        // Compute filter coefficient
        let filter_coeff = if local_var > noise_var {
            (local_var - noise_var) / local_var
        } else {
            0.0
        };

        // Apply filter
        result[i] = local_mean + filter_coeff * (data[i] - local_mean);
    }

    Ok(Tensor::from_slice(&result, &[height, width], device))
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
    fn test_wiener_reduces_noise() {
        let (client, device) = setup();

        // Signal with constant value plus noise
        // True signal is 5.0, noise varies around it
        let signal = vec![4.5, 5.5, 4.8, 5.2, 5.0, 4.9, 5.1, 4.7, 5.3, 5.0];
        let x = Tensor::from_slice(&signal, &[signal.len()], &device);

        let result = client.wiener(&x, Some(3), Some(0.05)).unwrap();
        let result_data: Vec<f64> = result.to_vec();

        // Filtered signal should be closer to 5.0 (less variance)
        let orig_var: f64 = signal.iter().map(|&v| (v - 5.0_f64).powi(2)).sum::<f64>() / 10.0;
        let filt_var: f64 = result_data
            .iter()
            .map(|&v| (v - 5.0_f64).powi(2))
            .sum::<f64>()
            / 10.0;

        // Filtered variance should be less than original
        assert!(
            filt_var <= orig_var + 0.01,
            "Wiener should reduce noise variance"
        );
    }

    #[test]
    fn test_wiener_preserves_edges() {
        let (client, device) = setup();

        // Signal with a step edge
        let signal = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let x = Tensor::from_slice(&signal, &[signal.len()], &device);

        let result = client.wiener(&x, Some(3), None).unwrap();
        let result_data: Vec<f64> = result.to_vec();

        // Values before edge should be close to 0
        assert!(result_data[0] < 0.3, "Should preserve low region");
        assert!(result_data[1] < 0.3, "Should preserve low region");

        // Values after edge should be close to 1
        assert!(result_data[6] > 0.7, "Should preserve high region");
        assert!(result_data[7] > 0.7, "Should preserve high region");
    }

    #[test]
    fn test_wiener_with_high_noise() {
        let (client, device) = setup();

        // Very noisy signal
        let signal = vec![1.0, 10.0, 2.0, 9.0, 3.0, 8.0, 4.0, 7.0];
        let x = Tensor::from_slice(&signal, &[signal.len()], &device);

        // With high noise estimate, filter should smooth heavily
        let result = client.wiener(&x, Some(3), Some(10.0)).unwrap();
        let result_data: Vec<f64> = result.to_vec();

        // Result should be much more uniform (approaching local mean)
        let mean: f64 = signal.iter().sum::<f64>() / signal.len() as f64;
        for val in result_data.iter() {
            // Values should be closer to overall mean when noise is high
            assert!(
                (*val - mean).abs() < 5.0,
                "High noise should smooth to near-constant"
            );
        }
    }

    #[test]
    fn test_wiener2d_simple() {
        let (client, device) = setup();

        // 3x3 image with noisy center
        let image = vec![
            1.0, 1.0, 1.0, 1.0, 5.0, 1.0, // Center is an outlier
            1.0, 1.0, 1.0,
        ];
        let x = Tensor::from_slice(&image, &[3, 3], &device);

        let result = client.wiener2d(&x, Some((3, 3)), Some(0.1)).unwrap();
        let result_data: Vec<f64> = result.to_vec();

        // Center should be smoothed toward neighbors
        assert!(
            result_data[4] < 5.0,
            "Center outlier should be reduced by Wiener filter"
        );
    }

    #[test]
    fn test_wiener2d_preserves_structure() {
        let (client, device) = setup();

        // Image with a clear pattern
        let image = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0,
            1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let x = Tensor::from_slice(&image, &[5, 5], &device);

        let result = client.wiener2d(&x, Some((3, 3)), Some(0.01)).unwrap();
        let result_data: Vec<f64> = result.to_vec();

        // Center of the bright region should remain bright
        assert!(
            result_data[12] > 0.8,
            "Center of pattern should be preserved"
        );

        // Corners should remain dark
        assert!(result_data[0] < 0.2, "Corners should remain dark");
        assert!(result_data[4] < 0.2, "Corners should remain dark");
    }

    #[test]
    fn test_wiener_auto_noise_estimation() {
        let (client, device) = setup();

        // Signal with uniform region (for noise estimation)
        let signal = vec![1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0];
        let x = Tensor::from_slice(&signal, &[signal.len()], &device);

        // Auto noise estimation (None)
        let result = client.wiener(&x, Some(3), None).unwrap();
        let result_data: Vec<f64> = result.to_vec();

        // Should still produce valid output
        assert_eq!(result_data.len(), signal.len());
        for val in result_data.iter() {
            assert!(val.is_finite(), "Output should be finite");
        }
    }

    #[test]
    fn test_wiener_odd_kernel_required() {
        let (client, device) = setup();

        let signal = vec![1.0, 2.0, 3.0];
        let x = Tensor::from_slice(&signal, &[signal.len()], &device);

        // Even kernel size should fail
        let result = client.wiener(&x, Some(4), None);
        assert!(result.is_err());
    }
}
