//! CPU implementation of median filter algorithms.
//!
//! Median filtering is CPU-only because:
//! 1. Computing median requires sorting/selection which is hard to parallelize
//! 2. The sliding window pattern doesn't map well to GPU SIMD
//! 3. Efficient median algorithms (running median) are sequential

use crate::signal::traits::medfilt::MedianFilterAlgorithms;
use numr::error::{Error, Result};
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl MedianFilterAlgorithms<CpuRuntime> for CpuClient {
    fn medfilt(&self, x: &Tensor<CpuRuntime>, kernel_size: usize) -> Result<Tensor<CpuRuntime>> {
        medfilt_cpu(x, kernel_size)
    }

    fn medfilt2d(
        &self,
        x: &Tensor<CpuRuntime>,
        kernel_size: (usize, usize),
    ) -> Result<Tensor<CpuRuntime>> {
        medfilt2d_cpu(x, kernel_size)
    }
}

/// Apply a 1D median filter to a signal (CPU implementation).
fn medfilt_cpu(x: &Tensor<CpuRuntime>, kernel_size: usize) -> Result<Tensor<CpuRuntime>> {
    if x.ndim() != 1 {
        return Err(Error::InvalidArgument {
            arg: "x",
            reason: "Input must be 1D".to_string(),
        });
    }

    if kernel_size == 0 {
        return Err(Error::InvalidArgument {
            arg: "kernel_size",
            reason: "Kernel size must be positive".to_string(),
        });
    }

    if kernel_size.is_multiple_of(2) {
        return Err(Error::InvalidArgument {
            arg: "kernel_size",
            reason: "Kernel size must be odd".to_string(),
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

    // CPU-specific: extract data for processing
    let data: Vec<f64> = x.to_vec();
    let half = kernel_size / 2;

    let mut result = Vec::with_capacity(n);

    for i in 0..n {
        // Determine window bounds
        let start = i.saturating_sub(half);
        let end = (i + half + 1).min(n);

        // Collect window values
        let mut window: Vec<f64> = data[start..end].to_vec();

        // Compute median
        let median = compute_median(&mut window);
        result.push(median);
    }

    Ok(Tensor::from_slice(&result, &[n], device))
}

/// Apply a 2D median filter to an image (CPU implementation).
fn medfilt2d_cpu(
    x: &Tensor<CpuRuntime>,
    kernel_size: (usize, usize),
) -> Result<Tensor<CpuRuntime>> {
    if x.ndim() != 2 {
        return Err(Error::InvalidArgument {
            arg: "x",
            reason: "Input must be 2D".to_string(),
        });
    }

    let (kh, kw) = kernel_size;

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

    // CPU-specific: extract data for processing
    let data: Vec<f64> = x.to_vec();
    let half_h = kh / 2;
    let half_w = kw / 2;

    let mut result = vec![0.0; height * width];

    for i in 0..height {
        for j in 0..width {
            // Determine window bounds
            let row_start = i.saturating_sub(half_h);
            let row_end = (i + half_h + 1).min(height);
            let col_start = j.saturating_sub(half_w);
            let col_end = (j + half_w + 1).min(width);

            // Collect window values
            let mut window = Vec::with_capacity((row_end - row_start) * (col_end - col_start));
            for row in row_start..row_end {
                for col in col_start..col_end {
                    window.push(data[row * width + col]);
                }
            }

            // Compute median
            let median = compute_median(&mut window);
            result[i * width + j] = median;
        }
    }

    Ok(Tensor::from_slice(&result, &[height, width], device))
}

/// Compute median of a slice using quickselect algorithm.
/// O(n) on average, better than O(n log n) for full sorting.
/// Note: This modifies the input slice in place.
fn compute_median(data: &mut [f64]) -> f64 {
    let n = data.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return data[0];
    }

    // For small arrays, just sort (more efficient than quickselect overhead)
    if n <= 9 {
        data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        return data[n / 2];
    }

    // Use quickselect for larger arrays
    let mid = n / 2;
    quickselect(data, mid)
}

/// Quickselect algorithm to find k-th smallest element.
/// Note: This modifies the input slice in place.
fn quickselect(data: &mut [f64], k: usize) -> f64 {
    let n = data.len();
    if n == 1 {
        return data[0];
    }

    // Use median-of-three pivot selection
    let mid = n / 2;
    let pivot_idx = if data[0] <= data[mid] {
        if data[mid] <= data[n - 1] {
            mid
        } else if data[0] <= data[n - 1] {
            n - 1
        } else {
            0
        }
    } else if data[0] <= data[n - 1] {
        0
    } else if data[mid] <= data[n - 1] {
        n - 1
    } else {
        mid
    };

    // Partition around pivot
    data.swap(pivot_idx, n - 1);
    let pivot = data[n - 1];

    let mut store_idx = 0;
    for i in 0..n - 1 {
        if data[i] < pivot {
            data.swap(i, store_idx);
            store_idx += 1;
        }
    }
    data.swap(store_idx, n - 1);

    // Recurse on appropriate partition
    if k == store_idx {
        data[store_idx]
    } else if k < store_idx {
        quickselect(&mut data[..store_idx], k)
    } else {
        quickselect(&mut data[store_idx + 1..], k - store_idx - 1)
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
    fn test_medfilt_simple() {
        let (client, device) = setup();

        // Signal with an outlier
        let signal = vec![1.0, 1.0, 1.0, 100.0, 1.0, 1.0, 1.0];
        let x = Tensor::from_slice(&signal, &[signal.len()], &device);

        let result = client.medfilt(&x, 3).unwrap();
        let result_data: Vec<f64> = result.to_vec();

        // The outlier (100.0) should be replaced with median of neighbors
        // Position 3 window: [1.0, 100.0, 1.0] -> median = 1.0
        assert!((result_data[3] - 1.0).abs() < 1e-10);

        // Non-outlier positions should remain ~1.0
        assert!((result_data[0] - 1.0).abs() < 1e-10);
        assert!((result_data[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_medfilt_larger_kernel() {
        let (client, device) = setup();

        // Signal with spike
        let signal = vec![1.0, 2.0, 3.0, 100.0, 5.0, 6.0, 7.0];
        let x = Tensor::from_slice(&signal, &[signal.len()], &device);

        let result = client.medfilt(&x, 5).unwrap();
        let result_data: Vec<f64> = result.to_vec();

        // Position 3: window [2.0, 3.0, 100.0, 5.0, 6.0] -> sorted [2.0, 3.0, 5.0, 6.0, 100.0] -> median = 5.0
        assert!((result_data[3] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_medfilt_preserves_monotonic() {
        let (client, device) = setup();

        // Monotonic signal should be mostly preserved
        let signal: Vec<f64> = (0..9).map(|i| i as f64).collect();
        let x = Tensor::from_slice(&signal, &[signal.len()], &device);

        let result = client.medfilt(&x, 3).unwrap();
        let result_data: Vec<f64> = result.to_vec();

        // Middle values should equal originals since they're monotonic
        for i in 1..8 {
            assert!((result_data[i] - signal[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_medfilt2d_simple() {
        let (client, device) = setup();

        // 3x3 image with center spike
        let image = vec![1.0, 1.0, 1.0, 1.0, 100.0, 1.0, 1.0, 1.0, 1.0];
        let x = Tensor::from_slice(&image, &[3, 3], &device);

        let result = client.medfilt2d(&x, (3, 3)).unwrap();
        let result_data: Vec<f64> = result.to_vec();

        // Center should be filtered to 1.0 (median of 8 ones and one 100)
        assert!((result_data[4] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_medfilt2d_larger() {
        let (client, device) = setup();

        // 5x5 image with noise
        let mut image = vec![0.0; 25];
        for (i, val) in image.iter_mut().enumerate() {
            *val = (i % 5) as f64;
        }
        image[12] = 100.0; // Add outlier at center

        let x = Tensor::from_slice(&image, &[5, 5], &device);

        let result = client.medfilt2d(&x, (3, 3)).unwrap();
        let result_data: Vec<f64> = result.to_vec();

        // Outlier should be smoothed
        assert!(result_data[12] < 10.0);
    }

    #[test]
    fn test_medfilt_edge_handling() {
        let (client, device) = setup();

        // Short signal
        let signal = vec![3.0, 1.0, 2.0];
        let x = Tensor::from_slice(&signal, &[signal.len()], &device);

        let result = client.medfilt(&x, 3).unwrap();
        let result_data: Vec<f64> = result.to_vec();

        // Middle value: window [3.0, 1.0, 2.0] -> sorted [1.0, 2.0, 3.0] -> median = 2.0
        assert!((result_data[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_medfilt_odd_kernel_required() {
        let (client, device) = setup();

        let signal = vec![1.0, 2.0, 3.0];
        let x = Tensor::from_slice(&signal, &[signal.len()], &device);

        // Even kernel size should fail
        let result = client.medfilt(&x, 4);
        assert!(result.is_err());
    }
}
