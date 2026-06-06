//! CPU implementation of signal analysis algorithms.
//!
//! Some algorithms (hilbert, resample) are GPU-accelerable and delegate to impl_generic.
//! Others (decimate, find_peaks, savgol) are CPU-only due to their sequential access patterns.

use crate::signal::impl_generic::{
    apply_butter_lowpass, apply_fir_lowpass, compute_prominences, compute_savgol_coeffs,
    filter_by_distance, hilbert_impl, resample_impl,
};
use crate::signal::traits::analysis::{
    DecimateFilterImpl, DecimateParams, HilbertResult, PeakParams, PeakResult,
    SignalAnalysisAlgorithms,
};
use numr::error::{Error, Result};
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl SignalAnalysisAlgorithms<CpuRuntime> for CpuClient {
    fn hilbert(&self, x: &Tensor<CpuRuntime>) -> Result<HilbertResult<CpuRuntime>> {
        hilbert_impl(self, x)
    }

    fn resample(
        &self,
        x: &Tensor<CpuRuntime>,
        num: usize,
        den: usize,
    ) -> Result<Tensor<CpuRuntime>> {
        resample_impl(self, x, num, den)
    }

    fn decimate(
        &self,
        x: &Tensor<CpuRuntime>,
        q: usize,
        params: DecimateParams,
    ) -> Result<Tensor<CpuRuntime>> {
        decimate_cpu(x, q, params)
    }

    fn find_peaks(
        &self,
        x: &Tensor<CpuRuntime>,
        params: PeakParams,
    ) -> Result<PeakResult<CpuRuntime>> {
        find_peaks_cpu(x, params)
    }

    fn savgol_filter(
        &self,
        x: &Tensor<CpuRuntime>,
        window_length: usize,
        polyorder: usize,
        deriv: usize,
    ) -> Result<Tensor<CpuRuntime>> {
        savgol_filter_cpu(x, window_length, polyorder, deriv)
    }
}

/// Decimate a signal (CPU implementation).
///
/// This is CPU-only because it uses IIR/FIR filtering which is inherently sequential.
fn decimate_cpu(
    x: &Tensor<CpuRuntime>,
    q: usize,
    params: DecimateParams,
) -> Result<Tensor<CpuRuntime>> {
    // CPU-specific: extract data for processing
    let x_data: Vec<f64> = x.to_vec();
    let n = x_data.len();
    let device = x.device();

    if n == 0 {
        return Err(Error::InvalidArgument {
            arg: "x",
            reason: "Input signal cannot be empty".to_string(),
        });
    }

    if q == 0 {
        return Err(Error::InvalidArgument {
            arg: "q",
            reason: "Decimation factor must be positive".to_string(),
        });
    }

    if q == 1 {
        return Ok(x.clone());
    }

    // Design anti-aliasing filter
    // Cutoff at 0.8 * Nyquist / q to prevent aliasing
    let cutoff = 0.8 / q as f64;

    // Apply filter
    let filtered = match params.ftype {
        DecimateFilterImpl::Iir => {
            // Use simple IIR Butterworth lowpass
            apply_butter_lowpass(&x_data, cutoff, params.n, params.zero_phase)
        }
        DecimateFilterImpl::Fir => {
            // Use FIR lowpass
            let fir_len = params.n * 2 * q + 1;
            apply_fir_lowpass(&x_data, cutoff, fir_len)
        }
    };

    // Downsample by taking every q-th sample
    let output_len = n.div_ceil(q);
    let mut result = Vec::with_capacity(output_len);

    for i in 0..output_len {
        let idx = i * q;
        if idx < filtered.len() {
            result.push(filtered[idx]);
        }
    }

    Ok(Tensor::from_slice(&result, &[result.len()], device))
}

/// Find peaks in a signal (CPU implementation).
///
/// This is CPU-only because it requires element-wise comparisons and
/// sequential filtering operations.
fn find_peaks_cpu(x: &Tensor<CpuRuntime>, params: PeakParams) -> Result<PeakResult<CpuRuntime>> {
    // CPU-specific: extract data for processing
    let x_data: Vec<f64> = x.to_vec();
    let n = x_data.len();
    let device = x.device();

    if n < 3 {
        // Need at least 3 points to find a peak
        return Ok(PeakResult {
            indices: vec![],
            heights: Tensor::from_slice(&[] as &[f64], &[0], device),
            prominences: None,
        });
    }

    // Find all local maxima
    let mut peaks: Vec<usize> = Vec::new();

    for i in 1..n - 1 {
        if x_data[i] > x_data[i - 1] && x_data[i] > x_data[i + 1] {
            peaks.push(i);
        }
    }

    // Filter by height
    if let Some(min_height) = params.height {
        peaks.retain(|&i| x_data[i] >= min_height);
    }

    // Filter by threshold (difference from neighbors)
    if let Some(threshold) = params.threshold {
        peaks.retain(|&i| {
            let left_diff = x_data[i] - x_data[i - 1];
            let right_diff = x_data[i] - x_data[i + 1];
            left_diff >= threshold && right_diff >= threshold
        });
    }

    // Filter by distance (keep highest peak when peaks are too close)
    if let Some(min_distance) = params.distance
        && min_distance > 0
    {
        peaks = filter_by_distance(&peaks, &x_data, min_distance);
    }

    // Compute prominences if requested
    let prominences = if params.prominence.is_some() {
        let proms = compute_prominences(&peaks, &x_data);

        // Filter by prominence
        if let Some(min_prom) = params.prominence {
            let filtered: Vec<(usize, f64)> = peaks
                .iter()
                .zip(proms.iter())
                .filter(|(_, p)| **p >= min_prom)
                .map(|(i, p)| (*i, *p))
                .collect();

            peaks = filtered.iter().map(|(i, _)| *i).collect();
            let new_proms: Vec<f64> = filtered.iter().map(|(_, p)| *p).collect();
            Some(Tensor::from_slice(&new_proms, &[new_proms.len()], device))
        } else {
            Some(Tensor::from_slice(&proms, &[proms.len()], device))
        }
    } else {
        None
    };

    // Extract heights
    let heights: Vec<f64> = peaks.iter().map(|&i| x_data[i]).collect();

    Ok(PeakResult {
        indices: peaks,
        heights: Tensor::from_slice(&heights, &[heights.len()], device),
        prominences,
    })
}

/// Apply Savitzky-Golay filter (CPU implementation).
///
/// This is CPU-only because it requires convolution with computed coefficients
/// and mirror boundary handling.
fn savgol_filter_cpu(
    x: &Tensor<CpuRuntime>,
    window_length: usize,
    polyorder: usize,
    deriv: usize,
) -> Result<Tensor<CpuRuntime>> {
    // CPU-specific: extract data for processing
    let x_data: Vec<f64> = x.to_vec();
    let n = x_data.len();
    let device = x.device();

    if window_length.is_multiple_of(2) {
        return Err(Error::InvalidArgument {
            arg: "window_length",
            reason: "window_length must be odd".to_string(),
        });
    }

    if window_length < polyorder + 2 {
        return Err(Error::InvalidArgument {
            arg: "window_length",
            reason: "window_length must be greater than polyorder + 1".to_string(),
        });
    }

    if deriv > polyorder {
        return Err(Error::InvalidArgument {
            arg: "deriv",
            reason: "deriv must be <= polyorder".to_string(),
        });
    }

    let half_window = window_length / 2;

    // Compute Savitzky-Golay coefficients
    let coeffs = compute_savgol_coeffs(window_length, polyorder, deriv);

    // Apply filter
    let mut result = Vec::with_capacity(n);

    for i in 0..n {
        let mut sum = 0.0;

        for (j, &coeff) in coeffs.iter().enumerate() {
            let k = i as isize + j as isize - half_window as isize;
            // Mirror boundary conditions
            let idx = if k < 0 {
                (-k) as usize
            } else if k >= n as isize {
                2 * n - 2 - k as usize
            } else {
                k as usize
            };

            if idx < n {
                sum += coeff * x_data[idx];
            }
        }

        result.push(sum);
    }

    Ok(Tensor::from_slice(&result, &[n], device))
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::CpuDevice;
    use std::f64::consts::PI;

    fn setup() -> (CpuClient, CpuDevice) {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        (client, device)
    }

    #[test]
    fn test_hilbert_sine() {
        let (client, device) = setup();

        let n = 256;
        let freq = 5.0;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * freq * i as f64 / n as f64).sin())
            .collect();

        let x = Tensor::from_slice(&signal, &[n], &device);
        let result = client.hilbert(&x).unwrap();

        let env = result.envelope().unwrap();
        let env_data: Vec<f64> = env.to_vec();

        // Ignore edge effects (first and last 10%)
        let start = n / 10;
        let end = n - n / 10;
        for &e in &env_data[start..end] {
            assert!((e - 1.0).abs() < 0.15, "Envelope should be ~1.0, got {}", e);
        }
    }

    #[test]
    fn test_hilbert_envelope() {
        let (client, device) = setup();

        let n = 512;
        let carrier_freq = 50.0;
        let mod_freq = 5.0;

        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                (1.0 + 0.5 * (2.0 * PI * mod_freq * t).sin()) * (2.0 * PI * carrier_freq * t).sin()
            })
            .collect();

        let x = Tensor::from_slice(&signal, &[n], &device);
        let result = client.hilbert(&x).unwrap();
        let env = result.envelope().unwrap();
        let env_data: Vec<f64> = env.to_vec();

        let env_mean: f64 = env_data.iter().sum::<f64>() / n as f64;
        assert!((env_mean - 1.0).abs() < 0.2, "Mean envelope should be ~1.0");
    }

    #[test]
    fn test_resample_upsample() {
        let (client, device) = setup();

        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let n = signal.len();
        let x = Tensor::from_slice(&signal, &[n], &device);

        let result = client.resample(&x, 2, 1).unwrap();
        let result_data: Vec<f64> = result.to_vec();

        assert_eq!(result_data.len(), n * 2);
    }

    #[test]
    fn test_resample_downsample() {
        let (client, device) = setup();

        let n = 100;
        let signal: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        let x = Tensor::from_slice(&signal, &[n], &device);

        let result = client.resample(&x, 1, 2).unwrap();
        let result_data: Vec<f64> = result.to_vec();

        assert_eq!(result_data.len(), n.div_ceil(2));
    }

    #[test]
    fn test_decimate() {
        let (client, device) = setup();

        let n = 256;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 2.0 * i as f64 / n as f64).sin())
            .collect();
        let x = Tensor::from_slice(&signal, &[n], &device);

        let params = DecimateParams::default();
        let result = client.decimate(&x, 4, params).unwrap();
        let result_data: Vec<f64> = result.to_vec();

        assert_eq!(result_data.len(), n.div_ceil(4));

        let max_val = result_data.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
        assert!(max_val > 0.1, "Decimated signal should preserve content");
    }

    #[test]
    fn test_find_peaks_simple() {
        let (client, device) = setup();

        let signal = vec![0.0, 1.0, 0.0, 2.0, 0.0, 1.5, 0.0];
        let x = Tensor::from_slice(&signal, &[signal.len()], &device);

        let params = PeakParams::new();
        let result = client.find_peaks(&x, params).unwrap();

        assert_eq!(result.indices, vec![1, 3, 5]);

        let heights: Vec<f64> = result.heights.to_vec();
        assert!((heights[0] - 1.0).abs() < 1e-10);
        assert!((heights[1] - 2.0).abs() < 1e-10);
        assert!((heights[2] - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_find_peaks_with_height() {
        let (client, device) = setup();

        let signal = vec![0.0, 1.0, 0.0, 2.0, 0.0, 0.5, 0.0];
        let x = Tensor::from_slice(&signal, &[signal.len()], &device);

        let params = PeakParams::new().with_height(1.0);
        let result = client.find_peaks(&x, params).unwrap();

        assert_eq!(result.indices, vec![1, 3]);
    }

    #[test]
    fn test_find_peaks_with_distance() {
        let (client, device) = setup();

        let signal = vec![0.0, 1.0, 0.5, 1.2, 0.0, 0.0, 2.0, 0.0];
        let x = Tensor::from_slice(&signal, &[signal.len()], &device);

        let params = PeakParams::new().with_distance(3);
        let result = client.find_peaks(&x, params).unwrap();

        assert!(result.indices.len() <= 3);
        assert!(result.indices.contains(&6));
    }

    #[test]
    fn test_savgol_smoothing() {
        let (client, device) = setup();

        let n = 101;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / (n - 1) as f64;
                t * t + 0.1 * ((i * 7) as f64).sin()
            })
            .collect();

        let x = Tensor::from_slice(&signal, &[n], &device);

        let result = client.savgol_filter(&x, 11, 2, 0).unwrap();
        let result_data: Vec<f64> = result.to_vec();

        assert_eq!(result_data.len(), n);

        for (i, &val) in result_data.iter().enumerate().take(80).skip(20) {
            let t = i as f64 / (n - 1) as f64;
            let expected = t * t;
            assert!(
                (val - expected).abs() < 0.15,
                "Smoothed value at {} should be close to {} (got {})",
                i,
                expected,
                val
            );
        }
    }

    #[test]
    fn test_savgol_derivative() {
        let (client, device) = setup();

        let n = 101;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = (i as f64 - 50.0) / 50.0;
                t * t
            })
            .collect();

        let x = Tensor::from_slice(&signal, &[n], &device);

        let result = client.savgol_filter(&x, 11, 3, 1).unwrap();
        let result_data: Vec<f64> = result.to_vec();

        let scale = 50.0;
        for (i, &val) in result_data.iter().enumerate().take(80).skip(20) {
            let t = (i as f64 - 50.0) / 50.0;
            let expected_deriv = 2.0 * t / scale;
            assert!(
                (val - expected_deriv).abs() < 0.1,
                "Derivative at {} should be ~{} (got {})",
                i,
                expected_deriv,
                val
            );
        }
    }

    #[test]
    fn test_find_peaks_with_prominence() {
        let (client, device) = setup();

        let signal = vec![0.0, 1.0, 0.8, 2.0, 0.5, 0.6, 0.0];
        let x = Tensor::from_slice(&signal, &[signal.len()], &device);

        let params = PeakParams::new().with_prominence(0.5);
        let result = client.find_peaks(&x, params).unwrap();

        assert!(
            result.indices.contains(&3),
            "Should find the prominent peak"
        );

        assert!(result.prominences.is_some());
    }
}
