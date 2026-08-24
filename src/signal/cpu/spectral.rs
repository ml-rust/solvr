//! CPU implementation of spectral analysis algorithms.

use crate::signal::impl_generic::{
    coherence_impl, csd_impl, lombscargle_impl, periodogram_impl, welch_impl,
};
use crate::signal::traits::spectral::{
    CoherenceResult, CsdResult, PeriodogramParams, PeriodogramResult, SpectralAnalysisAlgorithms,
    WelchParams, WelchResult,
};
use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl SpectralAnalysisAlgorithms<CpuRuntime> for CpuClient {
    fn welch(
        &self,
        x: &Tensor<CpuRuntime>,
        params: WelchParams<CpuRuntime>,
    ) -> Result<WelchResult<CpuRuntime>> {
        welch_impl(self, x, params)
    }

    fn periodogram(
        &self,
        x: &Tensor<CpuRuntime>,
        params: PeriodogramParams<CpuRuntime>,
    ) -> Result<PeriodogramResult<CpuRuntime>> {
        periodogram_impl(self, x, params)
    }

    fn csd(
        &self,
        x: &Tensor<CpuRuntime>,
        y: &Tensor<CpuRuntime>,
        params: WelchParams<CpuRuntime>,
    ) -> Result<CsdResult<CpuRuntime>> {
        csd_impl(self, x, y, params)
    }

    fn coherence(
        &self,
        x: &Tensor<CpuRuntime>,
        y: &Tensor<CpuRuntime>,
        params: WelchParams<CpuRuntime>,
    ) -> Result<CoherenceResult<CpuRuntime>> {
        coherence_impl(self, x, y, params)
    }

    fn lombscargle(
        &self,
        t: &Tensor<CpuRuntime>,
        x: &Tensor<CpuRuntime>,
        freqs: &Tensor<CpuRuntime>,
        normalize: bool,
    ) -> Result<Tensor<CpuRuntime>> {
        lombscargle_impl(self, t, x, freqs, normalize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signal::traits::spectral::{Detrend, PsdScaling, SpectralWindow};
    use numr::runtime::cpu::CpuDevice;
    use std::f64::consts::PI;

    fn setup() -> (CpuClient, CpuDevice) {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        (client, device)
    }

    #[test]
    fn test_welch_sine() {
        let (client, device) = setup();

        // Generate a pure sine wave at 10 Hz, sampled at 100 Hz
        let fs = 100.0;
        let f0 = 10.0;
        let n = 1000;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * f0 * i as f64 / fs).sin())
            .collect();

        let x = Tensor::from_slice(&signal, &[n], &device).unwrap();
        let params = WelchParams::new(device.clone())
            .with_fs(fs)
            .with_nperseg(256)
            .with_window(SpectralWindow::Hann);

        let result = client.welch(&x, params).unwrap();

        let freqs: Vec<f64> = result.freqs.to_vec();
        let psd: Vec<f64> = result.psd.to_vec();

        // Find peak frequency
        let (peak_idx, _peak_val) = psd
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        let peak_freq = freqs[peak_idx];

        // Peak should be near 10 Hz
        assert!(
            (peak_freq - f0).abs() < 1.0,
            "Expected peak near {} Hz, got {} Hz",
            f0,
            peak_freq
        );
    }

    #[test]
    fn test_periodogram_dc() {
        let (client, device) = setup();

        // Constant signal should have all power at DC
        let n = 256;
        let signal = vec![1.0; n];
        let x = Tensor::from_slice(&signal, &[n], &device).unwrap();

        let params = PeriodogramParams::new(device.clone())
            .with_fs(1.0)
            .with_window(SpectralWindow::Rectangular);

        let result = client.periodogram(&x, params).unwrap();
        let psd: Vec<f64> = result.psd.to_vec();

        // DC component should dominate
        let dc_power = psd[0];
        let other_power: f64 = psd[1..].iter().sum();

        assert!(
            dc_power > other_power * 100.0,
            "DC power ({}) should dominate other power ({})",
            dc_power,
            other_power
        );
    }

    #[test]
    fn test_csd_same_signal() {
        let (client, device) = setup();

        // CSD of a signal with itself should be real and equal to PSD
        let fs = 100.0;
        let n = 512;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 5.0 * i as f64 / fs).sin())
            .collect();

        let x = Tensor::from_slice(&signal, &[n], &device).unwrap();
        let params = WelchParams::new(device.clone())
            .with_fs(fs)
            .with_nperseg(128);

        let csd_result = client.csd(&x, &x, params.clone()).unwrap();
        let welch_result = client.welch(&x, params).unwrap();

        let csd_mag = csd_result.magnitude().unwrap();
        let csd_mag_data: Vec<f64> = csd_mag.to_vec();
        let psd_data: Vec<f64> = welch_result.psd.to_vec();

        // CSD(x, x) should equal PSD(x)
        for (c, p) in csd_mag_data.iter().zip(psd_data.iter()) {
            let rel_err = if p.abs() > 1e-10 {
                (c - p).abs() / p.abs()
            } else {
                (c - p).abs()
            };
            assert!(rel_err < 0.01, "CSD magnitude should match PSD");
        }

        // Imaginary part should be near zero
        let csd_im: Vec<f64> = csd_result.pxy_imag.to_vec();
        let max_im = csd_im.iter().map(|x| x.abs()).fold(0.0, f64::max);
        let csd_re: Vec<f64> = csd_result.pxy_real.to_vec();
        let max_re = csd_re.iter().map(|x| x.abs()).fold(0.0, f64::max);
        assert!(
            max_im < max_re * 0.01,
            "Imaginary part should be near zero for same signal"
        );
    }

    #[test]
    fn test_coherence_perfect() {
        let (client, device) = setup();

        // Two identical signals should have coherence = 1
        let n = 512;
        let signal: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();

        let x = Tensor::from_slice(&signal, &[n], &device).unwrap();
        let y = x.clone();

        let params = WelchParams::new(device.clone()).with_nperseg(64);
        let result = client.coherence(&x, &y, params).unwrap();

        let cxy: Vec<f64> = result.cxy.to_vec();

        // All coherence values should be 1.0 (or very close)
        for &c in &cxy {
            assert!(
                (c - 1.0).abs() < 0.01,
                "Coherence should be 1.0 for identical signals, got {}",
                c
            );
        }
    }

    #[test]
    fn test_coherence_uncorrelated() {
        let (client, device) = setup();

        // Uncorrelated signals should have low coherence
        let n = 1024;

        // Use deterministic "random" signals with different phases
        let x_data: Vec<f64> = (0..n)
            .map(|i| {
                let phase1 = (i as f64 * 0.1).sin();
                let phase2 = (i as f64 * 0.23 + 1.5).sin();
                let phase3 = (i as f64 * 0.37 + 2.7).sin();
                phase1 + phase2 + phase3
            })
            .collect();

        let y_data: Vec<f64> = (0..n)
            .map(|i| {
                let phase1 = (i as f64 * 0.11 + 0.5).cos();
                let phase2 = (i as f64 * 0.29 + 1.1).cos();
                let phase3 = (i as f64 * 0.41 + 3.2).cos();
                phase1 + phase2 + phase3
            })
            .collect();

        let x = Tensor::from_slice(&x_data, &[n], &device).unwrap();
        let y = Tensor::from_slice(&y_data, &[n], &device).unwrap();

        let params = WelchParams::new(device.clone()).with_nperseg(256);
        let result = client.coherence(&x, &y, params).unwrap();

        let cxy: Vec<f64> = result.cxy.to_vec();
        let avg_coh: f64 = cxy.iter().sum::<f64>() / cxy.len() as f64;

        // Average coherence should be relatively low (though not zero due to finite sample)
        assert!(
            avg_coh < 0.5,
            "Average coherence should be low for uncorrelated signals, got {}",
            avg_coh
        );
    }

    #[test]
    fn test_lombscargle_uniform() {
        let (client, device) = setup();

        // For uniform sampling, Lomb-Scargle should behave like standard periodogram
        let n = 100;
        let f0 = 2.0;
        let t_data: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
        let x_data: Vec<f64> = t_data
            .iter()
            .map(|&ti| (2.0 * PI * f0 * ti).sin())
            .collect();

        let t = Tensor::from_slice(&t_data, &[n], &device).unwrap();
        let x = Tensor::from_slice(&x_data, &[n], &device).unwrap();

        // Test frequencies
        let freqs_data: Vec<f64> = (1..50).map(|i| i as f64 * 0.1).collect();
        let freqs = Tensor::from_slice(&freqs_data, &[freqs_data.len()], &device).unwrap();

        let power = client.lombscargle(&t, &x, &freqs, true).unwrap();
        let power_data: Vec<f64> = power.to_vec();

        // Find peak
        let (peak_idx, _) = power_data
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        let peak_freq = freqs_data[peak_idx];

        // Peak should be near 2.0 Hz
        assert!(
            (peak_freq - f0).abs() < 0.2,
            "Expected peak near {} Hz, got {} Hz",
            f0,
            peak_freq
        );
    }

    #[test]
    fn test_welch_detrend() {
        let (client, device) = setup();

        // Signal with linear trend + sine
        let n = 512;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                10.0 * t + (2.0 * PI * 5.0 * t).sin() // Linear trend + sine
            })
            .collect();

        let x = Tensor::from_slice(&signal, &[n], &device).unwrap();

        // Without detrend
        let params_no_detrend = WelchParams {
            fs: 1.0,
            window: SpectralWindow::Hann,
            nperseg: Some(128),
            noverlap: None,
            nfft: None,
            detrend: Detrend::None,
            scaling: PsdScaling::Density,
            onesided: true,
            device: device.clone(),
        };

        // With linear detrend
        let params_detrend = WelchParams {
            detrend: Detrend::Linear,
            ..params_no_detrend.clone()
        };

        let result_no_detrend = client.welch(&x, params_no_detrend).unwrap();
        let result_detrend = client.welch(&x, params_detrend).unwrap();

        let psd_no: Vec<f64> = result_no_detrend.psd.to_vec();
        let psd_de: Vec<f64> = result_detrend.psd.to_vec();

        // DC power should be much lower with detrending
        assert!(
            psd_de[0] < psd_no[0],
            "DC power should be lower with detrending"
        );
    }

    #[test]
    fn test_periodogram_scaling() {
        let (client, device) = setup();

        let n = 256;
        let fs = 100.0;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 10.0 * i as f64 / fs).sin())
            .collect();
        let x = Tensor::from_slice(&signal, &[n], &device).unwrap();

        // Density scaling
        let params_density = PeriodogramParams {
            fs,
            window: SpectralWindow::Hann,
            nfft: None,
            detrend: Detrend::None,
            scaling: PsdScaling::Density,
            onesided: true,
            device: device.clone(),
        };

        // Spectrum scaling
        let params_spectrum = PeriodogramParams {
            scaling: PsdScaling::Spectrum,
            ..params_density.clone()
        };

        let result_density = client.periodogram(&x, params_density).unwrap();
        let result_spectrum = client.periodogram(&x, params_spectrum).unwrap();

        // Spectrum scaling should be fs times larger than density scaling
        let psd_d: Vec<f64> = result_density.psd.to_vec();
        let psd_s: Vec<f64> = result_spectrum.psd.to_vec();

        let ratio = psd_s[10] / psd_d[10]; // Pick a non-edge bin
        assert!(
            (ratio - fs).abs() < 1.0,
            "Spectrum should be fs times density, got ratio {}",
            ratio
        );
    }
}
