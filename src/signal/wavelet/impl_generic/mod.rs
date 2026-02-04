//! Generic implementations of wavelet transform algorithms.

use super::traits::{CwtResult, Dwt2dResult, DwtResult, ExtensionMode, WavedecResult};
use super::types::Wavelet;
use numr::error::{Error, Result};
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Compute single-level discrete wavelet transform.
pub fn dwt_impl<R, C>(
    _client: &C,
    x: &Tensor<R>,
    wavelet: &Wavelet,
    mode: ExtensionMode,
) -> Result<DwtResult<R>>
where
    R: Runtime,
    C: ScalarOps<R> + TensorOps<R> + RuntimeClient<R>,
{
    let x_data: Vec<f64> = x.to_vec();
    let n = x_data.len();
    let device = x.device();

    if n == 0 {
        return Err(Error::InvalidArgument {
            arg: "x",
            reason: "Input signal cannot be empty".to_string(),
        });
    }

    if wavelet.dec_lo.is_empty() {
        return Err(Error::InvalidArgument {
            arg: "wavelet",
            reason: "Wavelet has no filter coefficients (use DWT wavelets, not CWT)".to_string(),
        });
    }

    let filter_len = wavelet.filter_length();

    // Extend signal based on mode
    let extended = extend_signal(&x_data, filter_len, mode);

    // Convolve with low-pass and high-pass filters
    let approx_full = convolve_valid(&extended, &wavelet.dec_lo);
    let detail_full = convolve_valid(&extended, &wavelet.dec_hi);

    // Downsample by 2
    let approx: Vec<f64> = approx_full.iter().step_by(2).cloned().collect();
    let detail: Vec<f64> = detail_full.iter().step_by(2).cloned().collect();

    Ok(DwtResult {
        approx: Tensor::from_slice(&approx, &[approx.len()], device),
        detail: Tensor::from_slice(&detail, &[detail.len()], device),
    })
}

/// Compute inverse discrete wavelet transform.
pub fn idwt_impl<R, C>(
    _client: &C,
    coeffs: &DwtResult<R>,
    wavelet: &Wavelet,
    mode: ExtensionMode,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: ScalarOps<R> + TensorOps<R> + RuntimeClient<R>,
{
    let approx: Vec<f64> = coeffs.approx.to_vec();
    let detail: Vec<f64> = coeffs.detail.to_vec();
    let device = coeffs.approx.device();

    if wavelet.rec_lo.is_empty() {
        return Err(Error::InvalidArgument {
            arg: "wavelet",
            reason: "Wavelet has no filter coefficients".to_string(),
        });
    }

    // Upsample by 2 (insert zeros)
    let approx_up = upsample(&approx);
    let detail_up = upsample(&detail);

    // Extend for convolution
    let filter_len = wavelet.filter_length();
    let approx_ext = extend_signal(&approx_up, filter_len, mode);
    let detail_ext = extend_signal(&detail_up, filter_len, mode);

    // Convolve with reconstruction filters
    let approx_rec = convolve_valid(&approx_ext, &wavelet.rec_lo);
    let detail_rec = convolve_valid(&detail_ext, &wavelet.rec_hi);

    // Sum contributions
    let n = approx_rec.len().min(detail_rec.len());
    let result: Vec<f64> = approx_rec[..n]
        .iter()
        .zip(detail_rec[..n].iter())
        .map(|(&a, &d)| a + d)
        .collect();

    Ok(Tensor::from_slice(&result, &[result.len()], device))
}

/// Multi-level wavelet decomposition.
pub fn wavedec_impl<R, C>(
    client: &C,
    x: &Tensor<R>,
    wavelet: &Wavelet,
    mode: ExtensionMode,
    level: usize,
) -> Result<WavedecResult<R>>
where
    R: Runtime,
    C: ScalarOps<R> + TensorOps<R> + RuntimeClient<R>,
{
    if level == 0 {
        return Err(Error::InvalidArgument {
            arg: "level",
            reason: "Level must be at least 1".to_string(),
        });
    }

    let mut details = Vec::with_capacity(level);
    let mut current = x.clone();

    for _ in 0..level {
        let dwt_result = dwt_impl(client, &current, wavelet, mode)?;
        details.push(dwt_result.detail);
        current = dwt_result.approx;
    }

    Ok(WavedecResult {
        approx: current,
        details,
    })
}

/// Multi-level wavelet reconstruction.
pub fn waverec_impl<R, C>(
    client: &C,
    coeffs: &WavedecResult<R>,
    wavelet: &Wavelet,
    mode: ExtensionMode,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: ScalarOps<R> + TensorOps<R> + RuntimeClient<R>,
{
    let mut current = coeffs.approx.clone();

    // Reconstruct from coarsest to finest
    for detail in coeffs.details.iter().rev() {
        let dwt_coeffs = DwtResult {
            approx: current,
            detail: detail.clone(),
        };
        current = idwt_impl(client, &dwt_coeffs, wavelet, mode)?;
    }

    Ok(current)
}

/// 2D discrete wavelet transform.
pub fn dwt2_impl<R, C>(
    _client: &C,
    x: &Tensor<R>,
    wavelet: &Wavelet,
    mode: ExtensionMode,
) -> Result<Dwt2dResult<R>>
where
    R: Runtime,
    C: ScalarOps<R> + TensorOps<R> + RuntimeClient<R>,
{
    let shape = x.shape();
    if shape.len() != 2 {
        return Err(Error::InvalidArgument {
            arg: "x",
            reason: "Input must be 2D".to_string(),
        });
    }

    let rows = shape[0];
    let cols = shape[1];
    let x_data: Vec<f64> = x.to_vec();
    let device = x.device();

    if wavelet.dec_lo.is_empty() {
        return Err(Error::InvalidArgument {
            arg: "wavelet",
            reason: "Wavelet has no filter coefficients".to_string(),
        });
    }

    let filter_len = wavelet.filter_length();

    // Apply DWT to rows first
    let mut row_lo = Vec::new();
    let mut row_hi = Vec::new();

    for r in 0..rows {
        let row: Vec<f64> = (0..cols).map(|c| x_data[r * cols + c]).collect();
        let extended = extend_signal(&row, filter_len, mode);
        let lo = convolve_valid(&extended, &wavelet.dec_lo);
        let hi = convolve_valid(&extended, &wavelet.dec_hi);
        row_lo.extend(lo.iter().step_by(2));
        row_hi.extend(hi.iter().step_by(2));
    }

    let new_cols = row_lo.len() / rows;

    // Apply DWT to columns
    let mut ll = Vec::new();
    let mut lh = Vec::new();
    let mut hl = Vec::new();
    let mut hh = Vec::new();

    for c in 0..new_cols {
        // Column from row_lo
        let col_lo: Vec<f64> = (0..rows).map(|r| row_lo[r * new_cols + c]).collect();
        let extended_lo = extend_signal(&col_lo, filter_len, mode);
        let lo_lo = convolve_valid(&extended_lo, &wavelet.dec_lo);
        let lo_hi = convolve_valid(&extended_lo, &wavelet.dec_hi);
        ll.extend(lo_lo.iter().step_by(2));
        lh.extend(lo_hi.iter().step_by(2));

        // Column from row_hi
        let col_hi: Vec<f64> = (0..rows).map(|r| row_hi[r * new_cols + c]).collect();
        let extended_hi = extend_signal(&col_hi, filter_len, mode);
        let hi_lo = convolve_valid(&extended_hi, &wavelet.dec_lo);
        let hi_hi = convolve_valid(&extended_hi, &wavelet.dec_hi);
        hl.extend(hi_lo.iter().step_by(2));
        hh.extend(hi_hi.iter().step_by(2));
    }

    let new_rows = ll.len() / new_cols;

    // Transpose results (they're in column-major order)
    let ll_t = transpose(&ll, new_cols, new_rows);
    let lh_t = transpose(&lh, new_cols, new_rows);
    let hl_t = transpose(&hl, new_cols, new_rows);
    let hh_t = transpose(&hh, new_cols, new_rows);

    Ok(Dwt2dResult {
        ll: Tensor::from_slice(&ll_t, &[new_rows, new_cols], device),
        lh: Tensor::from_slice(&lh_t, &[new_rows, new_cols], device),
        hl: Tensor::from_slice(&hl_t, &[new_rows, new_cols], device),
        hh: Tensor::from_slice(&hh_t, &[new_rows, new_cols], device),
    })
}

/// Inverse 2D discrete wavelet transform.
pub fn idwt2_impl<R, C>(
    _client: &C,
    coeffs: &Dwt2dResult<R>,
    wavelet: &Wavelet,
    mode: ExtensionMode,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: ScalarOps<R> + TensorOps<R> + RuntimeClient<R>,
{
    let shape = coeffs.ll.shape();
    let rows = shape[0];
    let cols = shape[1];
    let device = coeffs.ll.device();

    let ll: Vec<f64> = coeffs.ll.to_vec();
    let lh: Vec<f64> = coeffs.lh.to_vec();
    let hl: Vec<f64> = coeffs.hl.to_vec();
    let hh: Vec<f64> = coeffs.hh.to_vec();

    let filter_len = wavelet.filter_length();

    // Reconstruct columns first
    let mut col_lo = vec![0.0; rows * 2 * cols];
    let mut col_hi = vec![0.0; rows * 2 * cols];

    for c in 0..cols {
        // LL + LH -> L column
        let ll_col: Vec<f64> = (0..rows).map(|r| ll[r * cols + c]).collect();
        let lh_col: Vec<f64> = (0..rows).map(|r| lh[r * cols + c]).collect();
        let ll_up = upsample(&ll_col);
        let lh_up = upsample(&lh_col);
        let ll_ext = extend_signal(&ll_up, filter_len, mode);
        let lh_ext = extend_signal(&lh_up, filter_len, mode);
        let ll_rec = convolve_valid(&ll_ext, &wavelet.rec_lo);
        let lh_rec = convolve_valid(&lh_ext, &wavelet.rec_hi);
        let n = ll_rec.len().min(lh_rec.len());
        for r in 0..n {
            col_lo[r * cols + c] = ll_rec[r] + lh_rec[r];
        }

        // HL + HH -> H column
        let hl_col: Vec<f64> = (0..rows).map(|r| hl[r * cols + c]).collect();
        let hh_col: Vec<f64> = (0..rows).map(|r| hh[r * cols + c]).collect();
        let hl_up = upsample(&hl_col);
        let hh_up = upsample(&hh_col);
        let hl_ext = extend_signal(&hl_up, filter_len, mode);
        let hh_ext = extend_signal(&hh_up, filter_len, mode);
        let hl_rec = convolve_valid(&hl_ext, &wavelet.rec_lo);
        let hh_rec = convolve_valid(&hh_ext, &wavelet.rec_hi);
        let n = hl_rec.len().min(hh_rec.len());
        for r in 0..n {
            col_hi[r * cols + c] = hl_rec[r] + hh_rec[r];
        }
    }

    let new_rows = rows * 2;

    // Reconstruct rows
    let mut result = Vec::with_capacity(new_rows * cols * 2);

    for r in 0..new_rows {
        let lo_row: Vec<f64> = (0..cols).map(|c| col_lo[r * cols + c]).collect();
        let hi_row: Vec<f64> = (0..cols).map(|c| col_hi[r * cols + c]).collect();
        let lo_up = upsample(&lo_row);
        let hi_up = upsample(&hi_row);
        let lo_ext = extend_signal(&lo_up, filter_len, mode);
        let hi_ext = extend_signal(&hi_up, filter_len, mode);
        let lo_rec = convolve_valid(&lo_ext, &wavelet.rec_lo);
        let hi_rec = convolve_valid(&hi_ext, &wavelet.rec_hi);
        let n = lo_rec.len().min(hi_rec.len());
        for c in 0..n {
            result.push(lo_rec[c] + hi_rec[c]);
        }
    }

    let new_cols = result.len() / new_rows;
    Ok(Tensor::from_slice(&result, &[new_rows, new_cols], device))
}

/// Compute continuous wavelet transform.
pub fn cwt_impl<R, C>(
    _client: &C,
    x: &Tensor<R>,
    scales: &Tensor<R>,
    wavelet: &Wavelet,
) -> Result<CwtResult<R>>
where
    R: Runtime,
    C: ScalarOps<R> + TensorOps<R> + RuntimeClient<R>,
{
    let x_data: Vec<f64> = x.to_vec();
    let scales_data: Vec<f64> = scales.to_vec();
    let n = x_data.len();
    let num_scales = scales_data.len();
    let device = x.device();

    if n == 0 {
        return Err(Error::InvalidArgument {
            arg: "x",
            reason: "Input signal cannot be empty".to_string(),
        });
    }

    if !wavelet.is_cwt_wavelet() {
        return Err(Error::InvalidArgument {
            arg: "wavelet",
            reason: "Must use CWT wavelet (Morlet or MexicanHat)".to_string(),
        });
    }

    let mut coeffs_real = Vec::with_capacity(num_scales * n);
    let mut coeffs_imag = Vec::with_capacity(num_scales * n);

    for &scale in &scales_data {
        // Generate wavelet at this scale
        let half_width = (scale * 10.0).ceil() as usize; // 10 sigma range
        let wavelet_len = 2 * half_width + 1;
        let t: Vec<f64> = (0..wavelet_len)
            .map(|i| i as f64 - half_width as f64)
            .collect();

        let psi = wavelet
            .evaluate(&t, scale)
            .unwrap_or_else(|| vec![0.0; wavelet_len]);

        // Convolve x with conjugate of wavelet
        // For real wavelets like Mexican Hat, this is just convolution
        // For Morlet, we need complex convolution

        for i in 0..n {
            let mut re = 0.0;
            let im = 0.0;

            for (j, &w) in psi.iter().enumerate() {
                let k = i as isize + j as isize - half_width as isize;
                if k >= 0 && (k as usize) < n {
                    re += x_data[k as usize] * w;
                }
            }

            // For Mexican Hat (real wavelet), imaginary part is 0
            // For Morlet, we'd need complex wavelet values
            coeffs_real.push(re);
            coeffs_imag.push(im);
        }
    }

    Ok(CwtResult {
        coeffs_real: Tensor::from_slice(&coeffs_real, &[num_scales, n], device),
        coeffs_imag: Tensor::from_slice(&coeffs_imag, &[num_scales, n], device),
        scales: scales.clone(),
    })
}

// ============================================================================
// Helper functions
// ============================================================================

/// Extend signal for convolution based on mode.
fn extend_signal(x: &[f64], filter_len: usize, mode: ExtensionMode) -> Vec<f64> {
    let n = x.len();
    let pad = filter_len - 1;

    let mut extended = Vec::with_capacity(n + 2 * pad);

    // Left extension
    match mode {
        ExtensionMode::Zero => {
            extended.extend(std::iter::repeat_n(0.0, pad));
        }
        ExtensionMode::Constant => {
            extended.extend(std::iter::repeat_n(x[0], pad));
        }
        ExtensionMode::Symmetric => {
            for i in (0..pad).rev() {
                let idx = if i < n { i } else { 2 * n - 2 - i };
                extended.push(x[idx.min(n - 1)]);
            }
        }
        ExtensionMode::Periodic => {
            for i in (0..pad).rev() {
                extended.push(x[i % n]);
            }
        }
        ExtensionMode::Smooth => {
            // Simple linear extrapolation
            let slope = if n > 1 { x[1] - x[0] } else { 0.0 };
            for i in (0..pad).rev() {
                extended.push(x[0] - slope * (i + 1) as f64);
            }
        }
    }

    // Original signal
    extended.extend_from_slice(x);

    // Right extension
    match mode {
        ExtensionMode::Zero => {
            extended.extend(std::iter::repeat_n(0.0, pad));
        }
        ExtensionMode::Constant => {
            extended.extend(std::iter::repeat_n(x[n - 1], pad));
        }
        ExtensionMode::Symmetric => {
            for i in 0..pad {
                let idx = if n > 1 + i { n - 2 - i } else { i % n };
                extended.push(x[idx]);
            }
        }
        ExtensionMode::Periodic => {
            for i in 0..pad {
                extended.push(x[i % n]);
            }
        }
        ExtensionMode::Smooth => {
            let slope = if n > 1 { x[n - 1] - x[n - 2] } else { 0.0 };
            for i in 0..pad {
                extended.push(x[n - 1] + slope * (i + 1) as f64);
            }
        }
    }

    extended
}

/// Valid convolution (no padding, output shorter than input).
fn convolve_valid(x: &[f64], h: &[f64]) -> Vec<f64> {
    let n = x.len();
    let m = h.len();
    if n < m {
        return vec![];
    }

    let out_len = n - m + 1;
    let mut result = Vec::with_capacity(out_len);

    for i in 0..out_len {
        let mut sum = 0.0;
        for (j, &hj) in h.iter().enumerate() {
            sum += x[i + j] * hj;
        }
        result.push(sum);
    }

    result
}

/// Upsample by inserting zeros.
fn upsample(x: &[f64]) -> Vec<f64> {
    let mut result = Vec::with_capacity(x.len() * 2);
    for &xi in x {
        result.push(xi);
        result.push(0.0);
    }
    result
}

/// Transpose a flattened matrix.
fn transpose(data: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut result = vec![0.0; data.len()];
    for r in 0..rows {
        for c in 0..cols {
            result[c * rows + r] = data[r * cols + c];
        }
    }
    result
}
