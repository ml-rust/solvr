//! Akima spline interpolation generic implementation using tensor operations.
//!
//! All computation uses tensor ops - data stays on device.
use crate::DType;

use crate::interpolate::error::InterpolateResult;
use numr::ops::{CompareOps, ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Compute slopes using the Akima method with pure tensor operations.
///
/// The Akima method uses weights based on the absolute differences between
/// adjacent slopes to reduce sensitivity to outliers.
///
/// All computation stays on device - no to_vec() calls.
pub fn akima_slopes<R, C>(client: &C, x: &Tensor<R>, y: &Tensor<R>) -> InterpolateResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + CompareOps<R> + RuntimeClient<R>,
{
    let n = x.shape()[0];
    let device = client.device();

    if n == 2 {
        // For 2 points, slope is just the secant: (y[1] - y[0]) / (x[1] - x[0])
        // Use narrow to slice and compute
        let x0 = x.narrow(0, 0, 1)?;
        let x1 = x.narrow(0, 1, 1)?;
        let y0 = y.narrow(0, 0, 1)?;
        let y1 = y.narrow(0, 1, 1)?;
        let dx = client.sub(&x1, &x0)?;
        let dy = client.sub(&y1, &y0)?;
        let secant = client.div(&dy, &dx)?;
        // Return [secant, secant] - cat two copies
        return Ok(client.cat(&[&secant, &secant], 0)?);
    }

    // Step 1: Compute secants between adjacent points using tensor slicing
    // m[i] = (y[i+1] - y[i]) / (x[i+1] - x[i]) for i = 0..n-2
    let x_lo = x.narrow(0, 0, n - 1)?; // x[0..n-1]
    let x_hi = x.narrow(0, 1, n - 1)?; // x[1..n]
    let y_lo = y.narrow(0, 0, n - 1)?; // y[0..n-1]
    let y_hi = y.narrow(0, 1, n - 1)?; // y[1..n]

    let dx = client.sub(&x_hi, &x_lo)?;
    let dy = client.sub(&y_hi, &y_lo)?;
    let m = client.div(&dy, &dx)?; // Shape: [n-1]

    // Step 2: Extend slopes at boundaries using parabolic extrapolation
    // m[-2] = 3*m[0] - 2*m[1]
    // m[-1] = 2*m[0] - m[1]
    // m[n-1] = 2*m[n-2] - m[n-3]
    // m[n] = 3*m[n-2] - 2*m[n-3]
    let m0 = m.narrow(0, 0, 1)?; // m[0]
    let m1 = m.narrow(0, 1, 1)?; // m[1]
    let m_last = m.narrow(0, n - 2, 1)?; // m[n-2] (last element)
    let m_second_last = m.narrow(0, n - 3, 1)?; // m[n-3] (second to last)

    // m_minus2 = 3*m[0] - 2*m[1]
    let m0_3 = client.mul_scalar(&m0, 3.0)?;
    let m1_2 = client.mul_scalar(&m1, 2.0)?;
    let m_minus2 = client.sub(&m0_3, &m1_2)?;

    // m_minus1 = 2*m[0] - m[1]
    let m0_2 = client.mul_scalar(&m0, 2.0)?;
    let m_minus1 = client.sub(&m0_2, &m1)?;

    // m_n = 2*m[n-2] - m[n-3]
    let m_last_2 = client.mul_scalar(&m_last, 2.0)?;
    let m_n = client.sub(&m_last_2, &m_second_last)?;

    // m_n_plus1 = 3*m[n-2] - 2*m[n-3]
    let m_last_3 = client.mul_scalar(&m_last, 3.0)?;
    let m_second_last_2 = client.mul_scalar(&m_second_last, 2.0)?;
    let m_n_plus1 = client.sub(&m_last_3, &m_second_last_2)?;

    // Build extended m array: [m_minus2, m_minus1, m[0..n-1], m_n, m_n_plus1]
    // Shape: [n+3]
    let m_ext = client.cat(&[&m_minus2, &m_minus1, &m, &m_n, &m_n_plus1], 0)?;

    // Step 3: Compute slopes at each point using vectorized Akima formula
    // For slope[i] (i = 0..n), we use m_ext indices:
    //   m_ext[i] = m[-2+i], m_ext[i+1] = m[-1+i], m_ext[i+2] = m[i], m_ext[i+3] = m[i+1]
    // So:
    //   dm1 = |m_ext[i+3] - m_ext[i+2]| = |m[i+1] - m[i]|
    //   dm2 = |m_ext[i+1] - m_ext[i]| = |m[i-1] - m[i-2]|
    //   slope = (dm1 * m_ext[i+1] + dm2 * m_ext[i+2]) / (dm1 + dm2)
    //         = (dm1 * m[i-1] + dm2 * m[i]) / (dm1 + dm2)
    // With fallback: if dm1 + dm2 < eps, slope = 0.5 * (m[i-1] + m[i])

    // Slice m_ext into 4 views of length n
    let m_i_minus2 = m_ext.narrow(0, 0, n)?; // m_ext[0..n] = m[-2..-2+n]
    let m_i_minus1 = m_ext.narrow(0, 1, n)?; // m_ext[1..n+1] = m[-1..-1+n]
    let m_i = m_ext.narrow(0, 2, n)?; // m_ext[2..n+2] = m[0..n]
    let m_i_plus1 = m_ext.narrow(0, 3, n)?; // m_ext[3..n+3] = m[1..n+1]

    // dm1 = |m[i+1] - m[i]|
    let diff1 = client.sub(&m_i_plus1, &m_i)?;
    let dm1 = client.abs(&diff1)?;

    // dm2 = |m[i-1] - m[i-2]|
    let diff2 = client.sub(&m_i_minus1, &m_i_minus2)?;
    let dm2 = client.abs(&diff2)?;

    // denom = dm1 + dm2
    let denom = client.add(&dm1, &dm2)?;

    // Simple slope (fallback): 0.5 * (m[i-1] + m[i])
    let sum_slopes = client.add(&m_i_minus1, &m_i)?;
    let slope_simple = client.mul_scalar(&sum_slopes, 0.5)?;

    // For the weighted slope, we need to avoid division by zero.
    // Add a small epsilon to the denominator to prevent NaN.
    // When denom is very small, the weighted and simple slopes converge anyway.
    let epsilon_data = vec![1e-14; n];
    let epsilon = Tensor::<R>::from_slice(&epsilon_data, &[n], device);

    // Make tensors contiguous to avoid issues with non-contiguous views
    let denom_contig = denom.contiguous();
    let dm1_contig = dm1.contiguous();
    let dm2_contig = dm2.contiguous();
    let m_i_minus1_contig = m_i_minus1.contiguous();
    let m_i_contig = m_i.contiguous();
    let slope_simple_contig = slope_simple.contiguous();

    // Safe denominator: denom + epsilon
    let safe_denom = client.add(&denom_contig, &epsilon)?;

    // Weighted slope with safe denominator: (dm1 * m[i-1] + dm2 * m[i]) / safe_denom
    let term1 = client.mul(&dm1_contig, &m_i_minus1_contig)?;
    let term2 = client.mul(&dm2_contig, &m_i_contig)?;
    let numer = client.add(&term1, &term2)?;
    let slope_weighted = client.div(&numer, &safe_denom)?;

    // Blend: when denom is large, use weighted; when small, blend towards simple
    // weight = denom / safe_denom = denom / (denom + epsilon)
    // For large denom: weight ≈ 1
    // For small denom: weight ≈ 0
    let weight = client.div(&denom_contig, &safe_denom)?;

    // one_minus_weight = 1 - weight
    let ones_data = vec![1.0; n];
    let ones = Tensor::<R>::from_slice(&ones_data, &[n], device);
    let one_minus_weight = client.sub(&ones, &weight)?;

    // slopes = slope_weighted * weight + slope_simple * (1 - weight)
    let weighted_term = client.mul(&slope_weighted, &weight)?;
    let simple_term = client.mul(&slope_simple_contig, &one_minus_weight)?;
    let slopes = client.add(&weighted_term, &simple_term)?;

    Ok(slopes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    fn setup() -> (CpuDevice, CpuClient) {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        (device, client)
    }

    #[test]
    fn test_akima_slopes_linear_data() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0], &[5], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[1.0, 3.0, 5.0, 7.0, 9.0], &[5], &device);

        let slopes = akima_slopes(&client, &x, &y).unwrap();
        let slopes_vec: Vec<f64> = slopes.to_vec();

        assert_eq!(slopes_vec.len(), 5);
        // For linear data, slopes should all be 2.0
        for &slope in &slopes_vec {
            assert!((slope - 2.0).abs() < 1e-10);
        }
    }
}
