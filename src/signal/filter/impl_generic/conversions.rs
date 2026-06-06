//! Generic implementations of filter representation conversions.
//!
//! Converts between transfer function (b, a) and zero-pole-gain (zpk) representations.
//!
//! Note: SOS conversions (tf2sos, zpk2sos, sos2tf, sos2zpk) are CPU-only because they
//! involve inherently sequential algorithms (pole/zero pairing, quadratic root finding)
//! with tiny data sizes. See cpu/conversions.rs for implementations.
use crate::DType;

use crate::signal::filter::types::{TransferFunction, ZpkFilter};
use numr::algorithm::polynomial::PolynomialAlgorithms;
use numr::error::{Error, Result};
use numr::ops::ScalarOps;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Convert transfer function to zeros, poles, and gain.
///
/// Uses polynomial root finding via companion matrix eigendecomposition.
/// The only `to_vec()` calls are for extracting two scalar coefficients (b0, a0)
/// which is acceptable at the API boundary.
pub fn tf2zpk_impl<R, C>(client: &C, tf: &TransferFunction<R>) -> Result<ZpkFilter<R>>
where
    R: Runtime<DType = DType>,
    C: PolynomialAlgorithms<R> + ScalarOps<R> + RuntimeClient<R>,
{
    let b = &tf.b;
    let a = &tf.a;

    // Validate inputs
    if b.ndim() != 1 || a.ndim() != 1 {
        return Err(Error::InvalidArgument {
            arg: "tf",
            reason: "Transfer function coefficients must be 1D".to_string(),
        });
    }

    if b.shape()[0] == 0 || a.shape()[0] == 0 {
        return Err(Error::InvalidArgument {
            arg: "tf",
            reason: "Transfer function coefficients cannot be empty".to_string(),
        });
    }

    // Coefficients are in descending order, need to reverse for polyroots
    // which expects ascending order
    let b_ascending = b.flip(0)?.contiguous()?;
    let a_ascending = a.flip(0)?.contiguous()?;

    // Find zeros (roots of numerator)
    let zeros = if b.shape()[0] > 1 {
        client.polyroots(&b_ascending)?
    } else {
        // Constant numerator has no zeros
        let device = b.device();
        numr::algorithm::polynomial::types::PolynomialRoots {
            roots_real: Tensor::zeros(&[0], b.dtype(), device),
            roots_imag: Tensor::zeros(&[0], b.dtype(), device),
        }
    };

    // Find poles (roots of denominator)
    let poles = if a.shape()[0] > 1 {
        client.polyroots(&a_ascending)?
    } else {
        let device = a.device();
        numr::algorithm::polynomial::types::PolynomialRoots {
            roots_real: Tensor::zeros(&[0], a.dtype(), device),
            roots_imag: Tensor::zeros(&[0], a.dtype(), device),
        }
    };

    // Compute gain: ratio of leading coefficients (highest power = index 0)
    let b0: f64 = b.narrow(0, 0, 1)?.item()?;
    let a0: f64 = a.narrow(0, 0, 1)?.item()?;
    let gain = b0 / a0;

    Ok(ZpkFilter::new(
        zeros.roots_real,
        zeros.roots_imag,
        poles.roots_real,
        poles.roots_imag,
        gain,
    ))
}

/// Convert zeros, poles, and gain to transfer function.
///
/// Uses polynomial multiplication via convolution.
/// All operations are tensor-based.
pub fn zpk2tf_impl<R, C>(client: &C, zpk: &ZpkFilter<R>) -> Result<TransferFunction<R>>
where
    R: Runtime<DType = DType>,
    C: PolynomialAlgorithms<R> + ScalarOps<R> + RuntimeClient<R>,
{
    let device = zpk.zeros_real.device();

    // Build numerator from zeros
    let b = if zpk.num_zeros() == 0 {
        // No zeros: numerator is just [gain]
        Tensor::from_slice(&[zpk.gain], &[1], device)
    } else {
        // Build polynomial from roots, then scale by gain
        let b_monic = client.polyfromroots(&zpk.zeros_real, &zpk.zeros_imag)?;
        // polyfromroots returns ascending order, we need descending
        let b_desc = b_monic.flip(0)?.contiguous()?;
        client.mul_scalar(&b_desc, zpk.gain)?
    };

    // Build denominator from poles
    let a = if zpk.num_poles() == 0 {
        // No poles: denominator is just [1]
        Tensor::from_slice(&[1.0], &[1], device)
    } else {
        let a_monic = client.polyfromroots(&zpk.poles_real, &zpk.poles_imag)?;
        // polyfromroots returns ascending order, we need descending
        a_monic.flip(0)?.contiguous()?
    };

    Ok(TransferFunction::new(b, a))
}
