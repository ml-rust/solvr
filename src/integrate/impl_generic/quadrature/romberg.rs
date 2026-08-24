//! Romberg integration using Richardson extrapolation.
use crate::DType;

use numr::error::{Error, Result};
use numr::ops::TensorOps;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use crate::integrate::{QuadResult, RombergOptions};

/// Romberg integration using Richardson extrapolation.
///
/// Applies Richardson extrapolation to the trapezoidal rule.
/// Each refinement level evaluates all new midpoints in a single batch,
/// avoiding GPU→CPU→GPU roundtrips in the refinement loop.
pub fn romberg_impl<R, C, F>(
    client: &C,
    f: F,
    a: f64,
    b: f64,
    options: &RombergOptions,
) -> Result<QuadResult<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + RuntimeClient<R>,
    F: Fn(&Tensor<R>) -> Result<Tensor<R>>,
{
    if a >= b {
        return Err(Error::InvalidArgument {
            arg: "a/b",
            reason: format!("romberg: invalid interval [{}, {}]", a, b),
        });
    }

    if options.max_levels == 0 {
        return Err(Error::InvalidArgument {
            arg: "max_levels",
            reason: "romberg: max_levels must be at least 1".to_string(),
        });
    }

    let mut neval = 0;
    let h = b - a;

    // Romberg table (only need current and previous row)
    let mut r_prev = vec![0.0; options.max_levels];
    let mut r_curr = vec![0.0; options.max_levels];

    // Initial trapezoidal estimate (1 interval) - batch evaluate endpoints
    let endpoints = Tensor::<R>::from_slice(&[a, b], &[2], client.device())?;
    let f_endpoints = f(&endpoints)?;
    let f_ab: Vec<f64> = f_endpoints.to_vec();
    neval += 2;

    r_prev[0] = h * (f_ab[0] + f_ab[1]) / 2.0;

    // Limit k to prevent overflow in 1 << k and 4^j calculations
    let max_k = options.max_levels.min(30);

    for k in 1..max_k {
        // Compute trapezoidal estimate with 2^k intervals
        let n: u64 = 1 << k;
        let h_k = h / n as f64;

        // Collect ALL new midpoints for batch evaluation
        // This is the critical fix: instead of calling eval_single() in a loop,
        // we batch all new midpoints into a single tensor evaluation
        let num_new_points = 1u64 << (k - 1);
        let new_points: Vec<f64> = (0..num_new_points)
            .map(|i| a + (2 * i + 1) as f64 * h_k)
            .collect();

        // Single batch evaluation - 2 device transfers total (up and down)
        let x_tensor =
            Tensor::<R>::from_slice(&new_points, &[num_new_points as usize], client.device())?;
        let f_values = f(&x_tensor)?;
        let f_data: Vec<f64> = f_values.to_vec();

        // Sum the function values
        let sum: f64 = f_data.iter().sum();
        neval += num_new_points as usize;

        // New trapezoidal estimate
        r_curr[0] = r_prev[0] / 2.0 + h_k * sum;

        // Richardson extrapolation
        for j in 1..=k {
            let factor = 4.0_f64.powi(j as i32);
            r_curr[j] = (factor * r_curr[j - 1] - r_prev[j - 1]) / (factor - 1.0);
        }

        // Check convergence
        let error = (r_curr[k] - r_prev[k - 1]).abs();
        let tolerance = options.atol + options.rtol * r_curr[k].abs();

        if error <= tolerance {
            return Ok(QuadResult {
                integral: Tensor::<R>::from_slice(&[r_curr[k]], &[], client.device())?,
                error,
                neval,
                converged: true,
            });
        }

        std::mem::swap(&mut r_prev, &mut r_curr);
    }

    // Return best estimate even if not converged
    let k = max_k - 1;
    let error = if k > 0 {
        (r_prev[k] - r_prev[k - 1]).abs()
    } else {
        r_prev[0].abs() * 0.1
    };

    Ok(QuadResult {
        integral: Tensor::<R>::from_slice(&[r_prev[k]], &[], client.device())?,
        error,
        neval,
        converged: false,
    })
}
