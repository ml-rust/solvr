//! Adaptive Gauss-Kronrod quadrature.
use crate::DType;

use numr::error::{Error, Result};
use numr::ops::TensorOps;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use crate::integrate::{QuadOptions, QuadResult};

/// Adaptive Gauss-Kronrod quadrature.
///
/// Uses the G7-K15 rule with adaptive interval subdivision.
/// All 15 evaluation points per interval are computed in a single batch.
pub fn quad_impl<R, C, F>(
    client: &C,
    f: F,
    a: f64,
    b: f64,
    options: &QuadOptions,
) -> Result<QuadResult<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + RuntimeClient<R>,
    F: Fn(&Tensor<R>) -> Result<Tensor<R>>,
{
    if a >= b {
        return Err(Error::InvalidArgument {
            arg: "a/b",
            reason: format!("quad: invalid interval [{}, {}]", a, b),
        });
    }

    if options.limit == 0 {
        return Err(Error::InvalidArgument {
            arg: "limit",
            reason: "quad: limit must be at least 1".to_string(),
        });
    }

    // Work queue for intervals: (a, b, integral, error)
    let mut intervals: Vec<(f64, f64, f64, f64)> = Vec::new();
    let mut neval = 0;

    // Initial evaluation
    let (integral, error, evals) = gauss_kronrod_15(client, &f, a, b)?;
    neval += evals;

    intervals.push((a, b, integral, error));

    let mut total_integral = integral;
    let mut total_error = error;
    let mut subdivisions = 0;

    while subdivisions < options.limit {
        // Check convergence
        let tolerance = options.atol + options.rtol * total_integral.abs();
        if total_error <= tolerance {
            return Ok(QuadResult {
                integral: Tensor::<R>::from_slice(&[total_integral], &[], client.device())?,
                error: total_error,
                neval,
                converged: true,
            });
        }

        // Find interval with largest error
        // Use proper error handling instead of unwrap() - handles NaN gracefully
        let max_idx = match intervals
            .iter()
            .enumerate()
            .max_by(|a, b| {
                a.1.3
                    .partial_cmp(&b.1.3)
                    .unwrap_or(std::cmp::Ordering::Less)
            })
            .map(|(i, _)| i)
        {
            Some(idx) => idx,
            None => {
                // This should never happen if limit > 0, but handle gracefully
                return Err(Error::InvalidArgument {
                    arg: "intervals",
                    reason: "quad: internal error - no intervals available".to_string(),
                });
            }
        };

        let (ia, ib, old_integral, old_error) = intervals.swap_remove(max_idx);

        // Subdivide at midpoint
        let mid = (ia + ib) / 2.0;

        let (int1, err1, evals1) = gauss_kronrod_15(client, &f, ia, mid)?;
        let (int2, err2, evals2) = gauss_kronrod_15(client, &f, mid, ib)?;
        neval += evals1 + evals2;

        // Update totals
        total_integral = total_integral - old_integral + int1 + int2;
        total_error = total_error - old_error + err1 + err2;

        intervals.push((ia, mid, int1, err1));
        intervals.push((mid, ib, int2, err2));

        subdivisions += 1;
    }

    // Didn't converge within limit
    Ok(QuadResult {
        integral: Tensor::<R>::from_slice(&[total_integral], &[], client.device())?,
        error: total_error,
        neval,
        converged: false,
    })
}

/// Gauss-Kronrod 15-point rule (G7-K15).
///
/// Returns (integral, error, neval).
/// All 15 points are evaluated in a single batch.
fn gauss_kronrod_15<R, C, F>(client: &C, f: &F, a: f64, b: f64) -> Result<(f64, f64, usize)>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + RuntimeClient<R>,
    F: Fn(&Tensor<R>) -> Result<Tensor<R>>,
{
    // Kronrod nodes (15 points, including 7 Gauss points)
    const XGK: [f64; 15] = [
        -0.9914553711208126,
        -0.9491079123427585,
        -0.8648644233597691,
        -0.7415311855993944,
        -0.5860872354676911,
        -0.4058451513773972,
        -0.2077849550078985,
        0.0,
        0.2077849550078985,
        0.4058451513773972,
        0.5860872354676911,
        0.7415311855993944,
        0.8648644233597691,
        0.9491079123427585,
        0.9914553711208126,
    ];

    // Kronrod weights (15 points)
    const WGK: [f64; 15] = [
        0.022935322010529224,
        0.063_092_092_629_978_56,
        0.10479001032225018,
        0.14065325971552592,
        0.169_004_726_639_267_9,
        0.190_350_578_064_785_4,
        0.20443294007529889,
        0.20948214108472782,
        0.20443294007529889,
        0.190_350_578_064_785_4,
        0.169_004_726_639_267_9,
        0.14065325971552592,
        0.10479001032225018,
        0.063_092_092_629_978_56,
        0.022935322010529224,
    ];

    // Gauss weights (7 points at indices 1,3,5,7,9,11,13)
    const WG: [f64; 7] = [
        0.129_484_966_168_869_7,
        0.27970539148927664,
        0.381_830_050_505_118_9,
        0.417_959_183_673_469_4,
        0.381_830_050_505_118_9,
        0.27970539148927664,
        0.129_484_966_168_869_7,
    ];

    let mid = (a + b) / 2.0;
    let half_width = (b - a) / 2.0;

    // Create tensor of evaluation points - single batch
    let eval_points: Vec<f64> = XGK.iter().map(|&x| mid + half_width * x).collect();
    let x_tensor = Tensor::<R>::from_slice(&eval_points, &[15], client.device())?;

    // Evaluate function at all 15 points in one call
    let f_values = f(&x_tensor)?;
    let fvals: Vec<f64> = f_values.to_vec();

    // Compute Kronrod (15-point) result
    let mut result_kronrod = 0.0;
    for (i, &fval) in fvals.iter().enumerate() {
        result_kronrod += WGK[i] * fval;
    }
    result_kronrod *= half_width;

    // Compute Gauss (7-point) result using odd-indexed points
    let mut result_gauss = 0.0;
    for (i, &w) in WG.iter().enumerate() {
        result_gauss += w * fvals[2 * i + 1];
    }
    result_gauss *= half_width;

    // Error estimate
    let error = (result_kronrod - result_gauss).abs();

    Ok((result_kronrod, error, 15))
}
