//! N-dimensional adaptive quadrature.
//!
//! Provides dblquad, tplquad, and general nquad for multi-dimensional integrals
//! using nested adaptive quadrature.

use std::cell::Cell;

use numr::error::Result;
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use crate::integrate::traits::{NQuadOptions, QuadResult};

use super::super::super::traits::QuadOptions;
use super::quad::quad_impl;

/// Double integral implementation.
///
/// Computes ∫∫ f(x, y) dy dx over [a, b] × [gfun(x), hfun(x)]
/// using nested adaptive quadrature.
pub fn dblquad_impl<R, C, F, G, H>(
    client: &C,
    f: F,
    a: f64,
    b: f64,
    gfun: G,
    hfun: H,
    options: &NQuadOptions,
) -> Result<QuadResult<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
    F: Fn(&Tensor<R>, &Tensor<R>) -> Result<Tensor<R>>,
    G: Fn(f64) -> f64,
    H: Fn(f64) -> f64,
{
    let device = client.device();
    let quad_options = QuadOptions {
        rtol: options.rtol,
        atol: options.atol,
        limit: options.limit,
    };

    // Use Cell for interior mutability inside Fn closure
    let total_neval = Cell::new(0usize);

    // Outer integral over x
    let outer_result = quad_impl(
        client,
        |x_tensor| {
            // For each x value, compute the inner integral
            let x_vec: Vec<f64> = x_tensor.to_vec();
            let n = x_vec.len();
            let mut results = Vec::with_capacity(n);

            for &x in &x_vec {
                let y_lower = gfun(x);
                let y_upper = hfun(x);

                if (y_upper - y_lower).abs() < 1e-14 {
                    results.push(0.0);
                    continue;
                }

                // Inner integral over y at fixed x
                let x_scalar = Tensor::<R>::from_slice(&[x], &[1], device);

                let inner_result = quad_impl(
                    client,
                    |y_tensor| f(&x_scalar.broadcast_to(y_tensor.shape())?, y_tensor),
                    y_lower,
                    y_upper,
                    &quad_options,
                )?;

                total_neval.set(total_neval.get() + inner_result.neval);
                let inner_val: Vec<f64> = inner_result.integral.to_vec();
                results.push(inner_val[0]);
            }

            Ok(Tensor::<R>::from_slice(&results, &[n], device))
        },
        a,
        b,
        &quad_options,
    )?;

    let final_neval = total_neval.get() + outer_result.neval;
    let outer_error = outer_result.error;

    Ok(QuadResult {
        integral: outer_result.integral,
        error: outer_error,
        neval: final_neval,
        converged: outer_result.converged,
    })
}

/// N-dimensional adaptive quadrature implementation.
///
/// Uses nested quadrature with tensor function evaluation.
/// Currently supports 1D and 2D integrals.
/// For higher dimensions (3+), use Monte Carlo methods (monte_carlo_impl or qmc_impl).
pub fn nquad_impl<R, C, F>(
    client: &C,
    f: F,
    bounds: &[(f64, f64)],
    options: &NQuadOptions,
) -> Result<QuadResult<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
    F: Fn(&Tensor<R>) -> Result<Tensor<R>>,
{
    let device = client.device();
    let n_dims = bounds.len();

    if n_dims == 0 {
        return Err(numr::error::Error::InvalidArgument {
            arg: "bounds",
            reason: "Bounds cannot be empty".to_string(),
        });
    }

    let quad_options = QuadOptions {
        rtol: options.rtol,
        atol: options.atol,
        limit: options.limit,
    };

    // For 1D, use standard quad
    if n_dims == 1 {
        return quad_impl(
            client,
            |x| {
                // Reshape x from [n] to [n, 1] for consistency
                let shape = x.shape();
                let n = shape[0];
                let x_2d = x.reshape(&[n, 1])?;
                f(&x_2d)
            },
            bounds[0].0,
            bounds[0].1,
            &quad_options,
        );
    }

    // For 2D, use explicit nested quadrature (no recursion)
    if n_dims == 2 {
        let (x_lo, x_hi) = bounds[0];
        let (y_lo, y_hi) = bounds[1];

        let total_neval = Cell::new(0usize);

        let outer = quad_impl(
            client,
            |x_tensor| {
                let x_vec: Vec<f64> = x_tensor.to_vec();
                let n = x_vec.len();
                let mut results = Vec::with_capacity(n);

                for &x in &x_vec {
                    let inner = quad_impl(
                        client,
                        |y_tensor| {
                            // Create coordinate tensor [n, 2]
                            let y_vec: Vec<f64> = y_tensor.to_vec();
                            let m = y_vec.len();
                            let mut coords = Vec::with_capacity(m * 2);
                            for &y in &y_vec {
                                coords.push(x);
                                coords.push(y);
                            }
                            f(&Tensor::<R>::from_slice(&coords, &[m, 2], device))
                        },
                        y_lo,
                        y_hi,
                        &quad_options,
                    )?;
                    total_neval.set(total_neval.get() + inner.neval);
                    let inner_val: Vec<f64> = inner.integral.to_vec();
                    results.push(inner_val[0]);
                }

                Ok(Tensor::<R>::from_slice(&results, &[n], device))
            },
            x_lo,
            x_hi,
            &quad_options,
        )?;

        return Ok(QuadResult {
            integral: outer.integral,
            error: outer.error,
            neval: total_neval.get() + outer.neval,
            converged: outer.converged,
        });
    }

    // For dimensions > 2, recommend Monte Carlo
    Err(numr::error::Error::InvalidArgument {
        arg: "bounds",
        reason: format!(
            "nquad only supports up to 2 dimensions. For {} dimensions, use monte_carlo or qmc_quad.",
            n_dims
        ),
    })
}

/// Triple integral - convenience wrapper around nquad.
#[allow(clippy::too_many_arguments)]
pub fn tplquad_impl<R, C, F, G1, H1, G2, H2>(
    client: &C,
    f: F,
    a: f64,
    b: f64,
    gfun: G1,
    hfun: H1,
    qfun: G2,
    rfun: H2,
    options: &NQuadOptions,
) -> Result<QuadResult<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
    F: Fn(&Tensor<R>, &Tensor<R>, &Tensor<R>) -> Result<Tensor<R>>,
    G1: Fn(f64) -> f64,
    H1: Fn(f64) -> f64,
    G2: Fn(f64, f64) -> f64,
    H2: Fn(f64, f64) -> f64,
{
    let device = client.device();

    let quad_options = QuadOptions {
        rtol: options.rtol,
        atol: options.atol,
        limit: options.limit,
    };

    // Outermost integral over x
    quad_impl(
        client,
        |x_tensor| {
            let x_vec: Vec<f64> = x_tensor.to_vec();
            let n = x_vec.len();
            let mut results = Vec::with_capacity(n);

            for &x in &x_vec {
                let y_lower = gfun(x);
                let y_upper = hfun(x);

                if (y_upper - y_lower).abs() < 1e-14 {
                    results.push(0.0);
                    continue;
                }

                // Middle integral over y
                let x_scalar = Tensor::<R>::from_slice(&[x], &[1], device);

                let middle_result = quad_impl(
                    client,
                    |y_tensor| {
                        let y_vec: Vec<f64> = y_tensor.to_vec();
                        let m = y_vec.len();
                        let mut y_results = Vec::with_capacity(m);

                        for &y in &y_vec {
                            let z_lower = qfun(x, y);
                            let z_upper = rfun(x, y);

                            if (z_upper - z_lower).abs() < 1e-14 {
                                y_results.push(0.0);
                                continue;
                            }

                            let y_scalar = Tensor::<R>::from_slice(&[y], &[1], device);

                            // Innermost integral over z
                            let inner_result = quad_impl(
                                client,
                                |z_tensor| {
                                    f(
                                        &x_scalar.broadcast_to(z_tensor.shape())?,
                                        &y_scalar.broadcast_to(z_tensor.shape())?,
                                        z_tensor,
                                    )
                                },
                                z_lower,
                                z_upper,
                                &quad_options,
                            )?;

                            let inner_val: Vec<f64> = inner_result.integral.to_vec();
                            y_results.push(inner_val[0]);
                        }

                        Ok(Tensor::<R>::from_slice(&y_results, &[m], device))
                    },
                    y_lower,
                    y_upper,
                    &quad_options,
                )?;

                let middle_val: Vec<f64> = middle_result.integral.to_vec();
                results.push(middle_val[0]);
            }

            Ok(Tensor::<R>::from_slice(&results, &[n], device))
        },
        a,
        b,
        &quad_options,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::ops::BinaryOps;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    fn setup() -> (CpuDevice, CpuClient) {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        (device, client)
    }

    #[test]
    fn test_dblquad_rectangle() {
        let (device, client) = setup();

        // Integrate f(x,y) = 1 over [0,1] × [0,1], exact = 1
        let result = dblquad_impl(
            &client,
            |_x, _y| {
                let shape = _y.shape();
                let n = shape[0];
                let ones = vec![1.0; n];
                Ok(Tensor::<CpuRuntime>::from_slice(&ones, &[n], &device))
            },
            0.0,
            1.0,
            |_x| 0.0,
            |_x| 1.0,
            &NQuadOptions::default(),
        )
        .unwrap();

        let integral: Vec<f64> = result.integral.to_vec();
        assert!(
            (integral[0] - 1.0).abs() < 1e-6,
            "integral = {}, expected 1.0",
            integral[0]
        );
    }

    #[test]
    fn test_dblquad_polynomial() {
        let (_device, client) = setup();

        // Integrate f(x,y) = x*y over [0,1] × [0,1], exact = 1/4
        let result = dblquad_impl(
            &client,
            |x, y| client.mul(x, y),
            0.0,
            1.0,
            |_x| 0.0,
            |_x| 1.0,
            &NQuadOptions::default(),
        )
        .unwrap();

        let integral: Vec<f64> = result.integral.to_vec();
        assert!(
            (integral[0] - 0.25).abs() < 1e-6,
            "integral = {}, expected 0.25",
            integral[0]
        );
    }

    #[test]
    fn test_dblquad_triangle() {
        let (device, client) = setup();

        // Integrate f(x,y) = 1 over triangle 0 ≤ x ≤ 1, 0 ≤ y ≤ x
        // Area = 1/2
        let result = dblquad_impl(
            &client,
            |_x, _y| {
                let shape = _y.shape();
                let n = shape[0];
                let ones = vec![1.0; n];
                Ok(Tensor::<CpuRuntime>::from_slice(&ones, &[n], &device))
            },
            0.0,
            1.0,
            |_x| 0.0,
            |x| x, // Upper bound is y = x
            &NQuadOptions::default(),
        )
        .unwrap();

        let integral: Vec<f64> = result.integral.to_vec();
        assert!(
            (integral[0] - 0.5).abs() < 1e-6,
            "integral = {}, expected 0.5",
            integral[0]
        );
    }

    #[test]
    fn test_nquad_1d() {
        let (device, client) = setup();

        // Integrate x^2 from 0 to 1, exact = 1/3
        let result = nquad_impl(
            &client,
            |x| {
                let data: Vec<f64> = x.to_vec();
                let n = data.len();
                let sq: Vec<f64> = data.iter().map(|&xi| xi * xi).collect();
                Ok(Tensor::<CpuRuntime>::from_slice(&sq, &[n], &device))
            },
            &[(0.0, 1.0)],
            &NQuadOptions::default(),
        )
        .unwrap();

        let integral: Vec<f64> = result.integral.to_vec();
        assert!(
            (integral[0] - 1.0 / 3.0).abs() < 1e-6,
            "integral = {}, expected 1/3",
            integral[0]
        );
    }

    #[test]
    fn test_nquad_2d() {
        let (device, client) = setup();

        // Integrate 1 over [0,1]^2, exact = 1
        let result = nquad_impl(
            &client,
            |x| {
                let shape = x.shape();
                let n = shape[0];
                let ones = vec![1.0; n];
                Ok(Tensor::<CpuRuntime>::from_slice(&ones, &[n], &device))
            },
            &[(0.0, 1.0), (0.0, 1.0)],
            &NQuadOptions::default(),
        )
        .unwrap();

        let integral: Vec<f64> = result.integral.to_vec();
        assert!(
            (integral[0] - 1.0).abs() < 0.01,
            "integral = {}, expected 1.0",
            integral[0]
        );
    }
}
