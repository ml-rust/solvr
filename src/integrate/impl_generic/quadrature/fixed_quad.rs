//! Fixed-order Gaussian quadrature using tensor operations.
//!
//! The Gauss-Legendre node computation is inherently scalar (one-time setup),
//! but the function evaluation and weighted sum use tensor ops for GPU acceleration.

use numr::error::{Error, Result};
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Fixed-order Gaussian quadrature.
///
/// Integrates a function from a to b using n-point Gauss-Legendre quadrature.
/// All n evaluation points are computed in a single batch using tensor operations.
pub fn fixed_quad_impl<R, C, F>(client: &C, f: F, a: f64, b: f64, n: usize) -> Result<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
    F: Fn(&Tensor<R>) -> Result<Tensor<R>>,
{
    if n == 0 {
        return Err(Error::InvalidArgument {
            arg: "n",
            reason: "fixed_quad: n must be at least 1".to_string(),
        });
    }

    // Get Gauss-Legendre nodes and weights for [-1, 1]
    // This is a one-time scalar computation (acceptable)
    let (nodes, weights) = gauss_legendre_nodes_weights(n);

    // Transform nodes from [-1, 1] to [a, b]
    let half_width = (b - a) / 2.0;
    let center = (a + b) / 2.0;

    let transformed_nodes: Vec<f64> = nodes.iter().map(|&x| center + half_width * x).collect();

    // Evaluate function at all nodes in a single batch
    let x_tensor = Tensor::<R>::from_slice(&transformed_nodes, &[n], client.device());
    let f_values = f(&x_tensor)?;

    // Create weight tensor and compute weighted sum using tensor ops
    let weight_tensor = Tensor::<R>::from_slice(&weights, &[n], client.device());

    // weighted = f_values * weights
    let weighted = client.mul(&f_values, &weight_tensor)?;

    // integral = sum(weighted) * half_width
    let sum = client.sum(&weighted, &[0], false)?;

    client.mul_scalar(&sum, half_width)
}

/// Compute Gauss-Legendre nodes and weights.
///
/// Uses Newton iteration to find roots of Legendre polynomials.
/// This is an inherently scalar one-time computation that produces
/// the quadrature points. The actual integration uses tensor ops.
fn gauss_legendre_nodes_weights(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut nodes = vec![0.0; n];
    let mut weights = vec![0.0; n];

    let m = n.div_ceil(2);

    for i in 0..m {
        // Initial guess using Chebyshev approximation
        let mut z = ((i as f64 + 0.75) / (n as f64 + 0.5) * std::f64::consts::PI).cos();

        // Newton iteration to find root of Legendre polynomial
        loop {
            let (p, dp) = legendre_p_and_dp(n, z);
            let z_new = z - p / dp;

            if (z_new - z).abs() < 1e-15 {
                z = z_new;
                break;
            }
            z = z_new;
        }

        let (_, dp) = legendre_p_and_dp(n, z);
        let w = 2.0 / ((1.0 - z * z) * dp * dp);

        nodes[i] = -z;
        nodes[n - 1 - i] = z;
        weights[i] = w;
        weights[n - 1 - i] = w;
    }

    (nodes, weights)
}

/// Evaluate Legendre polynomial P_n(x) and its derivative.
fn legendre_p_and_dp(n: usize, x: f64) -> (f64, f64) {
    if n == 0 {
        return (1.0, 0.0);
    }
    if n == 1 {
        return (x, 1.0);
    }

    let mut p_prev = 1.0;
    let mut p_curr = x;

    for k in 2..=n {
        let p_next = ((2 * k - 1) as f64 * x * p_curr - (k - 1) as f64 * p_prev) / k as f64;
        p_prev = p_curr;
        p_curr = p_next;
    }

    // Derivative: P'_n(x) = n * (x * P_n - P_{n-1}) / (x^2 - 1)
    let dp = n as f64 * (x * p_curr - p_prev) / (x * x - 1.0);

    (p_curr, dp)
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::{CpuClient, CpuDevice};

    fn get_client() -> CpuClient {
        let device = CpuDevice::new();
        CpuClient::new(device)
    }

    #[test]
    fn test_fixed_quad_constant() {
        let client = get_client();

        // Integrate f(x) = 1 from 0 to 1
        // Exact: 1.0
        let result = fixed_quad_impl(
            &client,
            |x| {
                Ok(Tensor::from_slice(
                    &vec![1.0; x.numel()],
                    x.shape(),
                    x.device(),
                ))
            },
            0.0,
            1.0,
            5,
        )
        .unwrap();

        let values: Vec<f64> = result.to_vec();
        assert!((values[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_fixed_quad_linear() {
        let client = get_client();

        // Integrate f(x) = x from 0 to 1
        // Exact: 0.5
        let result = fixed_quad_impl(&client, |x| Ok(x.clone()), 0.0, 1.0, 5).unwrap();

        let values: Vec<f64> = result.to_vec();
        assert!((values[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_fixed_quad_quadratic() {
        let client = get_client();

        // Integrate f(x) = x^2 from 0 to 1
        // Exact: 1/3
        let result = fixed_quad_impl(
            &client,
            |x| {
                let client = get_client();
                client.mul(x, x)
            },
            0.0,
            1.0,
            5,
        )
        .unwrap();

        let values: Vec<f64> = result.to_vec();
        assert!((values[0] - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_fixed_quad_polynomial() {
        let client = get_client();

        // Integrate f(x) = x^4 from 0 to 2
        // Exact: 32/5 = 6.4
        // Gauss-Legendre with n=3 is exact for polynomials up to degree 2n-1=5
        let result = fixed_quad_impl(
            &client,
            |x| {
                let client = get_client();
                let x2 = client.mul(x, x)?;
                client.mul(&x2, &x2)
            },
            0.0,
            2.0,
            3,
        )
        .unwrap();

        let values: Vec<f64> = result.to_vec();
        assert!((values[0] - 6.4).abs() < 1e-10);
    }
}
