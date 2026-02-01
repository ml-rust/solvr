use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

use crate::integrate::error::IntegrateResult;
use crate::integrate::impl_generic::ode::solve_ivp_impl;
use crate::integrate::impl_generic::quadrature::{
    cumulative_trapezoid_impl, fixed_quad_impl, quad_impl, romberg_impl, simpson_impl,
    trapezoid_impl, trapezoid_uniform_impl,
};
use crate::integrate::{
    IntegrationAlgorithms, ODEOptions, ODEResultTensor, QuadOptions, QuadResult, RombergOptions,
};

mod fixed_quad;
mod quad;
mod romberg;
mod simpson;
mod solve_ivp;
mod trapezoid;

impl IntegrationAlgorithms<CpuRuntime> for CpuClient {
    fn trapezoid(
        &self,
        y: &Tensor<CpuRuntime>,
        x: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        trapezoid_impl(self, y, x)
    }

    fn trapezoid_uniform(&self, y: &Tensor<CpuRuntime>, dx: f64) -> Result<Tensor<CpuRuntime>> {
        trapezoid_uniform_impl(self, y, dx)
    }

    fn cumulative_trapezoid(
        &self,
        y: &Tensor<CpuRuntime>,
        x: Option<&Tensor<CpuRuntime>>,
        dx: f64,
    ) -> Result<Tensor<CpuRuntime>> {
        cumulative_trapezoid_impl(self, y, x, dx)
    }

    fn simpson(
        &self,
        y: &Tensor<CpuRuntime>,
        x: Option<&Tensor<CpuRuntime>>,
        dx: f64,
    ) -> Result<Tensor<CpuRuntime>> {
        simpson_impl(self, y, x, dx)
    }

    fn fixed_quad<F>(&self, f: F, a: f64, b: f64, n: usize) -> Result<Tensor<CpuRuntime>>
    where
        F: Fn(&Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>>,
    {
        fixed_quad_impl(self, f, a, b, n)
    }

    fn quad<F>(&self, f: F, a: f64, b: f64, options: &QuadOptions) -> Result<QuadResult<CpuRuntime>>
    where
        F: Fn(&Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>>,
    {
        quad_impl(self, f, a, b, options)
    }

    fn romberg<F>(
        &self,
        f: F,
        a: f64,
        b: f64,
        options: &RombergOptions,
    ) -> Result<QuadResult<CpuRuntime>>
    where
        F: Fn(&Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>>,
    {
        romberg_impl(self, f, a, b, options)
    }

    fn solve_ivp<F>(
        &self,
        f: F,
        t_span: [f64; 2],
        y0: &Tensor<CpuRuntime>,
        options: &ODEOptions,
    ) -> IntegrateResult<ODEResultTensor<CpuRuntime>>
    where
        F: Fn(&Tensor<CpuRuntime>, &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>>,
    {
        solve_ivp_impl(self, f, t_span, y0, options)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::ops::ScalarOps;
    use numr::runtime::cpu::CpuDevice;

    fn setup() -> (CpuDevice, CpuClient) {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        (device, client)
    }

    #[test]
    fn test_trapezoid_cpu() {
        let (device, client) = setup();

        // Integrate y = x^2 from 0 to 1
        let n = 101;
        let x_data: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
        let y_data: Vec<f64> = x_data.iter().map(|&xi| xi * xi).collect();

        let x = Tensor::<CpuRuntime>::from_slice(&x_data, &[n], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&y_data, &[n], &device);

        let result = client.trapezoid(&y, &x).unwrap();
        let result_val: Vec<f64> = result.to_vec();

        // Exact value is 1/3
        assert!((result_val[0] - 1.0 / 3.0).abs() < 0.001);
    }

    #[test]
    fn test_simpson_cpu() {
        let (device, client) = setup();

        // Integrate y = x^2 from 0 to 1 (using uniform spacing)
        let n = 101;
        let dx = 1.0 / (n - 1) as f64;
        let y_data: Vec<f64> = (0..n)
            .map(|i| {
                let x = i as f64 * dx;
                x * x
            })
            .collect();

        let y = Tensor::<CpuRuntime>::from_slice(&y_data, &[n], &device);

        let result = client.simpson(&y, None, dx).unwrap();
        let result_val: Vec<f64> = result.to_vec();

        // Exact value is 1/3
        assert!((result_val[0] - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_fixed_quad_cpu() {
        let (device, client) = setup();

        // Integrate sin(x) from 0 to pi
        let result = client
            .fixed_quad(
                |x| {
                    let data: Vec<f64> = x.to_vec();
                    let sin_data: Vec<f64> = data.iter().map(|&xi| xi.sin()).collect();
                    Ok(Tensor::<CpuRuntime>::from_slice(
                        &sin_data,
                        x.shape(),
                        &device,
                    ))
                },
                0.0,
                std::f64::consts::PI,
                10,
            )
            .unwrap();

        let result_val: Vec<f64> = result.to_vec();

        // Exact value is 2.0
        assert!((result_val[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_quad_cpu() {
        let (device, client) = setup();

        // Integrate sin(x) from 0 to pi using adaptive quadrature
        let result = client
            .quad(
                |x| {
                    let data: Vec<f64> = x.to_vec();
                    let sin_data: Vec<f64> = data.iter().map(|&xi| xi.sin()).collect();
                    Ok(Tensor::<CpuRuntime>::from_slice(
                        &sin_data,
                        x.shape(),
                        &device,
                    ))
                },
                0.0,
                std::f64::consts::PI,
                &QuadOptions::default(),
            )
            .unwrap();

        let result_val: Vec<f64> = result.integral.to_vec();

        // Exact value is 2.0
        assert!((result_val[0] - 2.0).abs() < 1e-8);
        assert!(result.converged);
    }

    #[test]
    fn test_romberg_cpu() {
        let (device, client) = setup();

        // Integrate exp(x) from 0 to 1
        let result = client
            .romberg(
                |x| {
                    let data: Vec<f64> = x.to_vec();
                    let exp_data: Vec<f64> = data.iter().map(|&xi| xi.exp()).collect();
                    Ok(Tensor::<CpuRuntime>::from_slice(
                        &exp_data,
                        x.shape(),
                        &device,
                    ))
                },
                0.0,
                1.0,
                &RombergOptions::default(),
            )
            .unwrap();

        let result_val: Vec<f64> = result.integral.to_vec();

        // Exact value is e - 1
        let exact = std::f64::consts::E - 1.0;
        assert!((result_val[0] - exact).abs() < 1e-8);
        assert!(result.converged);
    }

    #[test]
    fn test_batch_trapezoid_cpu() {
        let (device, client) = setup();

        // Batch integration: integrate y = x and y = x^2 simultaneously
        let n = 101;
        let x_data: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();

        // First row: y = x (integral = 0.5)
        // Second row: y = x^2 (integral = 1/3)
        let mut y_data = Vec::with_capacity(2 * n);
        for &xi in &x_data {
            y_data.push(xi);
        }
        for &xi in &x_data {
            y_data.push(xi * xi);
        }

        let x = Tensor::<CpuRuntime>::from_slice(&x_data, &[n], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&y_data, &[2, n], &device);

        let result = client.trapezoid(&y, &x).unwrap();
        let result_val: Vec<f64> = result.to_vec();

        assert!((result_val[0] - 0.5).abs() < 0.001);
        assert!((result_val[1] - 1.0 / 3.0).abs() < 0.001);
    }

    #[test]
    fn test_solve_ivp_cpu() {
        let (device, client) = setup();

        // Solve dy/dt = -y, y(0) = 1
        // Solution: y(t) = exp(-t)
        let y0 = Tensor::<CpuRuntime>::from_slice(&[1.0], &[1], &device);

        let result = client
            .solve_ivp(
                |_t, y| client.mul_scalar(y, -1.0), // t is tensor [1], y is tensor [n]
                [0.0, 5.0],
                &y0,
                &crate::integrate::ODEOptions::default(),
            )
            .unwrap();

        assert!(result.success);
        assert_eq!(result.method, crate::integrate::ODEMethod::RK45);

        let y_final = result.y_final_vec();
        let exact = (-5.0_f64).exp();

        assert!(
            (y_final[0] - exact).abs() < 1e-4,
            "y_final = {}, exact = {}",
            y_final[0],
            exact
        );
    }

    #[test]
    fn test_solve_ivp_harmonic_oscillator_cpu() {
        let (device, client) = setup();

        // y'' + y = 0 as system: y1' = y2, y2' = -y1
        // y1(0) = 1, y2(0) = 0
        // Solution: y1 = cos(t), y2 = -sin(t)
        let y0 = Tensor::<CpuRuntime>::from_slice(&[1.0, 0.0], &[2], &device);

        let opts = crate::integrate::ODEOptions::with_tolerances(1e-6, 1e-8);

        let result = client
            .solve_ivp(
                |_t, y| {
                    // For harmonic oscillator: dy1/dt = y2, dy2/dt = -y1
                    // Transfer to host for indexing (this test is about the solver, not tensor ops)
                    let y_data: Vec<f64> = y.to_vec();
                    Ok(Tensor::<CpuRuntime>::from_slice(
                        &[y_data[1], -y_data[0]],
                        &[2],
                        &device,
                    ))
                },
                [0.0, 2.0 * std::f64::consts::PI],
                &y0,
                &opts,
            )
            .unwrap();

        assert!(result.success);

        // After one full period, should return to initial state
        let y_final = result.y_final_vec();
        assert!((y_final[0] - 1.0).abs() < 0.01, "y1 = {}", y_final[0]);
        assert!(y_final[1].abs() < 0.01, "y2 = {}", y_final[1]);
    }
}
