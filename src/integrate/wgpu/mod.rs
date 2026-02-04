use numr::autograd::DualTensor;
use numr::error::Result;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

use crate::integrate::error::IntegrateResult;
use crate::integrate::impl_generic::ode::{
    bdf_impl, bvp_impl, leapfrog_impl, lsoda_impl, radau_impl, solve_ivp_impl, verlet_impl,
};
use crate::integrate::impl_generic::quadrature::{
    cumulative_trapezoid_impl, dblquad_impl, fixed_quad_impl, monte_carlo_impl, nquad_impl,
    qmc_impl, quad_impl, romberg_impl, simpson_impl, tanh_sinh_impl, trapezoid_impl,
    trapezoid_uniform_impl,
};
use crate::integrate::ode::{
    BDFOptions, BVPOptions, LSODAOptions, RadauOptions, SymplecticOptions,
};
use crate::integrate::{
    BVPResult, IntegrationAlgorithms, MonteCarloOptions, MonteCarloResult, NQuadOptions,
    ODEOptions, ODEResultTensor, QMCOptions, QuadOptions, QuadResult, RombergOptions,
    SymplecticResult, TanhSinhOptions,
};

mod fixed_quad;
mod quad;
mod romberg;
mod simpson;
mod solve_ivp;
mod trapezoid;

impl IntegrationAlgorithms<WgpuRuntime> for WgpuClient {
    fn trapezoid(
        &self,
        y: &Tensor<WgpuRuntime>,
        x: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        trapezoid_impl(self, y, x)
    }

    fn trapezoid_uniform(&self, y: &Tensor<WgpuRuntime>, dx: f64) -> Result<Tensor<WgpuRuntime>> {
        trapezoid_uniform_impl(self, y, dx)
    }

    fn cumulative_trapezoid(
        &self,
        y: &Tensor<WgpuRuntime>,
        x: Option<&Tensor<WgpuRuntime>>,
        dx: f64,
    ) -> Result<Tensor<WgpuRuntime>> {
        cumulative_trapezoid_impl(self, y, x, dx)
    }

    fn simpson(
        &self,
        y: &Tensor<WgpuRuntime>,
        x: Option<&Tensor<WgpuRuntime>>,
        dx: f64,
    ) -> Result<Tensor<WgpuRuntime>> {
        simpson_impl(self, y, x, dx)
    }

    fn fixed_quad<F>(&self, f: F, a: f64, b: f64, n: usize) -> Result<Tensor<WgpuRuntime>>
    where
        F: Fn(&Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>>,
    {
        fixed_quad_impl(self, f, a, b, n)
    }

    fn quad<F>(
        &self,
        f: F,
        a: f64,
        b: f64,
        options: &QuadOptions,
    ) -> Result<QuadResult<WgpuRuntime>>
    where
        F: Fn(&Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>>,
    {
        quad_impl(self, f, a, b, options)
    }

    fn romberg<F>(
        &self,
        f: F,
        a: f64,
        b: f64,
        options: &RombergOptions,
    ) -> Result<QuadResult<WgpuRuntime>>
    where
        F: Fn(&Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>>,
    {
        romberg_impl(self, f, a, b, options)
    }

    fn solve_ivp<F>(
        &self,
        f: F,
        t_span: [f64; 2],
        y0: &Tensor<WgpuRuntime>,
        options: &ODEOptions,
    ) -> IntegrateResult<ODEResultTensor<WgpuRuntime>>
    where
        F: Fn(&Tensor<WgpuRuntime>, &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>>,
    {
        solve_ivp_impl(self, f, t_span, y0, options)
    }

    fn tanh_sinh<F>(
        &self,
        f: F,
        a: f64,
        b: f64,
        options: &TanhSinhOptions,
    ) -> Result<QuadResult<WgpuRuntime>>
    where
        F: Fn(&Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>>,
    {
        tanh_sinh_impl(self, f, a, b, options)
    }

    fn monte_carlo<F>(
        &self,
        f: F,
        bounds: &[(f64, f64)],
        options: &MonteCarloOptions,
    ) -> Result<MonteCarloResult<WgpuRuntime>>
    where
        F: Fn(&Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>>,
    {
        monte_carlo_impl(self, f, bounds, options)
    }

    fn qmc_quad<F>(
        &self,
        f: F,
        bounds: &[(f64, f64)],
        options: &QMCOptions,
    ) -> Result<QuadResult<WgpuRuntime>>
    where
        F: Fn(&Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>>,
    {
        qmc_impl(self, f, bounds, options)
    }

    fn dblquad<F, G, H>(
        &self,
        f: F,
        a: f64,
        b: f64,
        gfun: G,
        hfun: H,
        options: &NQuadOptions,
    ) -> Result<QuadResult<WgpuRuntime>>
    where
        F: Fn(&Tensor<WgpuRuntime>, &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>>,
        G: Fn(f64) -> f64,
        H: Fn(f64) -> f64,
    {
        dblquad_impl(self, f, a, b, gfun, hfun, options)
    }

    fn nquad<F>(
        &self,
        f: F,
        bounds: &[(f64, f64)],
        options: &NQuadOptions,
    ) -> Result<QuadResult<WgpuRuntime>>
    where
        F: Fn(&Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>>,
    {
        nquad_impl(self, f, bounds, options)
    }

    fn solve_ivp_bdf<F>(
        &self,
        f: F,
        t_span: [f64; 2],
        y0: &Tensor<WgpuRuntime>,
        options: &ODEOptions,
        bdf_options: &BDFOptions,
    ) -> IntegrateResult<ODEResultTensor<WgpuRuntime>>
    where
        F: Fn(
            &DualTensor<WgpuRuntime>,
            &DualTensor<WgpuRuntime>,
            &Self,
        ) -> Result<DualTensor<WgpuRuntime>>,
    {
        bdf_impl(self, f, t_span, y0, options, bdf_options)
    }

    fn solve_ivp_radau<F>(
        &self,
        f: F,
        t_span: [f64; 2],
        y0: &Tensor<WgpuRuntime>,
        options: &ODEOptions,
        radau_options: &RadauOptions,
    ) -> IntegrateResult<ODEResultTensor<WgpuRuntime>>
    where
        F: Fn(
            &DualTensor<WgpuRuntime>,
            &DualTensor<WgpuRuntime>,
            &Self,
        ) -> Result<DualTensor<WgpuRuntime>>,
    {
        radau_impl(self, f, t_span, y0, options, radau_options)
    }

    fn solve_ivp_lsoda<F>(
        &self,
        f: F,
        t_span: [f64; 2],
        y0: &Tensor<WgpuRuntime>,
        options: &ODEOptions,
        lsoda_options: &LSODAOptions,
    ) -> IntegrateResult<ODEResultTensor<WgpuRuntime>>
    where
        F: Fn(
            &DualTensor<WgpuRuntime>,
            &DualTensor<WgpuRuntime>,
            &Self,
        ) -> Result<DualTensor<WgpuRuntime>>,
    {
        lsoda_impl(self, f, t_span, y0, options, lsoda_options)
    }

    fn solve_bvp<F, BC>(
        &self,
        f: F,
        bc: BC,
        x: &Tensor<WgpuRuntime>,
        y: &Tensor<WgpuRuntime>,
        options: &BVPOptions,
    ) -> IntegrateResult<BVPResult<WgpuRuntime>>
    where
        F: Fn(&Tensor<WgpuRuntime>, &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>>,
        BC: Fn(&Tensor<WgpuRuntime>, &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>>,
    {
        bvp_impl(self, f, bc, x, y, options)
    }

    fn verlet<F>(
        &self,
        force: F,
        t_span: [f64; 2],
        q0: &Tensor<WgpuRuntime>,
        p0: &Tensor<WgpuRuntime>,
        options: &SymplecticOptions,
    ) -> IntegrateResult<SymplecticResult<WgpuRuntime>>
    where
        F: Fn(&Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>>,
    {
        verlet_impl(self, force, t_span, q0, p0, options)
    }

    fn leapfrog<F>(
        &self,
        force: F,
        t_span: [f64; 2],
        q0: &Tensor<WgpuRuntime>,
        p0: &Tensor<WgpuRuntime>,
        options: &SymplecticOptions,
    ) -> IntegrateResult<SymplecticResult<WgpuRuntime>>
    where
        F: Fn(&Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>>,
    {
        leapfrog_impl(self, force, t_span, q0, p0, options)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::wgpu::WgpuDevice;

    fn setup() -> Option<(WgpuDevice, WgpuClient)> {
        let device = WgpuDevice::new(0);
        let client = WgpuClient::new(device.clone()).ok()?;
        Some((device, client))
    }

    #[test]
    fn test_trapezoid_wgpu() {
        let Some((device, client)) = setup() else {
            eprintln!("Skipping WebGPU test: no device");
            return;
        };

        let n = 101;
        let x_data: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
        let y_data: Vec<f64> = x_data.iter().map(|&xi| xi * xi).collect();

        let x = Tensor::<WgpuRuntime>::from_slice(&x_data, &[n], &device);
        let y = Tensor::<WgpuRuntime>::from_slice(&y_data, &[n], &device);

        let result = client.trapezoid(&y, &x).unwrap();
        let result_val: Vec<f64> = result.to_vec();

        assert!((result_val[0] - 1.0 / 3.0).abs() < 0.001);
    }

    #[test]
    fn test_fixed_quad_wgpu() {
        let Some((device, client)) = setup() else {
            eprintln!("Skipping WebGPU test: no device");
            return;
        };

        let result = client
            .fixed_quad(
                |x| {
                    let data: Vec<f64> = x.to_vec();
                    let sin_data: Vec<f64> = data.iter().map(|&xi| xi.sin()).collect();
                    Ok(Tensor::<WgpuRuntime>::from_slice(
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
        assert!((result_val[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_quad_wgpu() {
        let Some((device, client)) = setup() else {
            eprintln!("Skipping WebGPU test: no device");
            return;
        };

        let result = client
            .quad(
                |x| {
                    let data: Vec<f64> = x.to_vec();
                    let sin_data: Vec<f64> = data.iter().map(|&xi| xi.sin()).collect();
                    Ok(Tensor::<WgpuRuntime>::from_slice(
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
        assert!((result_val[0] - 2.0).abs() < 1e-8);
        assert!(result.converged);
    }

    #[test]
    fn test_romberg_wgpu() {
        let Some((device, client)) = setup() else {
            eprintln!("Skipping WebGPU test: no device");
            return;
        };

        let result = client
            .romberg(
                |x| {
                    let data: Vec<f64> = x.to_vec();
                    let exp_data: Vec<f64> = data.iter().map(|&xi| xi.exp()).collect();
                    Ok(Tensor::<WgpuRuntime>::from_slice(
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
        let exact = std::f64::consts::E - 1.0;
        assert!((result_val[0] - exact).abs() < 1e-8);
        assert!(result.converged);
    }
}
