//! CPU implementation of Levenberg-Marquardt algorithm for root finding.

use crate::optimize::roots::impl_generic::levenberg_marquardt_impl;
use crate::optimize::roots::traits::LevenbergMarquardtAlgorithms;
use crate::optimize::roots::{RootOptions, RootTensorResult};
use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl LevenbergMarquardtAlgorithms<CpuRuntime> for CpuClient {
    fn levenberg_marquardt<F>(
        &self,
        f: F,
        x0: &Tensor<CpuRuntime>,
        options: &RootOptions,
    ) -> Result<RootTensorResult<CpuRuntime>>
    where
        F: Fn(&Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>>,
    {
        let result = levenberg_marquardt_impl(self, f, x0, options).map_err(|e| {
            numr::error::Error::backend_limitation("cpu", "levenberg_marquardt", e.to_string())
        })?;
        Ok(RootTensorResult {
            x: result.x,
            fun: result.fun,
            iterations: result.iterations,
            residual_norm: result.residual_norm,
            converged: result.converged,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::CpuDevice;

    fn linear_system(x: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        let data: Vec<f64> = x.to_vec();
        let residuals = vec![data[0] + data[1] - 3.0, 2.0 * data[0] - data[1]];
        Tensor::from_slice(&residuals, &[2], x.device())
    }

    #[test]
    fn test_levenberg_marquardt_linear() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());

        let x0 = Tensor::<CpuRuntime>::from_slice(&[0.0, 0.0], &[2], &device).unwrap();
        let result = client
            .levenberg_marquardt(linear_system, &x0, &RootOptions::default())
            .expect("levenberg_marquardt failed");

        assert!(result.converged);
        let x_data: Vec<f64> = result.x.to_vec();
        assert!((x_data[0] - 1.0).abs() < 1e-5);
        assert!((x_data[1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_compare_methods() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());

        let circle_system = |x: &Tensor<CpuRuntime>| -> Result<Tensor<CpuRuntime>> {
            let data: Vec<f64> = x.to_vec();
            let residuals = vec![
                data[0] * data[0] + data[1] * data[1] - 4.0,
                data[0] - data[1],
            ];
            Tensor::from_slice(&residuals, &[2], x.device())
        };

        let x0 = Tensor::<CpuRuntime>::from_slice(&[1.0, 1.0], &[2], &device).unwrap();
        let expected = (2.0_f64).sqrt();

        use crate::optimize::roots::traits::{Broyden1Algorithms, NewtonSystemAlgorithms};

        let newton_result = client
            .newton_system(circle_system, &x0, &RootOptions::default())
            .expect("newton failed");
        let broyden_result = client
            .broyden1(circle_system, &x0, &RootOptions::default())
            .expect("broyden failed");
        let lm_result = client
            .levenberg_marquardt(circle_system, &x0, &RootOptions::default())
            .expect("lm failed");

        let newton_x: Vec<f64> = newton_result.x.to_vec();
        let broyden_x: Vec<f64> = broyden_result.x.to_vec();
        let lm_x: Vec<f64> = lm_result.x.to_vec();

        assert!(newton_result.converged);
        assert!((newton_x[0] - expected).abs() < 1e-5);

        assert!(broyden_result.converged);
        assert!((broyden_x[0] - expected).abs() < 1e-5);

        assert!(lm_result.converged);
        assert!((lm_x[0] - expected).abs() < 1e-4);
    }
}
