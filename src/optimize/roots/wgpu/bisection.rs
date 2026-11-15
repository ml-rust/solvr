//! WGPU batched bisection delegation.

use crate::optimize::OptimizeResult;
use crate::optimize::roots::impl_generic::bisect_batched_impl;
use crate::optimize::roots::{
    BatchedBisectionAlgorithms, BatchedBisectionOptions, BatchedBracketResult,
};
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl BatchedBisectionAlgorithms<WgpuRuntime> for WgpuClient {
    fn bisect_batched<F>(
        &self,
        f: F,
        lower: &Tensor<WgpuRuntime>,
        upper: &Tensor<WgpuRuntime>,
        options: &BatchedBisectionOptions,
    ) -> OptimizeResult<BatchedBracketResult<WgpuRuntime>>
    where
        F: Fn(&Tensor<WgpuRuntime>) -> numr::error::Result<Tensor<WgpuRuntime>>,
    {
        bisect_batched_impl(self, f, lower, upper, options)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::ops::{BinaryOps, ScalarOps};
    use numr::runtime::wgpu::WgpuDevice;

    #[test]
    fn batched_bisection_matches_signed_scalar_brackets() {
        let device = WgpuDevice::new(0);
        let Ok(client) = WgpuClient::new(device.clone()) else {
            eprintln!("Skipping wgpu batched bisection: no device");
            return;
        };
        let lower = Tensor::<WgpuRuntime>::from_slice(&[0.0f32, -5.0, 1.0], &[3], &device).unwrap();
        let upper = Tensor::<WgpuRuntime>::from_slice(&[1.0f32, -4.0, 2.0], &[3], &device).unwrap();
        let targets =
            Tensor::<WgpuRuntime>::from_slice(&[2.0f32, -3.0, 1.0], &[3], &device).unwrap();
        for sign in [1.0, -1.0] {
            let options = BatchedBisectionOptions {
                max_expansions: 2,
                ..Default::default()
            };
            let result = client
                .bisect_batched(
                    |x| client.mul_scalar(&client.sub(x, &targets)?, sign),
                    &lower,
                    &upper,
                    &options,
                )
                .unwrap();
            assert_eq!(result.expansions, 1);
            assert_eq!(result.iterations, 0);
            assert_eq!(result.lower.to_vec::<f32>(), vec![2.0, -3.0, 1.0]);
            assert_eq!(result.upper.to_vec::<f32>(), vec![2.0, -3.0, 1.0]);
            assert_eq!(result.f_lower.to_vec::<f32>(), vec![0.0, 0.0, 0.0]);
            assert_eq!(result.f_upper.to_vec::<f32>(), vec![0.0, 0.0, 0.0]);
        }
        let upper = Tensor::<WgpuRuntime>::from_slice(&[3.0f32, 0.0, 2.0], &[3], &device).unwrap();
        let result = client
            .bisect_batched(
                |x| client.sub(x, &targets),
                &lower,
                &upper,
                &Default::default(),
            )
            .unwrap();
        let lo = result.lower.to_vec::<f32>();
        let hi = result.upper.to_vec::<f32>();
        for (index, root) in [2.0, -3.0, 1.0].into_iter().enumerate() {
            assert!(lo[index] <= root && root <= hi[index]);
            assert!(hi[index] - lo[index] <= 4e-6);
        }
        assert!(result.iterations > 0);
    }
}
