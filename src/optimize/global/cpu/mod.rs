//! CPU implementations of global optimization algorithms.

mod basinhopping;
mod differential_evolution;
mod dual_annealing;
mod simulated_annealing;

#[cfg(test)]
mod tests {
    use crate::optimize::global::{
        BasinHoppingAlgorithms, DifferentialEvolutionAlgorithms, DualAnnealingAlgorithms,
        GlobalOptions, SimulatedAnnealingAlgorithms,
    };
    use numr::error::Result;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
    use numr::tensor::Tensor;

    fn sphere_tensor(x: &Tensor<CpuRuntime>) -> Result<f64> {
        let data: Vec<f64> = x.to_vec();
        Ok(data.iter().map(|&xi| xi * xi).sum())
    }

    #[test]
    fn test_compare_methods() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());

        let lower = Tensor::<CpuRuntime>::from_slice(&[-5.0, -5.0], &[2], &device);
        let upper = Tensor::<CpuRuntime>::from_slice(&[5.0, 5.0], &[2], &device);

        // DE and DA converge faster
        let opts = GlobalOptions {
            max_iter: 500,
            seed: Some(42),
            ..Default::default()
        };

        let de_result = client
            .differential_evolution(sphere_tensor, &lower, &upper, &opts)
            .expect("DE failed");
        let da_result = client
            .dual_annealing(sphere_tensor, &lower, &upper, &opts)
            .expect("DA failed");

        assert!(de_result.fun < 1e-4);
        assert!(da_result.fun < 1e-4);

        // SA needs more iterations for similar convergence
        let sa_opts = GlobalOptions {
            max_iter: 5000,
            seed: Some(42),
            ..Default::default()
        };

        let sa_result = client
            .simulated_annealing(sphere_tensor, &lower, &upper, &sa_opts)
            .expect("SA failed");
        assert!(sa_result.fun < 1.0); // SA is stochastic, use relaxed tolerance
    }

    #[test]
    fn test_basinhopping() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());

        let x0 = Tensor::<CpuRuntime>::from_slice(&[2.0, 2.0, 2.0], &[3], &device);
        let lower = Tensor::<CpuRuntime>::from_slice(&[-10.0, -10.0, -10.0], &[3], &device);
        let upper = Tensor::<CpuRuntime>::from_slice(&[10.0, 10.0, 10.0], &[3], &device);

        let opts = GlobalOptions {
            max_iter: 50,
            seed: Some(42),
            ..Default::default()
        };

        let result = client
            .basinhopping(sphere_tensor, &x0, &lower, &upper, &opts)
            .expect("basinhopping failed");
        assert!(result.fun < 1e-4);
    }
}
