//! CPU implementation of information theory algorithms.

use crate::stats::impl_generic::{
    cross_entropy_impl, differential_entropy_impl, entropy_impl, kl_divergence_impl,
    mutual_information_impl, nll_loss_impl,
};
use crate::stats::traits::InformationTheoryAlgorithms;
use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl InformationTheoryAlgorithms<CpuRuntime> for CpuClient {
    fn entropy(&self, pk: &Tensor<CpuRuntime>, base: Option<f64>) -> Result<Tensor<CpuRuntime>> {
        entropy_impl(self, pk, base)
    }

    fn differential_entropy(&self, x: &Tensor<CpuRuntime>, k: usize) -> Result<Tensor<CpuRuntime>> {
        differential_entropy_impl(self, x, k)
    }

    fn kl_divergence(
        &self,
        pk: &Tensor<CpuRuntime>,
        qk: &Tensor<CpuRuntime>,
        base: Option<f64>,
    ) -> Result<Tensor<CpuRuntime>> {
        kl_divergence_impl(self, pk, qk, base)
    }

    fn mutual_information(
        &self,
        x: &Tensor<CpuRuntime>,
        y: &Tensor<CpuRuntime>,
        bins: usize,
        base: Option<f64>,
    ) -> Result<Tensor<CpuRuntime>> {
        mutual_information_impl(self, x, y, bins, base)
    }

    fn cross_entropy(
        &self,
        pk: &Tensor<CpuRuntime>,
        qk: &Tensor<CpuRuntime>,
        base: Option<f64>,
    ) -> Result<Tensor<CpuRuntime>> {
        cross_entropy_impl(self, pk, qk, base)
    }

    fn nll_loss(
        &self,
        log_probs: &Tensor<CpuRuntime>,
        targets: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        nll_loss_impl(self, log_probs, targets)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stats::helpers::extract_scalar;
    use numr::runtime::cpu::CpuDevice;

    fn setup() -> (CpuClient, CpuDevice) {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        (client, device)
    }

    #[test]
    fn test_entropy_uniform() {
        let (client, device) = setup();
        // Uniform distribution over 4 outcomes: H = log(4) ≈ 1.386 nats
        let pk =
            Tensor::<CpuRuntime>::from_slice(&[0.25f64, 0.25, 0.25, 0.25], &[4], &device).unwrap();
        let result = client.entropy(&pk, None).unwrap();
        let val = extract_scalar(&result).unwrap();
        assert!((val - 4.0_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_entropy_bits() {
        let (client, device) = setup();
        // Fair coin: H = 1 bit
        let pk = Tensor::<CpuRuntime>::from_slice(&[0.5f64, 0.5], &[2], &device).unwrap();
        let result = client.entropy(&pk, Some(2.0)).unwrap();
        let val = extract_scalar(&result).unwrap();
        assert!((val - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_entropy_certain() {
        let (client, device) = setup();
        // Certain outcome: H = 0
        let pk = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 0.0, 0.0], &[3], &device).unwrap();
        let result = client.entropy(&pk, None).unwrap();
        let val = extract_scalar(&result).unwrap();
        assert!(val.abs() < 1e-10);
    }

    #[test]
    fn test_kl_divergence() {
        let (client, device) = setup();
        let pk = Tensor::<CpuRuntime>::from_slice(&[0.5f64, 0.5], &[2], &device).unwrap();
        let qk = Tensor::<CpuRuntime>::from_slice(&[0.5f64, 0.5], &[2], &device).unwrap();

        // KL(P||P) = 0
        let result = client.kl_divergence(&pk, &qk, None).unwrap();
        let val = extract_scalar(&result).unwrap();
        assert!(val.abs() < 1e-10);
    }

    #[test]
    fn test_kl_divergence_asymmetric() {
        let (client, device) = setup();
        let pk = Tensor::<CpuRuntime>::from_slice(&[0.9f64, 0.1], &[2], &device).unwrap();
        let qk = Tensor::<CpuRuntime>::from_slice(&[0.1f64, 0.9], &[2], &device).unwrap();

        let kl_pq = extract_scalar(&client.kl_divergence(&pk, &qk, None).unwrap()).unwrap();
        let kl_qp = extract_scalar(&client.kl_divergence(&qk, &pk, None).unwrap()).unwrap();

        // KL divergence is non-negative
        assert!(kl_pq > 0.0);
        assert!(kl_qp > 0.0);
        // KL is asymmetric in general (same here by symmetry of swap)
        assert!((kl_pq - kl_qp).abs() < 1e-10);
    }

    #[test]
    fn test_mutual_information() {
        let (client, device) = setup();
        // Perfectly correlated: y = x
        let x = Tensor::<CpuRuntime>::from_slice(
            &[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            &[10],
            &device,
        )
        .unwrap();
        let y = Tensor::<CpuRuntime>::from_slice(
            &[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            &[10],
            &device,
        )
        .unwrap();

        let result = client.mutual_information(&x, &y, 5, None).unwrap();
        let val = extract_scalar(&result).unwrap();
        // MI should be positive for correlated data
        assert!(val > 0.0);
    }

    #[test]
    fn test_cross_entropy_same() {
        let (client, device) = setup();
        // Cross-entropy of P with itself = entropy of P
        let pk =
            Tensor::<CpuRuntime>::from_slice(&[0.25f64, 0.25, 0.25, 0.25], &[4], &device).unwrap();
        let ce = extract_scalar(&client.cross_entropy(&pk, &pk, None).unwrap()).unwrap();
        let h = extract_scalar(&client.entropy(&pk, None).unwrap()).unwrap();
        assert!((ce - h).abs() < 1e-10);
    }

    #[test]
    fn test_cross_entropy_different() {
        let (client, device) = setup();
        let pk = Tensor::<CpuRuntime>::from_slice(&[0.9f64, 0.1], &[2], &device).unwrap();
        let qk = Tensor::<CpuRuntime>::from_slice(&[0.5f64, 0.5], &[2], &device).unwrap();

        let ce = extract_scalar(&client.cross_entropy(&pk, &qk, None).unwrap()).unwrap();
        let h = extract_scalar(&client.entropy(&pk, None).unwrap()).unwrap();
        // Cross-entropy >= entropy (Gibbs inequality)
        assert!(ce >= h - 1e-10);
    }

    #[test]
    fn test_cross_entropy_bits() {
        let (client, device) = setup();
        // H(P, Q) with uniform P and uniform Q in bits should be 1
        let pk = Tensor::<CpuRuntime>::from_slice(&[0.5f64, 0.5], &[2], &device).unwrap();
        let ce = extract_scalar(&client.cross_entropy(&pk, &pk, Some(2.0)).unwrap()).unwrap();
        assert!((ce - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_nll_loss_basic() {
        use numr::ops::ActivationOps;

        let (client, device) = setup();
        // 3 samples, 4 classes
        let logits = Tensor::<CpuRuntime>::from_slice(
            &[
                1.0f64, 2.0, 3.0, 4.0, // sample 0: class 3 has highest
                4.0, 3.0, 2.0, 1.0, // sample 1: class 0 has highest
                1.0, 1.0, 1.0, 1.0, // sample 2: uniform
            ],
            &[3, 4],
            &device,
        )
        .unwrap();
        let log_probs = client.log_softmax(&logits, -1).unwrap();
        let targets = Tensor::<CpuRuntime>::from_slice(&[3.0f64, 0.0, 2.0], &[3], &device).unwrap();

        let loss = extract_scalar(&client.nll_loss(&log_probs, &targets).unwrap()).unwrap();
        // Loss should be positive (negative of negative log prob)
        assert!(loss > 0.0);
        // When targets match the highest logit, loss should be relatively small
        assert!(loss < 2.0);
    }

    #[test]
    fn test_nll_loss_perfect() {
        use numr::ops::ActivationOps;

        let (client, device) = setup();
        // Perfect predictions: very high logit at target class
        let logits = Tensor::<CpuRuntime>::from_slice(
            &[
                -100.0f64, -100.0, 100.0, -100.0, // sample 0: class 2
                100.0, -100.0, -100.0, -100.0, // sample 1: class 0
            ],
            &[2, 4],
            &device,
        )
        .unwrap();
        let log_probs = client.log_softmax(&logits, -1).unwrap();
        let targets = Tensor::<CpuRuntime>::from_slice(&[2.0f64, 0.0], &[2], &device).unwrap();

        let loss = extract_scalar(&client.nll_loss(&log_probs, &targets).unwrap()).unwrap();
        // Near-perfect predictions → loss ≈ 0
        assert!(loss.abs() < 1e-5);
    }

    #[test]
    fn test_differential_entropy() {
        let (client, device) = setup();
        // Generate uniform-ish data
        let data: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();
        let x = Tensor::<CpuRuntime>::from_slice(&data, &[100], &device).unwrap();

        let result = client.differential_entropy(&x, 3).unwrap();
        let val = extract_scalar(&result).unwrap();
        // Differential entropy of Uniform(0,1) = 0 nats (ln(1) = 0)
        // With finite samples, should be close to 0
        assert!(val.abs() < 1.0); // Loose bound
    }
}
