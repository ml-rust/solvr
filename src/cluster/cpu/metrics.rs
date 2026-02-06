//! CPU implementation of cluster evaluation metrics.

use crate::cluster::impl_generic::{
    adjusted_rand_score_impl, calinski_harabasz_score_impl, davies_bouldin_score_impl,
    homogeneity_completeness_v_measure_impl, normalized_mutual_info_score_impl,
    silhouette_samples_impl, silhouette_score_impl,
};
use crate::cluster::traits::metrics::{ClusterMetricsAlgorithms, HCVScore};
use numr::error::Result;
use numr::ops::DistanceMetric;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl ClusterMetricsAlgorithms<CpuRuntime> for CpuClient {
    fn silhouette_score(
        &self,
        data: &Tensor<CpuRuntime>,
        labels: &Tensor<CpuRuntime>,
        metric: DistanceMetric,
    ) -> Result<Tensor<CpuRuntime>> {
        silhouette_score_impl(self, data, labels, metric)
    }

    fn silhouette_samples(
        &self,
        data: &Tensor<CpuRuntime>,
        labels: &Tensor<CpuRuntime>,
        metric: DistanceMetric,
    ) -> Result<Tensor<CpuRuntime>> {
        silhouette_samples_impl(self, data, labels, metric)
    }

    fn calinski_harabasz_score(
        &self,
        data: &Tensor<CpuRuntime>,
        labels: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        calinski_harabasz_score_impl(self, data, labels)
    }

    fn davies_bouldin_score(
        &self,
        data: &Tensor<CpuRuntime>,
        labels: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        davies_bouldin_score_impl(self, data, labels)
    }

    fn adjusted_rand_score(
        &self,
        labels_true: &Tensor<CpuRuntime>,
        labels_pred: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        adjusted_rand_score_impl(self, labels_true, labels_pred)
    }

    fn normalized_mutual_info_score(
        &self,
        labels_true: &Tensor<CpuRuntime>,
        labels_pred: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        normalized_mutual_info_score_impl(self, labels_true, labels_pred)
    }

    fn homogeneity_completeness_v_measure(
        &self,
        labels_true: &Tensor<CpuRuntime>,
        labels_pred: &Tensor<CpuRuntime>,
    ) -> Result<HCVScore<CpuRuntime>> {
        homogeneity_completeness_v_measure_impl(self, labels_true, labels_pred)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::CpuDevice;

    fn setup() -> (CpuClient, CpuDevice) {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        (client, device)
    }

    #[test]
    fn test_silhouette_score() {
        let (client, device) = setup();

        #[rustfmt::skip]
        let data = Tensor::<CpuRuntime>::from_slice(
            &[
                0.0, 0.0,
                0.1, 0.1,
                0.2, 0.0,
                10.0, 10.0,
                10.1, 10.1,
                10.2, 10.0,
            ],
            &[6, 2],
            &device,
        );

        let labels = Tensor::<CpuRuntime>::from_slice(&[0i64, 0, 0, 1, 1, 1], &[6], &device);

        let score = client
            .silhouette_score(&data, &labels, DistanceMetric::Euclidean)
            .unwrap();
        let val: f64 = score.item().unwrap();
        // Well-separated clusters should have high silhouette score
        assert!(val > 0.5, "Expected silhouette > 0.5, got {}", val);
    }

    #[test]
    fn test_adjusted_rand_score_perfect() {
        let (client, device) = setup();

        let labels_true = Tensor::<CpuRuntime>::from_slice(&[0i64, 0, 0, 1, 1, 1], &[6], &device);
        let labels_pred = Tensor::<CpuRuntime>::from_slice(&[1i64, 1, 1, 0, 0, 0], &[6], &device);

        let ari = client
            .adjusted_rand_score(&labels_true, &labels_pred)
            .unwrap();
        let val: f64 = ari.item().unwrap();
        // Perfect agreement (up to permutation) should give ARI = 1.0
        assert!((val - 1.0).abs() < 1e-6, "Expected ARI = 1.0, got {}", val);
    }

    #[test]
    fn test_calinski_harabasz_score() {
        let (client, device) = setup();

        #[rustfmt::skip]
        let data = Tensor::<CpuRuntime>::from_slice(
            &[
                0.0, 0.0,
                0.1, 0.1,
                0.2, 0.0,
                10.0, 10.0,
                10.1, 10.1,
                10.2, 10.0,
            ],
            &[6, 2],
            &device,
        );

        let labels = Tensor::<CpuRuntime>::from_slice(&[0i64, 0, 0, 1, 1, 1], &[6], &device);

        let ch = client.calinski_harabasz_score(&data, &labels).unwrap();
        let val: f64 = ch.item().unwrap();
        // Well-separated clusters should have a high CH index
        assert!(val > 1.0, "Expected CH > 1.0, got {}", val);
    }

    #[test]
    fn test_davies_bouldin_score() {
        let (client, device) = setup();

        #[rustfmt::skip]
        let data = Tensor::<CpuRuntime>::from_slice(
            &[
                0.0, 0.0,
                0.1, 0.1,
                0.2, 0.0,
                10.0, 10.0,
                10.1, 10.1,
                10.2, 10.0,
            ],
            &[6, 2],
            &device,
        );

        let labels = Tensor::<CpuRuntime>::from_slice(&[0i64, 0, 0, 1, 1, 1], &[6], &device);

        let db = client.davies_bouldin_score(&data, &labels).unwrap();
        let val: f64 = db.item().unwrap();
        // Lower is better; well-separated clusters should be small
        assert!(val < 1.0, "Expected DB < 1.0, got {}", val);
    }

    #[test]
    fn test_nmi_and_hcv() {
        let (client, device) = setup();

        let labels_true = Tensor::<CpuRuntime>::from_slice(&[0i64, 0, 0, 1, 1, 1], &[6], &device);
        let labels_pred = Tensor::<CpuRuntime>::from_slice(&[0i64, 0, 0, 1, 1, 1], &[6], &device);

        let nmi = client
            .normalized_mutual_info_score(&labels_true, &labels_pred)
            .unwrap();
        let nmi_val: f64 = nmi.item().unwrap();
        assert!(
            (nmi_val - 1.0).abs() < 1e-6,
            "Expected NMI = 1.0, got {}",
            nmi_val
        );

        let hcv = client
            .homogeneity_completeness_v_measure(&labels_true, &labels_pred)
            .unwrap();
        let h: f64 = hcv.homogeneity.item().unwrap();
        let c: f64 = hcv.completeness.item().unwrap();
        let v: f64 = hcv.v_measure.item().unwrap();
        assert!((h - 1.0).abs() < 1e-6, "Expected H = 1.0, got {}", h);
        assert!((c - 1.0).abs() < 1e-6, "Expected C = 1.0, got {}", c);
        assert!((v - 1.0).abs() < 1e-6, "Expected V = 1.0, got {}", v);
    }
}
