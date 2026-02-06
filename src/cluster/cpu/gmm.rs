//! CPU implementation of Gaussian Mixture Model.

use crate::cluster::impl_generic::{
    gmm_fit_impl, gmm_predict_impl, gmm_predict_proba_impl, gmm_score_impl,
};
use crate::cluster::traits::gmm::{GmmAlgorithms, GmmModel, GmmOptions};
use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl GmmAlgorithms<CpuRuntime> for CpuClient {
    fn gmm_fit(
        &self,
        data: &Tensor<CpuRuntime>,
        options: &GmmOptions,
    ) -> Result<GmmModel<CpuRuntime>> {
        gmm_fit_impl(self, data, options)
    }

    fn gmm_predict(
        &self,
        model: &GmmModel<CpuRuntime>,
        data: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        gmm_predict_impl(self, model, data)
    }

    fn gmm_predict_proba(
        &self,
        model: &GmmModel<CpuRuntime>,
        data: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        gmm_predict_proba_impl(self, model, data)
    }

    fn gmm_score(
        &self,
        model: &GmmModel<CpuRuntime>,
        data: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        gmm_score_impl(self, model, data)
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
    fn test_gmm_fit_predict() {
        let (client, device) = setup();

        #[rustfmt::skip]
        let data = Tensor::<CpuRuntime>::from_slice(
            &[
                0.0, 0.0,
                0.1, 0.1,
                0.2, 0.0,
                0.0, 0.2,
                10.0, 10.0,
                10.1, 10.1,
                10.2, 10.0,
                10.0, 10.2,
            ],
            &[8, 2],
            &device,
        );

        let options = GmmOptions {
            n_components: 2,
            max_iter: 100,
            n_init: 1,
            ..Default::default()
        };

        let model = client.gmm_fit(&data, &options).unwrap();
        assert_eq!(model.means.shape(), &[2, 2]);
        assert_eq!(model.weights.shape(), &[2]);

        let labels = client.gmm_predict(&model, &data).unwrap();
        assert_eq!(labels.shape(), &[8]);

        let proba = client.gmm_predict_proba(&model, &data).unwrap();
        assert_eq!(proba.shape(), &[8, 2]);

        let scores = client.gmm_score(&model, &data).unwrap();
        assert_eq!(scores.shape(), &[8]);
    }

    #[test]
    fn test_gmm_cluster_assignment() {
        let (client, device) = setup();

        // Two well-separated clusters — GMM should assign them to different components
        #[rustfmt::skip]
        let data = Tensor::<CpuRuntime>::from_slice(
            &[
                0.0, 0.0, 0.1, 0.1, 0.2, 0.0, 0.0, 0.2,
                10.0, 10.0, 10.1, 10.1, 10.2, 10.0, 10.0, 10.2,
            ],
            &[8, 2],
            &device,
        );

        let options = GmmOptions {
            n_components: 2,
            max_iter: 100,
            n_init: 3,
            ..Default::default()
        };

        let model = client.gmm_fit(&data, &options).unwrap();
        let labels = client.gmm_predict(&model, &data).unwrap();
        let l: Vec<f64> = labels.to_vec();
        // First 4 points should share a label, last 4 another
        assert_eq!(l[0], l[1]);
        assert_eq!(l[1], l[2]);
        assert_eq!(l[2], l[3]);
        assert_eq!(l[4], l[5]);
        assert_eq!(l[5], l[6]);
        assert_eq!(l[6], l[7]);
        assert_ne!(l[0], l[4]);
    }

    #[test]
    fn test_gmm_probabilities_sum_to_one() {
        let (client, device) = setup();

        #[rustfmt::skip]
        let data = Tensor::<CpuRuntime>::from_slice(
            &[0.0, 0.0, 0.1, 0.1, 5.0, 5.0, 5.1, 5.1],
            &[4, 2],
            &device,
        );

        let options = GmmOptions {
            n_components: 2,
            max_iter: 100,
            n_init: 1,
            ..Default::default()
        };

        let model = client.gmm_fit(&data, &options).unwrap();
        let proba = client.gmm_predict_proba(&model, &data).unwrap();
        let p: Vec<f64> = proba.to_vec();
        // Each row should sum to ~1.0
        for i in 0..4 {
            let row_sum = p[i * 2] + p[i * 2 + 1];
            assert!((row_sum - 1.0).abs() < 1e-5, "row {} sum = {}", i, row_sum);
        }
    }
}
