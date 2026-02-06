//! CPU implementation of Bayesian Gaussian Mixture Model.

use crate::cluster::impl_generic::{
    bayesian_gmm_fit_impl, bayesian_gmm_predict_impl, bayesian_gmm_predict_proba_impl,
    bayesian_gmm_score_impl,
};
use crate::cluster::traits::bayesian_gmm::{
    BayesianGmmAlgorithms, BayesianGmmModel, BayesianGmmOptions,
};
use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl BayesianGmmAlgorithms<CpuRuntime> for CpuClient {
    fn bayesian_gmm_fit(
        &self,
        data: &Tensor<CpuRuntime>,
        options: &BayesianGmmOptions,
    ) -> Result<BayesianGmmModel<CpuRuntime>> {
        bayesian_gmm_fit_impl(self, data, options)
    }

    fn bayesian_gmm_predict(
        &self,
        model: &BayesianGmmModel<CpuRuntime>,
        data: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        bayesian_gmm_predict_impl(self, model, data)
    }

    fn bayesian_gmm_predict_proba(
        &self,
        model: &BayesianGmmModel<CpuRuntime>,
        data: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        bayesian_gmm_predict_proba_impl(self, model, data)
    }

    fn bayesian_gmm_score(
        &self,
        model: &BayesianGmmModel<CpuRuntime>,
        data: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        bayesian_gmm_score_impl(self, model, data)
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
    fn test_bayesian_gmm_fit_predict() {
        let (client, device) = setup();

        #[rustfmt::skip]
        let data = Tensor::<CpuRuntime>::from_slice(
            &[
                0.0, 0.0, 0.1, 0.1, 0.2, 0.0, 0.0, 0.2,
                10.0, 10.0, 10.1, 10.1, 10.2, 10.0, 10.0, 10.2,
            ],
            &[8, 2],
            &device,
        );

        let options = BayesianGmmOptions {
            n_components: 2,
            max_iter: 100,
            n_init: 1,
            ..Default::default()
        };

        let model = client.bayesian_gmm_fit(&data, &options).unwrap();
        assert_eq!(model.means.shape(), &[2, 2]);
        assert_eq!(model.weights.shape(), &[2]);

        let labels = client.bayesian_gmm_predict(&model, &data).unwrap();
        assert_eq!(labels.shape(), &[8]);

        // The two groups should be separated — no label shared across groups
        let l: Vec<i64> = labels.to_vec();
        let group_a: std::collections::HashSet<i64> = l[0..4].iter().copied().collect();
        let group_b: std::collections::HashSet<i64> = l[4..8].iter().copied().collect();
        assert!(
            group_a.is_disjoint(&group_b),
            "Groups should be separated: {:?} vs {:?}",
            group_a,
            group_b
        );
    }

    #[test]
    fn test_bayesian_gmm_probabilities() {
        let (client, device) = setup();

        #[rustfmt::skip]
        let data = Tensor::<CpuRuntime>::from_slice(
            &[0.0, 0.0, 0.1, 0.1, 5.0, 5.0, 5.1, 5.1],
            &[4, 2],
            &device,
        );

        let options = BayesianGmmOptions {
            n_components: 3,
            max_iter: 100,
            n_init: 1,
            ..Default::default()
        };

        let model = client.bayesian_gmm_fit(&data, &options).unwrap();
        let proba = client.bayesian_gmm_predict_proba(&model, &data).unwrap();
        assert_eq!(proba.shape(), &[4, 3]);

        // Each row should sum to ~1.0
        let p: Vec<f64> = proba.to_vec();
        for i in 0..4 {
            let row_sum: f64 = (0..3).map(|j| p[i * 3 + j]).sum();
            assert!((row_sum - 1.0).abs() < 1e-4, "row {} sum = {}", i, row_sum);
        }
    }

    #[test]
    fn test_bayesian_gmm_dirichlet_distribution() {
        use crate::cluster::traits::bayesian_gmm::WeightConcentrationPrior;

        let (client, device) = setup();

        #[rustfmt::skip]
        let data = Tensor::<CpuRuntime>::from_slice(
            &[
                0.0, 0.0, 0.1, 0.1, 0.2, 0.0,
                10.0, 10.0, 10.1, 10.1, 10.2, 10.0,
            ],
            &[6, 2],
            &device,
        );

        let options = BayesianGmmOptions {
            n_components: 3,
            max_iter: 50,
            n_init: 1,
            weight_concentration_prior_type: WeightConcentrationPrior::DirichletDistribution,
            ..Default::default()
        };

        let model = client.bayesian_gmm_fit(&data, &options).unwrap();
        assert_eq!(model.means.shape(), &[3, 2]);

        let labels = client.bayesian_gmm_predict(&model, &data).unwrap();
        assert_eq!(labels.shape(), &[6]);
    }
}
