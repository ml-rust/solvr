//! Generic Bayesian Gaussian Mixture Model implementation via variational EM.
//!
//! Uses Dirichlet process or Dirichlet distribution priors on mixture weights
//! and Wishart priors on precision matrices. Automatic component selection
//! via weight concentration — inactive components shrink to zero weight.

use crate::cluster::impl_generic::gmm::GmmClient;
use crate::cluster::traits::bayesian_gmm::{
    BayesianGmmModel, BayesianGmmOptions, WeightConcentrationPrior,
};
use crate::cluster::traits::gmm::{CovarianceType, GmmInit};
use crate::cluster::traits::kmeans::{KMeansInit, KMeansOptions};
use crate::cluster::validation::{validate_cluster_dtype, validate_data_2d, validate_n_clusters};
use numr::dtype::DType;
use numr::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Fit Bayesian GMM to data.
pub fn bayesian_gmm_fit_impl<R, C>(
    client: &C,
    data: &Tensor<R>,
    options: &BayesianGmmOptions,
) -> Result<BayesianGmmModel<R>>
where
    R: Runtime,
    C: GmmClient<R>,
{
    validate_cluster_dtype(data.dtype(), "bayesian_gmm")?;
    validate_data_2d(data.shape(), "bayesian_gmm")?;
    validate_n_clusters(options.n_components, data.shape()[0], "bayesian_gmm")?;

    let mut best_model: Option<BayesianGmmModel<R>> = None;
    let mut best_elbo = f64::NEG_INFINITY;

    for _ in 0..options.n_init {
        let model = bayesian_gmm_fit_single(client, data, options)?;
        if model.lower_bound > best_elbo {
            best_elbo = model.lower_bound;
            best_model = Some(model);
        }
    }

    best_model.ok_or_else(|| numr::error::Error::InvalidArgument {
        arg: "n_init",
        reason: "n_init must be >= 1".to_string(),
    })
}

/// Single variational EM run.
fn bayesian_gmm_fit_single<R, C>(
    client: &C,
    data: &Tensor<R>,
    options: &BayesianGmmOptions,
) -> Result<BayesianGmmModel<R>>
where
    R: Runtime,
    C: GmmClient<R>,
{
    let n = data.shape()[0];
    let d = data.shape()[1];
    let k = options.n_components;
    let dtype = data.dtype();
    let device = data.device();

    // Prior hyperparameters
    let alpha_0 = options.weight_concentration_prior.unwrap_or(1.0 / k as f64);
    let beta_0 = options.mean_precision_prior.unwrap_or(1.0);
    let nu_0 = options.degrees_of_freedom_prior.unwrap_or(d as f64);

    // Initialize means via kmeans or random
    let means = match options.init {
        GmmInit::KMeans => {
            let km_opts = KMeansOptions {
                n_clusters: k,
                max_iter: 10,
                tol: 1e-3,
                n_init: 1,
                init: KMeansInit::KMeansPlusPlus,
                ..Default::default()
            };
            let km_result = super::kmeans::kmeans_impl(client, data, &km_opts)?;
            km_result.centroids
        }
        GmmInit::Random => {
            let perm = client.randperm(n)?;
            let indices = perm.narrow(0, 0, k)?;
            client.index_select(data, 0, &indices)?
        }
    };

    // Prior mean = data mean
    let mean_prior = client.mean(data, &[0], false)?; // [d]

    // Initialize posterior parameters
    let mut nk = Tensor::<R>::full_scalar(&[k], dtype, n as f64 / k as f64, device);
    let mut means_post = means;

    // beta_k = beta_0 + nk
    let beta_0_t = Tensor::<R>::full_scalar(&[k], dtype, beta_0, device);
    let mut beta_k = client.add(&beta_0_t, &nk)?; // [k]

    // nu_k = nu_0 + nk
    let nu_0_t = Tensor::<R>::full_scalar(&[k], dtype, nu_0, device);
    let mut nu_k = client.add(&nu_0_t, &nk)?; // [k]

    // Initialize covariances from data
    let data_var = client.var(data, &[0], false, 1)?; // [d]
    let reg_t = Tensor::<R>::full_scalar(&[1], dtype, options.reg_covar, device);

    let mut covariances = match options.covariance_type {
        CovarianceType::Diagonal => {
            let var_exp = data_var.unsqueeze(0)?.broadcast_to(&[k, d])?;
            client.add(&var_exp, &reg_t.broadcast_to(&[k, d])?)?
        }
        CovarianceType::Spherical => {
            let mean_var = client.mean(&data_var, &[0], false)?;
            let mean_var_k = mean_var.broadcast_to(&[k])?;
            client.add(&mean_var_k, &reg_t.broadcast_to(&[k])?)?
        }
        CovarianceType::Full => {
            let diag = client.diagflat(&data_var)?;
            let reg_eye = client.mul_scalar(
                &client.diagflat(&Tensor::<R>::ones(&[d], dtype, device))?,
                options.reg_covar,
            )?;
            let cov = client.add(&diag, &reg_eye)?;
            cov.unsqueeze(0)?.broadcast_to(&[k, d, d])?.contiguous()
        }
        CovarianceType::Tied => {
            let diag = client.diagflat(&data_var)?;
            let reg_eye = client.mul_scalar(
                &client.diagflat(&Tensor::<R>::ones(&[d], dtype, device))?,
                options.reg_covar,
            )?;
            client.add(&diag, &reg_eye)?
        }
    };

    // Weight concentration posterior
    let mut weight_concentration = match options.weight_concentration_prior_type {
        WeightConcentrationPrior::DirichletProcess => {
            // Stick-breaking: two parameters per component [2, k]
            let ones = Tensor::<R>::ones(&[k], dtype, device);
            let alpha_t = Tensor::<R>::full_scalar(&[k], dtype, alpha_0, device);
            client.cat(&[&ones.unsqueeze(0)?, &alpha_t.unsqueeze(0)?], 0)? // [2, k]
        }
        WeightConcentrationPrior::DirichletDistribution => {
            // Single concentration per component [k]
            Tensor::<R>::full_scalar(&[k], dtype, alpha_0 + n as f64 / k as f64, device)
        }
    };

    let mut prev_elbo = f64::NEG_INFINITY;
    let mut converged = false;
    let mut n_iter = 0;
    let mut lower_bound = f64::NEG_INFINITY;

    for iter in 0..options.max_iter {
        n_iter = iter + 1;

        // E-step: compute expected log weights and log responsibilities
        let log_weights = compute_expected_log_weights(
            client,
            &weight_concentration,
            options.weight_concentration_prior_type,
            k,
            dtype,
            device,
        )?;

        // Expected log det precision (from Wishart posterior)
        let log_det_precision =
            compute_expected_log_det(client, &nu_k, &covariances, options, d, k, dtype, device)?;

        // Compute log responsibilities
        let log_resp = compute_bayesian_log_resp(
            client,
            data,
            &means_post,
            &covariances,
            &beta_k,
            &nu_k,
            &log_weights,
            &log_det_precision,
            options,
            n,
            d,
            k,
            dtype,
            device,
        )?;

        // Compute ELBO (single scalar for convergence)
        let max_log = client.max(&log_resp, &[1], true)?;
        let shifted = client.sub(&log_resp, &max_log)?;
        let exp_shifted = client.exp(&shifted)?;
        let sum_exp = client.sum(&exp_shifted, &[1], true)?;
        let lse = client.add(&client.log(&sum_exp)?, &max_log)?;
        let elbo: f64 = client.mean(&lse, &[0, 1], false)?.item()?;
        lower_bound = elbo;

        if (elbo - prev_elbo).abs() < options.tol {
            converged = true;
            break;
        }
        prev_elbo = elbo;

        // Normalize responsibilities
        let resp = client.exp(&client.sub(&log_resp, &lse.broadcast_to(&[n, k])?)?)?;

        // M-step: update posterior parameters
        nk = client.sum(&resp, &[0], false)?; // [k]
        let nk_safe = client.maximum(&nk, &Tensor::<R>::full_scalar(&[k], dtype, 1e-32, device))?;

        // Update means
        let resp_t = resp.transpose(0, 1)?;
        let weighted_sum = client.matmul(&resp_t, data)?; // [k, d]
        let nk_exp = nk_safe.unsqueeze(1)?.broadcast_to(&[k, d])?;
        let x_bar = client.div(&weighted_sum, &nk_exp)?; // [k, d] — sample means

        // Posterior mean: (beta_0 * m_0 + nk * x_bar) / (beta_0 + nk)
        let beta_0_m0 =
            client.mul_scalar(&mean_prior.unsqueeze(0)?.broadcast_to(&[k, d])?, beta_0)?;
        let nk_xbar = client.mul(&nk_exp, &x_bar)?;
        let numerator = client.add(&beta_0_m0, &nk_xbar)?;
        beta_k = client.add(&beta_0_t, &nk)?;
        let beta_k_exp = beta_k.unsqueeze(1)?.broadcast_to(&[k, d])?;
        means_post = client.div(&numerator, &beta_k_exp)?;

        // Posterior degrees of freedom
        nu_k = client.add(&nu_0_t, &nk)?;

        // Update covariances
        covariances = update_bayesian_covariances(
            client,
            data,
            &resp,
            &x_bar,
            &mean_prior,
            &nk_safe,
            &beta_k,
            options,
            beta_0,
            n,
            d,
            k,
            dtype,
            device,
        )?;

        // Update weight concentration
        weight_concentration =
            update_weight_concentration(client, &nk, options, alpha_0, k, dtype, device)?;
    }

    // Compute effective weights from posterior
    let weights = compute_effective_weights(
        client,
        &weight_concentration,
        options.weight_concentration_prior_type,
        k,
        dtype,
        device,
    )?;

    // Compute precisions
    let precisions_cholesky =
        compute_precisions(client, &covariances, options, k, d, dtype, device)?;

    Ok(BayesianGmmModel {
        weights,
        means: means_post,
        covariances,
        precisions_cholesky,
        weight_concentration,
        mean_precision: beta_k,
        degrees_of_freedom: nu_k,
        converged,
        n_iter,
        lower_bound,
    })
}

/// Compute expected log weights from weight concentration posterior.
fn compute_expected_log_weights<R, C>(
    client: &C,
    weight_concentration: &Tensor<R>,
    prior_type: WeightConcentrationPrior,
    k: usize,
    _dtype: DType,
    _device: &R::Device,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: GmmClient<R>,
{
    match prior_type {
        WeightConcentrationPrior::DirichletProcess => {
            // Stick-breaking: weight_concentration is [2, k]
            // E[log(V_j)] = digamma(alpha_1) - digamma(alpha_1 + alpha_2)
            // E[log(1-V_j)] = digamma(alpha_2) - digamma(alpha_1 + alpha_2)
            // Since we don't have digamma on tensors, use the approximation:
            // digamma(x) ≈ ln(x) - 1/(2x) for large x
            // For the stick-breaking construction, approximate weights as alpha_1/(alpha_1+alpha_2)
            let alpha_1 = weight_concentration.narrow(0, 0, 1)?.squeeze(Some(0)); // [k]
            let alpha_2 = weight_concentration.narrow(0, 1, 1)?.squeeze(Some(0)); // [k]
            let sum_alpha = client.add(&alpha_1, &alpha_2)?;
            let log_weights = client.sub(&client.log(&alpha_1)?, &client.log(&sum_alpha)?)?;
            Ok(log_weights)
        }
        WeightConcentrationPrior::DirichletDistribution => {
            // E[log(pi_j)] = digamma(alpha_j) - digamma(sum(alpha))
            // Approximate: log(alpha_j) - log(sum(alpha))
            let log_alpha = client.log(weight_concentration)?;
            let sum_alpha = client.sum(weight_concentration, &[0], false)?;
            let log_sum = client.log(&sum_alpha)?;
            client.sub(&log_alpha, &log_sum.broadcast_to(&[k])?)
        }
    }
}

/// Compute expected log determinant of precision from Wishart posterior.
#[allow(clippy::too_many_arguments)]
fn compute_expected_log_det<R, C>(
    client: &C,
    nu_k: &Tensor<R>,
    covariances: &Tensor<R>,
    options: &BayesianGmmOptions,
    d: usize,
    k: usize,
    dtype: DType,
    device: &R::Device,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: GmmClient<R>,
{
    // E[log|Lambda|] ≈ -log|Sigma| + d*log(2) + sum_{i=1}^{d} digamma((nu+1-i)/2)
    // Approximate: -log|Sigma| + d*log(nu/2) (since digamma(x) ≈ log(x) for large x)
    match options.covariance_type {
        CovarianceType::Diagonal => {
            // log|Sigma| = sum(log(cov)) per component
            let log_cov = client.log(covariances)?; // [k, d]
            let log_det = client.sum(&log_cov, &[1], false)?; // [k]
            let neg_log_det = client.mul_scalar(&log_det, -1.0)?;
            // + d*log(nu/2)
            let half_nu = client.mul_scalar(nu_k, 0.5)?;
            let log_half_nu = client.log(&half_nu)?;
            let d_log = client.mul_scalar(&log_half_nu, d as f64)?;
            client.add(&neg_log_det, &d_log)
        }
        CovarianceType::Spherical => {
            let log_cov = client.log(covariances)?; // [k]
            let log_det = client.mul_scalar(&log_cov, d as f64)?;
            let neg_log_det = client.mul_scalar(&log_det, -1.0)?;
            let half_nu = client.mul_scalar(nu_k, 0.5)?;
            let log_half_nu = client.log(&half_nu)?;
            let d_log = client.mul_scalar(&log_half_nu, d as f64)?;
            client.add(&neg_log_det, &d_log)
        }
        CovarianceType::Full => {
            // Per-component slogdet
            let mut log_dets = Vec::new();
            for j in 0..k {
                let cov_j = covariances.narrow(0, j, 1)?.contiguous().reshape(&[d, d])?;
                let slogdet = client.slogdet(&cov_j)?;
                log_dets.push(slogdet.logabsdet.unsqueeze(0)?);
            }
            let refs: Vec<&Tensor<R>> = log_dets.iter().collect();
            let log_det = client.cat(&refs, 0)?; // [k]
            let neg_log_det = client.mul_scalar(&log_det, -1.0)?;
            let half_nu = client.mul_scalar(nu_k, 0.5)?;
            let log_half_nu = client.log(&half_nu)?;
            let d_log = client.mul_scalar(&log_half_nu, d as f64)?;
            client.add(&neg_log_det, &d_log)
        }
        CovarianceType::Tied => {
            // Single covariance
            let slogdet = client.slogdet(covariances)?;
            let log_det: f64 = slogdet.logabsdet.item()?;
            // Return [k] with same value for all components
            let half_nu = client.mul_scalar(nu_k, 0.5)?;
            let log_half_nu = client.log(&half_nu)?;
            let d_log = client.mul_scalar(&log_half_nu, d as f64)?;
            let neg_det = Tensor::<R>::full_scalar(&[k], dtype, -log_det, device);
            client.add(&neg_det, &d_log)
        }
    }
}

/// Compute Bayesian log responsibilities.
#[allow(clippy::too_many_arguments)]
fn compute_bayesian_log_resp<R, C>(
    client: &C,
    data: &Tensor<R>,
    means: &Tensor<R>,
    covariances: &Tensor<R>,
    beta_k: &Tensor<R>,
    nu_k: &Tensor<R>,
    log_weights: &Tensor<R>,
    log_det_precision: &Tensor<R>,
    options: &BayesianGmmOptions,
    n: usize,
    d: usize,
    k: usize,
    dtype: DType,
    device: &R::Device,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: GmmClient<R>,
{
    let log_2pi = (2.0 * std::f64::consts::PI).ln();

    // Compute Mahalanobis-like distances weighted by expected precision
    let maha = match options.covariance_type {
        CovarianceType::Diagonal => {
            let data_exp = data.unsqueeze(1)?.broadcast_to(&[n, k, d])?;
            let means_exp = means.unsqueeze(0)?.broadcast_to(&[n, k, d])?;
            let diff = client.sub(&data_exp, &means_exp)?;
            let diff_sq = client.mul(&diff, &diff)?;
            let cov_exp = covariances.unsqueeze(0)?.broadcast_to(&[n, k, d])?;
            let scaled = client.div(&diff_sq, &cov_exp)?;
            client.sum(&scaled, &[2], false)? // [n, k]
        }
        CovarianceType::Spherical => {
            let sq_dists =
                client.cdist(data, means, numr::ops::DistanceMetric::SquaredEuclidean)?;
            let cov_exp = covariances.unsqueeze(0)?.broadcast_to(&[n, k])?;
            client.div(&sq_dists, &cov_exp)?
        }
        CovarianceType::Full => {
            let mut maha_slices = Vec::new();
            for j in 0..k {
                let cov_j = covariances.narrow(0, j, 1)?.contiguous().reshape(&[d, d])?;
                let inv_cov = client.inverse(&cov_j)?;
                let mean_j = means.narrow(0, j, 1)?;
                let diff = client.sub(data, &mean_j.broadcast_to(&[n, d])?)?;
                let tmp = client.matmul(&diff, &inv_cov)?;
                let m_j = client.sum(&client.mul(&tmp, &diff)?, &[1], false)?;
                maha_slices.push(m_j.unsqueeze(1)?);
            }
            let refs: Vec<&Tensor<R>> = maha_slices.iter().collect();
            client.cat(&refs, 1)?
        }
        CovarianceType::Tied => {
            let inv_cov = client.inverse(covariances)?;
            let mut maha_slices = Vec::new();
            for j in 0..k {
                let mean_j = means.narrow(0, j, 1)?;
                let diff = client.sub(data, &mean_j.broadcast_to(&[n, d])?)?;
                let tmp = client.matmul(&diff, &inv_cov)?;
                let m_j = client.sum(&client.mul(&tmp, &diff)?, &[1], false)?;
                maha_slices.push(m_j.unsqueeze(1)?);
            }
            let refs: Vec<&Tensor<R>> = maha_slices.iter().collect();
            client.cat(&refs, 1)?
        }
    };

    // Multiply Mahalanobis by nu_k (expected precision scaling)
    let nu_exp = nu_k.unsqueeze(0)?.broadcast_to(&[n, k])?;
    let weighted_maha = client.mul(&maha, &nu_exp)?;

    // Add correction term d/beta_k (uncertainty in mean)
    let d_over_beta = client.div_scalar(
        &Tensor::<R>::full_scalar(&[k], dtype, d as f64, device),
        1.0,
    )?;
    let d_over_beta = client.div(&d_over_beta, beta_k)?; // [k]
    let d_over_beta_exp = d_over_beta.unsqueeze(0)?.broadcast_to(&[n, k])?;

    // log_resp = log_weights + 0.5*log_det_precision - 0.5*d*log(2pi) - 0.5*(nu*maha + d/beta)
    let const_term = Tensor::<R>::full_scalar(&[1], dtype, -0.5 * d as f64 * log_2pi, device);
    let half_log_det =
        client.mul_scalar(&log_det_precision.unsqueeze(0)?.broadcast_to(&[n, k])?, 0.5)?;
    let half_maha = client.mul_scalar(&client.add(&weighted_maha, &d_over_beta_exp)?, -0.5)?;

    let log_w_exp = log_weights.unsqueeze(0)?.broadcast_to(&[n, k])?;
    let result = client.add(&log_w_exp, &half_log_det)?;
    let result = client.add(&result, &const_term.broadcast_to(&[n, k])?)?;
    client.add(&result, &half_maha)
}

/// Update covariances with Bayesian prior contribution.
#[allow(clippy::too_many_arguments)]
fn update_bayesian_covariances<R, C>(
    client: &C,
    data: &Tensor<R>,
    resp: &Tensor<R>,
    x_bar: &Tensor<R>,
    mean_prior: &Tensor<R>,
    nk: &Tensor<R>,
    beta_k: &Tensor<R>,
    options: &BayesianGmmOptions,
    beta_0: f64,
    n: usize,
    d: usize,
    k: usize,
    dtype: DType,
    device: &R::Device,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: GmmClient<R>,
{
    let reg = options.reg_covar;

    match options.covariance_type {
        CovarianceType::Diagonal => {
            // Same as standard GMM M-step but with prior contribution
            let data_exp = data.unsqueeze(1)?.broadcast_to(&[n, k, d])?;
            let means_exp = x_bar.unsqueeze(0)?.broadcast_to(&[n, k, d])?;
            let diff = client.sub(&data_exp, &means_exp)?;
            let diff_sq = client.mul(&diff, &diff)?;
            let resp_exp = resp.unsqueeze(2)?.broadcast_to(&[n, k, d])?;
            let weighted = client.mul(&resp_exp, &diff_sq)?;
            let s_k = client.sum(&weighted, &[0], false)?; // [k, d]
            let nk_exp = nk.unsqueeze(1)?.broadcast_to(&[k, d])?;
            let cov_ml = client.div(&s_k, &nk_exp)?;

            // Prior contribution: beta_0 * nk / beta_k * (x_bar - m_0)^2
            let m0_exp = mean_prior.unsqueeze(0)?.broadcast_to(&[k, d])?;
            let diff_prior = client.sub(x_bar, &m0_exp)?;
            let diff_prior_sq = client.mul(&diff_prior, &diff_prior)?;
            let beta_ratio = client.div(
                &client.mul_scalar(&nk_exp, beta_0)?,
                &beta_k.unsqueeze(1)?.broadcast_to(&[k, d])?,
            )?;
            let prior_term = client.mul(&beta_ratio, &diff_prior_sq)?;

            let cov = client.add(&cov_ml, &prior_term)?;
            let reg_t = Tensor::<R>::full_scalar(&[k, d], dtype, reg, device);
            client.add(&cov, &reg_t)
        }
        CovarianceType::Spherical => {
            let data_exp = data.unsqueeze(1)?.broadcast_to(&[n, k, d])?;
            let means_exp = x_bar.unsqueeze(0)?.broadcast_to(&[n, k, d])?;
            let diff = client.sub(&data_exp, &means_exp)?;
            let diff_sq = client.mul(&diff, &diff)?;
            let resp_exp = resp.unsqueeze(2)?.broadcast_to(&[n, k, d])?;
            let weighted = client.mul(&resp_exp, &diff_sq)?;
            let s_k = client.sum(&weighted, &[0], false)?; // [k, d]
            let nk_exp = nk.unsqueeze(1)?.broadcast_to(&[k, d])?;
            let cov_diag = client.div(&s_k, &nk_exp)?;
            let cov_ml = client.mean(&cov_diag, &[1], false)?; // [k]

            // Prior contribution
            let m0_exp = mean_prior.unsqueeze(0)?.broadcast_to(&[k, d])?;
            let diff_prior = client.sub(x_bar, &m0_exp)?;
            let diff_prior_sq = client.mul(&diff_prior, &diff_prior)?;
            let prior_mean = client.mean(&diff_prior_sq, &[1], false)?; // [k]
            let beta_ratio = client.div(&client.mul_scalar(nk, beta_0)?, beta_k)?;
            let prior_term = client.mul(&beta_ratio, &prior_mean)?;

            let cov = client.add(&cov_ml, &prior_term)?;
            let reg_t = Tensor::<R>::full_scalar(&[k], dtype, reg, device);
            client.add(&cov, &reg_t)
        }
        CovarianceType::Full => {
            let mut cov_slices = Vec::new();
            for j in 0..k {
                let mean_j = x_bar.narrow(0, j, 1)?;
                let diff = client.sub(data, &mean_j.broadcast_to(&[n, d])?)?;
                let resp_j = resp.narrow(1, j, 1)?;
                let weighted_diff = client.mul(&diff, &resp_j.broadcast_to(&[n, d])?)?;
                let s_j = client.matmul(&weighted_diff.transpose(0, 1)?, &diff)?;
                let nk_j = nk.narrow(0, j, 1)?;
                let cov_ml = client.div(&s_j, &nk_j.broadcast_to(&[d, d])?)?;

                // Prior: beta_0*nk_j/beta_k_j * (x_bar_j - m_0)(x_bar_j - m_0)^T
                let m0 = mean_prior.unsqueeze(0)?; // [1, d]
                let diff_prior = client.sub(&mean_j, &m0)?; // [1, d]
                let outer = client.matmul(&diff_prior.transpose(0, 1)?, &diff_prior)?; // [d, d]
                let beta_k_j = beta_k.narrow(0, j, 1)?;
                // scale = beta_0 * nk_j / beta_k_j — computed on-device
                let nk_beta = client.div(&nk_j, &beta_k_j)?; // scalar tensor
                let nk_beta_scaled = client.mul_scalar(&nk_beta, beta_0)?;
                let prior_term = client.mul(&outer, &nk_beta_scaled.broadcast_to(&[d, d])?)?;

                let cov_j = client.add(&cov_ml, &prior_term)?;
                let reg_eye = client.mul_scalar(
                    &client.diagflat(&Tensor::<R>::ones(&[d], dtype, device))?,
                    reg,
                )?;
                let cov_j = client.add(&cov_j, &reg_eye)?;
                cov_slices.push(cov_j.unsqueeze(0)?);
            }
            let refs: Vec<&Tensor<R>> = cov_slices.iter().collect();
            client.cat(&refs, 0)
        }
        CovarianceType::Tied => {
            let mut total_cov = Tensor::<R>::zeros(&[d, d], dtype, device);
            for j in 0..k {
                let mean_j = x_bar.narrow(0, j, 1)?;
                let diff = client.sub(data, &mean_j.broadcast_to(&[n, d])?)?;
                let resp_j = resp.narrow(1, j, 1)?;
                let weighted = client.mul(&diff, &resp_j.broadcast_to(&[n, d])?)?;
                let cov_j = client.matmul(&weighted.transpose(0, 1)?, &diff)?;
                total_cov = client.add(&total_cov, &cov_j)?;
            }
            let n_f = Tensor::<R>::full_scalar(&[d, d], dtype, n as f64, device);
            let cov_ml = client.div(&total_cov, &n_f)?;

            // Prior contribution (averaged)
            let m0 = mean_prior.unsqueeze(0)?.broadcast_to(&[k, d])?;
            let diff_prior = client.sub(x_bar, &m0)?; // [k, d]
            let mut prior_outer = Tensor::<R>::zeros(&[d, d], dtype, device);
            for j in 0..k {
                let dp_j = diff_prior.narrow(0, j, 1)?; // [1, d]
                let outer = client.matmul(&dp_j.transpose(0, 1)?, &dp_j)?;
                let nk_j = nk.narrow(0, j, 1)?;
                let beta_k_j = beta_k.narrow(0, j, 1)?;
                // scale = beta_0 * nk_j / beta_k_j — computed on-device
                let nk_beta = client.div(&nk_j, &beta_k_j)?;
                let nk_beta_scaled = client.mul_scalar(&nk_beta, beta_0)?;
                let scaled = client.mul(&outer, &nk_beta_scaled.broadcast_to(&[d, d])?)?;
                prior_outer = client.add(&prior_outer, &scaled)?;
            }
            let prior_avg = client.div_scalar(&prior_outer, k as f64)?;
            let cov = client.add(&cov_ml, &prior_avg)?;

            let reg_eye = client.mul_scalar(
                &client.diagflat(&Tensor::<R>::ones(&[d], dtype, device))?,
                reg,
            )?;
            client.add(&cov, &reg_eye)
        }
    }
}

/// Update weight concentration posterior.
fn update_weight_concentration<R, C>(
    client: &C,
    nk: &Tensor<R>,
    options: &BayesianGmmOptions,
    alpha_0: f64,
    k: usize,
    dtype: DType,
    device: &R::Device,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: GmmClient<R>,
{
    match options.weight_concentration_prior_type {
        WeightConcentrationPrior::DirichletProcess => {
            // Stick-breaking update:
            // alpha_1[j] = 1 + nk[j]
            // alpha_2[j] = alpha_0 + sum(nk[j+1:])
            let ones = Tensor::<R>::ones(&[k], dtype, device);
            let alpha_1 = client.add(&ones, nk)?; // [k]

            // Reverse cumsum for alpha_2: sum(nk[j+1:]) = total - cumsum(nk)[j]
            let total_nk = client.sum(nk, &[0], false)?;
            let cum_nk = client.cumsum(nk, 0)?;
            let remaining = client.sub(&total_nk.broadcast_to(&[k])?, &cum_nk)?;
            let alpha_0_t = Tensor::<R>::full_scalar(&[k], dtype, alpha_0, device);
            let alpha_2 = client.add(&alpha_0_t, &remaining)?; // [k]

            client.cat(&[&alpha_1.unsqueeze(0)?, &alpha_2.unsqueeze(0)?], 0) // [2, k]
        }
        WeightConcentrationPrior::DirichletDistribution => {
            // alpha_j = alpha_0 + nk_j
            let alpha_0_t = Tensor::<R>::full_scalar(&[k], dtype, alpha_0, device);
            client.add(&alpha_0_t, nk)
        }
    }
}

/// Compute effective weights from posterior concentration.
fn compute_effective_weights<R, C>(
    client: &C,
    weight_concentration: &Tensor<R>,
    prior_type: WeightConcentrationPrior,
    k: usize,
    dtype: DType,
    device: &R::Device,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: GmmClient<R>,
{
    match prior_type {
        WeightConcentrationPrior::DirichletProcess => {
            // Stick-breaking weights: w_j = V_j * prod(1 - V_i, i < j)
            // V_j = alpha_1[j] / (alpha_1[j] + alpha_2[j])
            let alpha_1 = weight_concentration.narrow(0, 0, 1)?.squeeze(Some(0));
            let alpha_2 = weight_concentration.narrow(0, 1, 1)?.squeeze(Some(0));
            let sum_alpha = client.add(&alpha_1, &alpha_2)?;
            let v = client.div(&alpha_1, &sum_alpha)?; // [k]

            // log(1 - V_j)
            let one = Tensor::<R>::ones(&[k], dtype, device);
            let one_minus_v = client.sub(&one, &v)?;
            let log_one_minus_v = client.log(&client.maximum(
                &one_minus_v,
                &Tensor::<R>::full_scalar(&[k], dtype, 1e-32, device),
            )?)?;

            // Cumulative sum of log(1-V) shifted right (exclusive prefix sum)
            let cum_log = client.cumsum(&log_one_minus_v, 0)?;
            // Shift right: [0, cum[0], cum[1], ..., cum[k-2]]
            let zero = Tensor::<R>::zeros(&[1], dtype, device);
            let cum_shifted = client.cat(&[&zero, &cum_log.narrow(0, 0, k - 1)?], 0)?;

            let log_v = client
                .log(&client.maximum(&v, &Tensor::<R>::full_scalar(&[k], dtype, 1e-32, device))?)?;
            let log_weights = client.add(&log_v, &cum_shifted)?;
            let weights = client.exp(&log_weights)?;

            // Normalize
            let total = client.sum(&weights, &[0], false)?;
            client.div(&weights, &total.broadcast_to(&[k])?)
        }
        WeightConcentrationPrior::DirichletDistribution => {
            // Dirichlet: w_j = alpha_j / sum(alpha)
            let total = client.sum(weight_concentration, &[0], false)?;
            client.div(weight_concentration, &total.broadcast_to(&[k])?)
        }
    }
}

/// Compute precision matrices from covariances.
#[allow(clippy::too_many_arguments)]
fn compute_precisions<R, C>(
    client: &C,
    covariances: &Tensor<R>,
    options: &BayesianGmmOptions,
    k: usize,
    d: usize,
    dtype: DType,
    device: &R::Device,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: GmmClient<R>,
{
    match options.covariance_type {
        CovarianceType::Diagonal => client.div(
            &Tensor::<R>::ones(&[k, d], dtype, device),
            &client.sqrt(covariances)?,
        ),
        CovarianceType::Spherical => client.div(
            &Tensor::<R>::ones(&[k], dtype, device),
            &client.sqrt(covariances)?,
        ),
        CovarianceType::Full => {
            let mut inv_slices = Vec::new();
            for j in 0..k {
                let cov_j = covariances.narrow(0, j, 1)?.contiguous().reshape(&[d, d])?;
                let inv_j = client.inverse(&cov_j)?;
                inv_slices.push(inv_j.unsqueeze(0)?);
            }
            let refs: Vec<&Tensor<R>> = inv_slices.iter().collect();
            client.cat(&refs, 0)
        }
        CovarianceType::Tied => client.inverse(covariances),
    }
}

/// Predict most likely component.
pub fn bayesian_gmm_predict_impl<R, C>(
    client: &C,
    model: &BayesianGmmModel<R>,
    data: &Tensor<R>,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: GmmClient<R>,
{
    let proba = bayesian_gmm_predict_proba_impl(client, model, data)?;
    client.argmax(&proba, 1, false)
}

/// Predict component probabilities.
pub fn bayesian_gmm_predict_proba_impl<R, C>(
    client: &C,
    model: &BayesianGmmModel<R>,
    data: &Tensor<R>,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: GmmClient<R>,
{
    // Use the effective weights and standard Gaussian log-likelihood
    // (point estimates from the posterior, not full Bayesian prediction)
    use crate::cluster::traits::gmm::GmmModel;

    let gmm_model = GmmModel {
        weights: model.weights.clone(),
        means: model.means.clone(),
        covariances: model.covariances.clone(),
        precisions_cholesky: model.precisions_cholesky.clone(),
        converged: model.converged,
        n_iter: model.n_iter,
        lower_bound: model.lower_bound,
    };

    super::gmm::gmm_predict_proba_impl(client, &gmm_model, data)
}

/// Compute per-sample log-likelihood.
pub fn bayesian_gmm_score_impl<R, C>(
    client: &C,
    model: &BayesianGmmModel<R>,
    data: &Tensor<R>,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: GmmClient<R>,
{
    use crate::cluster::traits::gmm::GmmModel;

    let gmm_model = GmmModel {
        weights: model.weights.clone(),
        means: model.means.clone(),
        covariances: model.covariances.clone(),
        precisions_cholesky: model.precisions_cholesky.clone(),
        converged: model.converged,
        n_iter: model.n_iter,
        lower_bound: model.lower_bound,
    };

    super::gmm::gmm_score_impl(client, &gmm_model, data)
}
