//! Generic Gaussian Mixture Model implementation via EM algorithm.
//!
//! Supports Full, Tied, Diagonal, and Spherical covariance types.
//! All computation on-device; single scalar (log-likelihood) transfer per iteration.

use crate::cluster::traits::gmm::{CovarianceType, GmmInit, GmmModel, GmmOptions};
use crate::cluster::traits::kmeans::{KMeansInit, KMeansOptions};
use crate::cluster::validation::{validate_cluster_dtype, validate_data_2d, validate_n_clusters};
use numr::dtype::DType;
use numr::error::Result;
use numr::ops::{
    CompareOps, ConditionalOps, CumulativeOps, DistanceMetric, DistanceOps, IndexingOps, LinalgOps,
    MatmulOps, RandomOps, ReduceOps, ScalarOps, ShapeOps, SortingOps, StatisticalOps, TensorOps,
    TypeConversionOps, UnaryOps, UtilityOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Trait bounds needed for GMM.
pub trait GmmClient<R: Runtime>:
    DistanceOps<R>
    + IndexingOps<R>
    + ReduceOps<R>
    + ScalarOps<R>
    + TensorOps<R>
    + TypeConversionOps<R>
    + UnaryOps<R>
    + CumulativeOps<R>
    + ConditionalOps<R>
    + CompareOps<R>
    + RandomOps<R>
    + SortingOps<R>
    + ShapeOps<R>
    + UtilityOps<R>
    + MatmulOps<R>
    + LinalgOps<R>
    + StatisticalOps<R>
    + RuntimeClient<R>
{
}

impl<R, C> GmmClient<R> for C
where
    R: Runtime,
    C: DistanceOps<R>
        + IndexingOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TensorOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + CumulativeOps<R>
        + ConditionalOps<R>
        + CompareOps<R>
        + RandomOps<R>
        + SortingOps<R>
        + ShapeOps<R>
        + UtilityOps<R>
        + MatmulOps<R>
        + LinalgOps<R>
        + StatisticalOps<R>
        + RuntimeClient<R>,
{
}

/// Fit GMM to data.
pub fn gmm_fit_impl<R, C>(client: &C, data: &Tensor<R>, options: &GmmOptions) -> Result<GmmModel<R>>
where
    R: Runtime,
    C: GmmClient<R>,
{
    validate_cluster_dtype(data.dtype(), "gmm")?;
    validate_data_2d(data.shape(), "gmm")?;
    validate_n_clusters(options.n_components, data.shape()[0], "gmm")?;

    let n = data.shape()[0];
    let d = data.shape()[1];
    let k = options.n_components;
    let dtype = data.dtype();
    let device = data.device();

    let mut best_model: Option<GmmModel<R>> = None;
    let mut best_ll = f64::NEG_INFINITY;

    for _ in 0..options.n_init {
        let model = gmm_fit_single(client, data, options, n, d, k, dtype, device)?;
        if model.lower_bound > best_ll {
            best_ll = model.lower_bound;
            best_model = Some(model);
        }
    }

    best_model.ok_or_else(|| numr::error::Error::InvalidArgument {
        arg: "n_init",
        reason: "n_init must be >= 1 to produce a model".to_string(),
    })
}

#[allow(clippy::too_many_arguments)]
fn gmm_fit_single<R, C>(
    client: &C,
    data: &Tensor<R>,
    options: &GmmOptions,
    n: usize,
    d: usize,
    k: usize,
    dtype: DType,
    device: &R::Device,
) -> Result<GmmModel<R>>
where
    R: Runtime,
    C: GmmClient<R>,
{
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

    // Initialize weights uniformly
    let mut weights = Tensor::<R>::full_scalar(&[k], dtype, 1.0 / k as f64, device);
    let mut means = means;

    // Initialize covariances based on data variance
    let data_var = client.var(data, &[0], false, 1)?; // [d]
    let reg = Tensor::<R>::full_scalar(&[1], dtype, options.reg_covar, device);

    let mut covariances = match options.covariance_type {
        CovarianceType::Diagonal => {
            // [k, d] — each component starts with data variance
            let var_exp = data_var.unsqueeze(0)?.broadcast_to(&[k, d])?;
            client.add(&var_exp, &reg.broadcast_to(&[k, d])?)?
        }
        CovarianceType::Spherical => {
            // [k] — mean of data variance
            let mean_var = client.mean(&data_var, &[0], false)?; // scalar
            let mean_var_k = mean_var.broadcast_to(&[k])?;
            client.add(&mean_var_k, &reg.broadcast_to(&[k])?)?
        }
        CovarianceType::Full => {
            // [k, d, d] — each component starts with diagonal of data variance
            let diag = client.diagflat(&data_var)?; // [d, d]
            let reg_eye = client.mul_scalar(
                &client.diagflat(&Tensor::<R>::ones(&[d], dtype, device))?,
                options.reg_covar,
            )?;
            let cov = client.add(&diag, &reg_eye)?;
            cov.unsqueeze(0)?.broadcast_to(&[k, d, d])?.contiguous()
        }
        CovarianceType::Tied => {
            // [d, d] — single covariance for all components
            let diag = client.diagflat(&data_var)?;
            let reg_eye = client.mul_scalar(
                &client.diagflat(&Tensor::<R>::ones(&[d], dtype, device))?,
                options.reg_covar,
            )?;
            client.add(&diag, &reg_eye)?
        }
    };

    let mut prev_ll = f64::NEG_INFINITY;
    let mut converged = false;
    let mut n_iter = 0;
    let mut lower_bound = f64::NEG_INFINITY;

    for iter in 0..options.max_iter {
        n_iter = iter + 1;

        // E-step: compute log responsibilities
        let log_resp = compute_log_responsibilities(
            client,
            data,
            &weights,
            &means,
            &covariances,
            options,
            n,
            d,
            k,
            dtype,
            device,
        )?;

        // Log-likelihood
        // log_resp is [n, k], responsibilities in log space
        // log_likelihood = mean of logsumexp(log_resp, dim=1)
        let max_log = client.max(&log_resp, &[1], true)?; // [n, 1]
        let shifted = client.sub(&log_resp, &max_log)?;
        let exp_shifted = client.exp(&shifted)?;
        let sum_exp = client.sum(&exp_shifted, &[1], true)?; // [n, 1]
        let lse = client.add(&client.log(&sum_exp)?, &max_log)?; // [n, 1]
        let ll: f64 = client.mean(&lse, &[0, 1], false)?.item()?;
        lower_bound = ll;

        if (ll - prev_ll).abs() < options.tol {
            converged = true;
            break;
        }
        prev_ll = ll;

        // Normalize responsibilities: resp = softmax(log_resp)
        let resp = client.exp(&client.sub(&log_resp, &lse.broadcast_to(&[n, k])?)?)?; // [n, k]

        // M-step
        let nk = client.sum(&resp, &[0], false)?; // [k]
        let nk_safe = client.maximum(&nk, &Tensor::<R>::full_scalar(&[k], dtype, 1e-32, device))?;

        // Update weights
        weights = client.div_scalar(&nk_safe, n as f64)?;

        // Update means: means[j] = sum(resp[:,j] * data) / nk[j]
        // resp_t [k, n] @ data [n, d] = [k, d]
        let resp_t = resp.transpose(0, 1)?; // [k, n]
        let weighted_sum = client.matmul(&resp_t, data)?; // [k, d]
        let nk_exp = nk_safe.unsqueeze(1)?.broadcast_to(&[k, d])?;
        means = client.div(&weighted_sum, &nk_exp)?;

        // Update covariances
        covariances = update_covariances(
            client, data, &resp, &means, &nk_safe, options, n, d, k, dtype, device,
        )?;
    }

    // Compute precisions_cholesky (just store inverse variance for diagonal/spherical,
    // or inverse covariance for full/tied)
    let precisions_cholesky = match options.covariance_type {
        CovarianceType::Diagonal => {
            // 1/sqrt(cov) [k, d]
            client.div(
                &Tensor::<R>::ones(&[k, d], dtype, device),
                &client.sqrt(&covariances)?,
            )?
        }
        CovarianceType::Spherical => {
            // 1/sqrt(cov) [k]
            client.div(
                &Tensor::<R>::ones(&[k], dtype, device),
                &client.sqrt(&covariances)?,
            )?
        }
        CovarianceType::Full => {
            // Store inverse covariance [k, d, d]
            // Process each component
            // For now, compute and store as [k, d, d]
            let mut inv_slices = Vec::new();
            for j in 0..k {
                let cov_j = covariances.narrow(0, j, 1)?.contiguous().reshape(&[d, d])?;
                let inv_j = client.inverse(&cov_j)?;
                inv_slices.push(inv_j.unsqueeze(0)?);
            }
            let refs: Vec<&Tensor<R>> = inv_slices.iter().collect();
            client.cat(&refs, 0)?
        }
        CovarianceType::Tied => client.inverse(&covariances)?,
    };

    Ok(GmmModel {
        weights,
        means,
        covariances,
        precisions_cholesky,
        converged,
        n_iter,
        lower_bound,
    })
}

/// Compute log responsibilities [n, k].
#[allow(clippy::too_many_arguments)]
fn compute_log_responsibilities<R, C>(
    client: &C,
    data: &Tensor<R>,
    weights: &Tensor<R>,
    means: &Tensor<R>,
    covariances: &Tensor<R>,
    options: &GmmOptions,
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
    // log_resp[i, j] = log(weight[j]) + log_gaussian(x_i; mean_j, cov_j)
    let log_weights = client.log(weights)?; // [k]
    let log_weights_exp = log_weights.unsqueeze(0)?.broadcast_to(&[n, k])?; // [n, k]

    let log_gauss = match options.covariance_type {
        CovarianceType::Diagonal => {
            // log N(x; mu, diag(sigma^2)) = -0.5 * (d*log(2pi) + sum(log(sigma^2)) + sum((x-mu)^2 / sigma^2))
            let log_2pi = (2.0 * std::f64::consts::PI).ln();

            // For each component j: compute log det and Mahalanobis distance
            // covariances is [k, d]
            let log_cov = client.log(covariances)?; // [k, d]
            let log_det = client.sum(&log_cov, &[1], false)?; // [k]

            // (x - mu)^2 / sigma^2 for all pairs
            // data [n, d], means [k, d]
            // Expand: data [n, 1, d], means [1, k, d]
            let data_exp = data.unsqueeze(1)?.broadcast_to(&[n, k, d])?; // [n, k, d]
            let means_exp = means.unsqueeze(0)?.broadcast_to(&[n, k, d])?; // [n, k, d]
            let diff = client.sub(&data_exp, &means_exp)?; // [n, k, d]
            let diff_sq = client.mul(&diff, &diff)?; // [n, k, d]
            let cov_exp = covariances.unsqueeze(0)?.broadcast_to(&[n, k, d])?;
            let maha = client.div(&diff_sq, &cov_exp)?; // [n, k, d]
            let maha_sum = client.sum(&maha, &[2], false)?; // [n, k]

            // log_gauss = -0.5 * (d*log(2pi) + log_det + maha)
            let const_term =
                Tensor::<R>::full_scalar(&[1], dtype, -0.5 * d as f64 * log_2pi, device);
            let log_det_exp = log_det.unsqueeze(0)?.broadcast_to(&[n, k])?;
            let half = Tensor::<R>::full_scalar(&[1], dtype, -0.5, device);
            let log_det_term = client.mul(&half.broadcast_to(&[n, k])?, &log_det_exp)?;
            let maha_term = client.mul(&half.broadcast_to(&[n, k])?, &maha_sum)?;
            let result = client.add(&const_term.broadcast_to(&[n, k])?, &log_det_term)?;
            client.add(&result, &maha_term)?
        }
        CovarianceType::Spherical => {
            let log_2pi = (2.0 * std::f64::consts::PI).ln();
            // covariances is [k]
            let log_cov = client.log(covariances)?; // [k]
            let log_det = client.mul_scalar(&log_cov, d as f64)?; // [k] (d * log(sigma^2))

            // Squared distances
            let sq_dists = client.cdist(data, means, DistanceMetric::SquaredEuclidean)?; // [n, k]
            let cov_exp = covariances.unsqueeze(0)?.broadcast_to(&[n, k])?;
            let maha = client.div(&sq_dists, &cov_exp)?; // [n, k]

            let const_term =
                Tensor::<R>::full_scalar(&[1], dtype, -0.5 * d as f64 * log_2pi, device);
            let log_det_exp = log_det.unsqueeze(0)?.broadcast_to(&[n, k])?;
            let half = Tensor::<R>::full_scalar(&[1], dtype, -0.5, device);
            let log_det_term = client.mul(&half.broadcast_to(&[n, k])?, &log_det_exp)?;
            let maha_term = client.mul(&half.broadcast_to(&[n, k])?, &maha)?;
            let result = client.add(&const_term.broadcast_to(&[n, k])?, &log_det_term)?;
            client.add(&result, &maha_term)?
        }
        CovarianceType::Full => {
            let log_2pi = (2.0 * std::f64::consts::PI).ln();
            // covariances is [k, d, d]
            // Process per-component for slogdet and Mahalanobis
            let mut log_gauss_slices = Vec::new();
            for j in 0..k {
                let cov_j = covariances.narrow(0, j, 1)?.contiguous().reshape(&[d, d])?;
                let slogdet = client.slogdet(&cov_j)?;
                let log_det_j: f64 = slogdet.logabsdet.item()?;

                let mean_j = means.narrow(0, j, 1)?; // [1, d]
                let diff = client.sub(data, &mean_j.broadcast_to(&[n, d])?)?; // [n, d]
                let inv_cov = client.inverse(&cov_j)?; // [d, d]
                let tmp = client.matmul(&diff, &inv_cov)?; // [n, d]
                let maha = client.sum(&client.mul(&tmp, &diff)?, &[1], false)?; // [n]

                let val = -0.5 * (d as f64 * log_2pi + log_det_j);
                let const_t = Tensor::<R>::full_scalar(&[n], dtype, val, device);
                let half = Tensor::<R>::full_scalar(&[n], dtype, -0.5, device);
                let maha_term = client.mul(&half, &maha)?;
                let lg_j = client.add(&const_t, &maha_term)?; // [n]
                log_gauss_slices.push(lg_j.unsqueeze(1)?); // [n, 1]
            }
            let refs: Vec<&Tensor<R>> = log_gauss_slices.iter().collect();
            client.cat(&refs, 1)? // [n, k]
        }
        CovarianceType::Tied => {
            let log_2pi = (2.0 * std::f64::consts::PI).ln();
            // Single covariance [d, d]
            let slogdet = client.slogdet(covariances)?;
            let log_det: f64 = slogdet.logabsdet.item()?;
            let inv_cov = client.inverse(covariances)?; // [d, d]

            let mut log_gauss_slices = Vec::new();
            for j in 0..k {
                let mean_j = means.narrow(0, j, 1)?;
                let diff = client.sub(data, &mean_j.broadcast_to(&[n, d])?)?;
                let tmp = client.matmul(&diff, &inv_cov)?;
                let maha = client.sum(&client.mul(&tmp, &diff)?, &[1], false)?;

                let val = -0.5 * (d as f64 * log_2pi + log_det);
                let const_t = Tensor::<R>::full_scalar(&[n], dtype, val, device);
                let half = Tensor::<R>::full_scalar(&[n], dtype, -0.5, device);
                let maha_term = client.mul(&half, &maha)?;
                let lg_j = client.add(&const_t, &maha_term)?;
                log_gauss_slices.push(lg_j.unsqueeze(1)?);
            }
            let refs: Vec<&Tensor<R>> = log_gauss_slices.iter().collect();
            client.cat(&refs, 1)?
        }
    };

    // log_resp = log_weights + log_gauss
    client.add(&log_weights_exp, &log_gauss)
}

/// Update covariances in M-step.
#[allow(clippy::too_many_arguments)]
fn update_covariances<R, C>(
    client: &C,
    data: &Tensor<R>,
    resp: &Tensor<R>,
    means: &Tensor<R>,
    nk: &Tensor<R>,
    options: &GmmOptions,
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
            // cov[j, :] = sum(resp[:, j] * (x - mean_j)^2) / nk[j]
            let data_exp = data.unsqueeze(1)?.broadcast_to(&[n, k, d])?;
            let means_exp = means.unsqueeze(0)?.broadcast_to(&[n, k, d])?;
            let diff = client.sub(&data_exp, &means_exp)?;
            let diff_sq = client.mul(&diff, &diff)?;
            let resp_exp = resp.unsqueeze(2)?.broadcast_to(&[n, k, d])?;
            let weighted = client.mul(&resp_exp, &diff_sq)?;
            let sum_weighted = client.sum(&weighted, &[0], false)?; // [k, d]
            let nk_exp = nk.unsqueeze(1)?.broadcast_to(&[k, d])?;
            let cov = client.div(&sum_weighted, &nk_exp)?;
            let reg_t = Tensor::<R>::full_scalar(&[k, d], dtype, reg, device);
            client.add(&cov, &reg_t)
        }
        CovarianceType::Spherical => {
            // cov[j] = mean over d of diagonal variance
            let data_exp = data.unsqueeze(1)?.broadcast_to(&[n, k, d])?;
            let means_exp = means.unsqueeze(0)?.broadcast_to(&[n, k, d])?;
            let diff = client.sub(&data_exp, &means_exp)?;
            let diff_sq = client.mul(&diff, &diff)?;
            let resp_exp = resp.unsqueeze(2)?.broadcast_to(&[n, k, d])?;
            let weighted = client.mul(&resp_exp, &diff_sq)?;
            let sum_weighted = client.sum(&weighted, &[0], false)?; // [k, d]
            let nk_exp = nk.unsqueeze(1)?.broadcast_to(&[k, d])?;
            let cov_diag = client.div(&sum_weighted, &nk_exp)?;
            let cov = client.mean(&cov_diag, &[1], false)?; // [k]
            let reg_t = Tensor::<R>::full_scalar(&[k], dtype, reg, device);
            client.add(&cov, &reg_t)
        }
        CovarianceType::Full => {
            // cov[j] = (resp[:, j] * (x - mean_j))^T @ (x - mean_j) / nk[j]
            let mut cov_slices = Vec::new();
            for j in 0..k {
                let mean_j = means.narrow(0, j, 1)?; // [1, d]
                let diff = client.sub(data, &mean_j.broadcast_to(&[n, d])?)?; // [n, d]
                let resp_j = resp.narrow(1, j, 1)?; // [n, 1]
                let weighted_diff = client.mul(&diff, &resp_j.broadcast_to(&[n, d])?)?; // [n, d]
                let cov_j = client.matmul(&weighted_diff.transpose(0, 1)?, &diff)?; // [d, d]
                let nk_j = nk.narrow(0, j, 1)?; // [1]
                let nk_dd = nk_j.broadcast_to(&[d, d])?;
                let cov_j = client.div(&cov_j, &nk_dd)?;
                // Add regularization to diagonal
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
            // Single covariance = weighted sum over all components
            let data_exp = data.unsqueeze(1)?.broadcast_to(&[n, k, d])?;
            let means_exp = means.unsqueeze(0)?.broadcast_to(&[n, k, d])?;
            let diff = client.sub(&data_exp, &means_exp)?; // [n, k, d]

            let mut total_cov = Tensor::<R>::zeros(&[d, d], dtype, device);
            for j in 0..k {
                let diff_j = diff.narrow(1, j, 1)?.contiguous().reshape(&[n, d])?; // [n, d]
                let resp_j = resp.narrow(1, j, 1)?; // [n, 1]
                let weighted = client.mul(&diff_j, &resp_j.broadcast_to(&[n, d])?)?;
                let cov_j = client.matmul(&weighted.transpose(0, 1)?, &diff_j)?; // [d, d]
                total_cov = client.add(&total_cov, &cov_j)?;
            }
            let n_f = Tensor::<R>::full_scalar(&[d, d], dtype, n as f64, device);
            total_cov = client.div(&total_cov, &n_f)?;
            let reg_eye = client.mul_scalar(
                &client.diagflat(&Tensor::<R>::ones(&[d], dtype, device))?,
                reg,
            )?;
            client.add(&total_cov, &reg_eye)
        }
    }
}

/// Predict most likely component for each point.
pub fn gmm_predict_impl<R, C>(
    client: &C,
    model: &GmmModel<R>,
    data: &Tensor<R>,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: GmmClient<R>,
{
    let resp = gmm_predict_proba_impl(client, model, data)?;
    client.argmax(&resp, 1, false)
}

/// Predict component probabilities [n, k].
pub fn gmm_predict_proba_impl<R, C>(
    client: &C,
    model: &GmmModel<R>,
    data: &Tensor<R>,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: GmmClient<R>,
{
    let n = data.shape()[0];
    let d = data.shape()[1];
    let k = model.weights.shape()[0];
    let dtype = data.dtype();
    let device = data.device();

    // Infer covariance type from shape
    let cov_type = infer_covariance_type(&model.covariances, k, d);

    let options = GmmOptions {
        covariance_type: cov_type,
        ..Default::default()
    };

    let log_resp = compute_log_responsibilities(
        client,
        data,
        &model.weights,
        &model.means,
        &model.covariances,
        &options,
        n,
        d,
        k,
        dtype,
        device,
    )?;

    // Softmax
    let max_log = client.max(&log_resp, &[1], true)?;
    let shifted = client.sub(&log_resp, &max_log)?;
    let exp_shifted = client.exp(&shifted)?;
    let sum_exp = client.sum(&exp_shifted, &[1], true)?;
    client.div(&exp_shifted, &sum_exp)
}

/// Compute per-sample log-likelihood.
pub fn gmm_score_impl<R, C>(client: &C, model: &GmmModel<R>, data: &Tensor<R>) -> Result<Tensor<R>>
where
    R: Runtime,
    C: GmmClient<R>,
{
    let n = data.shape()[0];
    let d = data.shape()[1];
    let k = model.weights.shape()[0];
    let dtype = data.dtype();
    let device = data.device();

    let cov_type = infer_covariance_type(&model.covariances, k, d);
    let options = GmmOptions {
        covariance_type: cov_type,
        ..Default::default()
    };

    let log_resp = compute_log_responsibilities(
        client,
        data,
        &model.weights,
        &model.means,
        &model.covariances,
        &options,
        n,
        d,
        k,
        dtype,
        device,
    )?;

    // logsumexp along components
    let max_log = client.max(&log_resp, &[1], true)?;
    let shifted = client.sub(&log_resp, &max_log)?;
    let exp_shifted = client.exp(&shifted)?;
    let sum_exp = client.sum(&exp_shifted, &[1], true)?;
    let lse = client.add(&client.log(&sum_exp)?, &max_log)?;
    lse.contiguous().reshape(&[n])
}

/// Infer covariance type from tensor shape.
fn infer_covariance_type(covariances: &Tensor<impl Runtime>, k: usize, d: usize) -> CovarianceType {
    let shape = covariances.shape();
    if shape == [k, d, d] {
        CovarianceType::Full
    } else if shape == [d, d] {
        CovarianceType::Tied
    } else if shape == [k, d] {
        CovarianceType::Diagonal
    } else {
        CovarianceType::Spherical
    }
}
