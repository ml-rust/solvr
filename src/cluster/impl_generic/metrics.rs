//! Generic cluster evaluation metrics implementation.

use crate::cluster::traits::metrics::HCVScore;
use crate::cluster::validation::{validate_cluster_dtype, validate_data_2d, validate_labels};
use numr::dtype::DType;
use numr::error::Result;
use numr::ops::{
    CompareOps, ConditionalOps, DistanceMetric, DistanceOps, IndexingOps, ReduceOps, ScalarOps,
    ShapeOps, TensorOps, TypeConversionOps, UnaryOps, UtilityOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Compute per-sample silhouette coefficients.
pub fn silhouette_samples_impl<R, C>(
    client: &C,
    data: &Tensor<R>,
    labels: &Tensor<R>,
    metric: DistanceMetric,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: DistanceOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TensorOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ShapeOps<R>
        + UnaryOps<R>
        + UtilityOps<R>
        + IndexingOps<R>
        + TypeConversionOps<R>
        + RuntimeClient<R>,
{
    validate_cluster_dtype(data.dtype(), "silhouette_samples")?;
    validate_data_2d(data.shape(), "silhouette_samples")?;
    validate_labels(labels.shape(), labels.dtype(), "silhouette_samples")?;

    let n = data.shape()[0];
    let dtype = data.dtype();
    let device = data.device();

    // Pairwise distance matrix [n, n]
    let dists = client.cdist(data, data, metric)?;

    // labels as float [n]
    let labels_f = client.cast(labels, dtype)?;

    // same_cluster[i,j] = (labels[i] == labels[j])
    let labels_row = labels_f.unsqueeze(1)?; // [n, 1]
    let labels_col = labels_f.unsqueeze(0)?; // [1, n]
    let same_f = client.eq(&labels_row, &labels_col)?; // [n, n] F64 0/1

    // a_i: mean intra-cluster distance (exclude self)
    // sum of distances to same-cluster points
    let dist_same = client.mul(&dists, &same_f)?; // zero out different cluster
    let sum_same = client.sum(&dist_same, &[1], false)?; // [n]
    // count of same-cluster (including self) minus 1
    let count_same = client.sum(&same_f, &[1], false)?; // [n]
    let count_same_excl = client.sub_scalar(&count_same, 1.0)?;
    let count_safe = client.maximum(&count_same_excl, &Tensor::<R>::ones(&[n], dtype, device))?;
    let a_i = client.div(&sum_same, &count_safe)?; // [n]

    // b_i: min mean distance to any other cluster
    // For each unique cluster k, compute mean distance from each point to cluster k
    // then for points NOT in k, track the minimum such mean
    let max_label: f64 = client.max(&labels_f, &[0], false)?.item()?;
    let n_clusters = (max_label as usize) + 1;

    let inf_tensor = Tensor::<R>::full_scalar(&[n], dtype, f64::INFINITY, device);
    let mut b_i = inf_tensor;

    for k in 0..n_clusters {
        let k_tensor = Tensor::<R>::full_scalar(&[1], dtype, k as f64, device);
        // mask_k[j] = (labels[j] == k) → [1, n] for broadcasting
        let mask_k = client.eq(&labels_col, &k_tensor)?; // [1, n]
        let mask_k_f = client.cast(&mask_k, dtype)?; // [1, n]

        // sum of distances from each point i to points in cluster k
        let dist_to_k = client.mul(&dists, &mask_k_f)?; // [n, n]
        let sum_to_k = client.sum(&dist_to_k, &[1], false)?; // [n]
        let count_k = client.sum(&mask_k_f, &[1], false)?; // [n] (same value repeated)
        let count_k_safe = client.maximum(&count_k, &Tensor::<R>::ones(&[n], dtype, device))?;
        let mean_to_k = client.div(&sum_to_k, &count_k_safe)?; // [n]

        // Only consider for points NOT in cluster k
        let in_k = client.eq(
            &labels_f,
            &Tensor::<R>::full_scalar(&[1], dtype, k as f64, device),
        )?;
        // Set in-cluster points to infinity so they don't affect minimum
        let inf_n = Tensor::<R>::full_scalar(&[n], dtype, f64::INFINITY, device);
        let mean_to_k = client.where_cond(&in_k, &inf_n, &mean_to_k)?;

        b_i = client.minimum(&b_i, &mean_to_k)?;
    }

    // s_i = (b_i - a_i) / max(a_i, b_i)
    let num = client.sub(&b_i, &a_i)?;
    let den = client.maximum(&a_i, &b_i)?;
    let den_safe = client.maximum(&den, &Tensor::<R>::full_scalar(&[n], dtype, 1e-10, device))?;
    client.div(&num, &den_safe)
}

/// Mean silhouette coefficient (scalar).
pub fn silhouette_score_impl<R, C>(
    client: &C,
    data: &Tensor<R>,
    labels: &Tensor<R>,
    metric: DistanceMetric,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: DistanceOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TensorOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ShapeOps<R>
        + UnaryOps<R>
        + UtilityOps<R>
        + IndexingOps<R>
        + TypeConversionOps<R>
        + RuntimeClient<R>,
{
    let samples = silhouette_samples_impl(client, data, labels, metric)?;
    client.mean(&samples, &[0], false)
}

/// Calinski-Harabasz index.
pub fn calinski_harabasz_score_impl<R, C>(
    client: &C,
    data: &Tensor<R>,
    labels: &Tensor<R>,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: DistanceOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TensorOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ShapeOps<R>
        + UnaryOps<R>
        + UtilityOps<R>
        + IndexingOps<R>
        + TypeConversionOps<R>
        + RuntimeClient<R>,
{
    validate_cluster_dtype(data.dtype(), "calinski_harabasz_score")?;
    validate_data_2d(data.shape(), "calinski_harabasz_score")?;
    validate_labels(labels.shape(), labels.dtype(), "calinski_harabasz_score")?;

    let n = data.shape()[0];
    let d = data.shape()[1];
    let dtype = data.dtype();
    let device = data.device();

    let labels_f = client.cast(labels, dtype)?;
    let max_label: f64 = client.max(&labels_f, &[0], false)?.item()?;
    let k = (max_label as usize) + 1;

    if k <= 1 || n <= k {
        return Ok(Tensor::<R>::full_scalar(&[], dtype, 0.0, device));
    }

    // Global centroid [1, d]
    let global_centroid = client.mean(data, &[0], true)?;

    // Compute cluster centroids and sizes via scatter_reduce
    let labels_exp = labels.unsqueeze(1)?.broadcast_to(&[n, d])?;
    let dst = Tensor::<R>::zeros(&[k, d], dtype, device);
    let cluster_sums = client.scatter_reduce(
        &dst,
        0,
        &labels_exp,
        data,
        numr::ops::ScatterReduceOp::Sum,
        false,
    )?;
    let counts = client.bincount(labels, None, k)?;
    let counts_f = client.cast(&counts, dtype)?;
    let counts_safe = client.maximum(&counts_f, &Tensor::<R>::ones(&[k], dtype, device))?;
    let counts_exp = counts_safe.unsqueeze(1)?.broadcast_to(&[k, d])?;
    let centroids = client.div(&cluster_sums, &counts_exp)?; // [k, d]

    // BGD: sum_k(n_k * ||c_k - c_global||^2)
    let diff = client.sub(&centroids, &global_centroid)?; // [k, d]
    let sq_diff = client.mul(&diff, &diff)?;
    let sq_dist_per_cluster = client.sum(&sq_diff, &[1], false)?; // [k]
    let weighted = client.mul(&sq_dist_per_cluster, &counts_f)?;
    let bgd = client.sum(&weighted, &[0], false)?; // scalar

    // WGD: sum of squared distances from each point to its centroid
    // Gather centroids for each point: centroids[labels[i]]
    let point_centroids = client.index_select(&centroids, 0, labels)?; // [n, d]
    let point_diff = client.sub(data, &point_centroids)?;
    let point_sq = client.mul(&point_diff, &point_diff)?;
    let wgd = client.sum(&point_sq, &[0, 1], false)?; // scalar

    // CH = (BGD / (k-1)) / (WGD / (n-k))
    let bgd_norm = client.div_scalar(&bgd, (k - 1) as f64)?;
    let wgd_norm = client.div_scalar(&wgd, (n - k) as f64)?;
    client.div(&bgd_norm, &wgd_norm)
}

/// Davies-Bouldin index.
pub fn davies_bouldin_score_impl<R, C>(
    client: &C,
    data: &Tensor<R>,
    labels: &Tensor<R>,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: DistanceOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TensorOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ShapeOps<R>
        + UnaryOps<R>
        + UtilityOps<R>
        + IndexingOps<R>
        + TypeConversionOps<R>
        + RuntimeClient<R>,
{
    validate_cluster_dtype(data.dtype(), "davies_bouldin_score")?;
    validate_data_2d(data.shape(), "davies_bouldin_score")?;
    validate_labels(labels.shape(), labels.dtype(), "davies_bouldin_score")?;

    let n = data.shape()[0];
    let d = data.shape()[1];
    let dtype = data.dtype();
    let device = data.device();

    let labels_f = client.cast(labels, dtype)?;
    let max_label: f64 = client.max(&labels_f, &[0], false)?.item()?;
    let k = (max_label as usize) + 1;

    if k <= 1 {
        return Ok(Tensor::<R>::full_scalar(&[], dtype, 0.0, device));
    }

    // Compute centroids [k, d]
    let labels_exp = labels.unsqueeze(1)?.broadcast_to(&[n, d])?;
    let dst = Tensor::<R>::zeros(&[k, d], dtype, device);
    let cluster_sums = client.scatter_reduce(
        &dst,
        0,
        &labels_exp,
        data,
        numr::ops::ScatterReduceOp::Sum,
        false,
    )?;
    let counts = client.bincount(labels, None, k)?;
    let counts_f = client.cast(&counts, dtype)?;
    let counts_safe = client.maximum(&counts_f, &Tensor::<R>::ones(&[k], dtype, device))?;
    let counts_exp = counts_safe.unsqueeze(1)?.broadcast_to(&[k, d])?;
    let centroids = client.div(&cluster_sums, &counts_exp)?;

    // Intra-cluster mean distances S_i
    let point_centroids = client.index_select(&centroids, 0, labels)?; // [n, d]
    let point_diff = client.sub(data, &point_centroids)?;
    let point_dist =
        client.sqrt(&client.sum(&client.mul(&point_diff, &point_diff)?, &[1], false)?)?; // [n]

    // Sum distances per cluster, divide by count → S_i
    let dst_s = Tensor::<R>::zeros(&[k], dtype, device);
    let labels_1d = labels.reshape(&[n])?;
    let s_sums = client.scatter_reduce(
        &dst_s,
        0,
        &labels_1d,
        &point_dist,
        numr::ops::ScatterReduceOp::Sum,
        false,
    )?;
    let s_i = client.div(&s_sums, &counts_safe)?; // [k]

    // Inter-centroid distances [k, k]
    let inter_dists = client.cdist(&centroids, &centroids, DistanceMetric::Euclidean)?;

    // R_ij = (S_i + S_j) / M_ij
    let s_row = s_i.unsqueeze(1)?.broadcast_to(&[k, k])?; // [k, k]
    let s_col = s_i.unsqueeze(0)?.broadcast_to(&[k, k])?; // [k, k]
    let s_sum = client.add(&s_row, &s_col)?;

    // Avoid division by zero on diagonal
    let inter_safe = client.add(
        &inter_dists,
        &Tensor::<R>::full_scalar(&[1], dtype, 1e-10, device),
    )?;
    let r_matrix = client.div(&s_sum, &inter_safe)?;

    // Set diagonal to 0 (i != j)
    let idx = client.arange(0.0, k as f64, 1.0, dtype)?;
    let idx_row = idx.unsqueeze(1)?;
    let idx_col = idx.unsqueeze(0)?;
    let diag = client.eq(&idx_row, &idx_col)?;
    let diag_u8 = client.cast(&diag, DType::U8)?;
    let r_matrix = client.masked_fill(&r_matrix, &diag_u8, 0.0)?;

    // DB = mean of max_j(R_ij) for each i
    let max_r = client.max(&r_matrix, &[1], false)?; // [k]
    client.mean(&max_r, &[0], false)
}

/// Build contingency matrix from two label vectors.
/// Returns (contingency [n_true, n_pred], n_true, n_pred).
fn contingency_matrix<R, C>(
    client: &C,
    labels_true: &Tensor<R>,
    labels_pred: &Tensor<R>,
) -> Result<(Tensor<R>, usize, usize)>
where
    R: Runtime,
    C: ReduceOps<R>
        + ScalarOps<R>
        + TensorOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ShapeOps<R>
        + UnaryOps<R>
        + UtilityOps<R>
        + IndexingOps<R>
        + TypeConversionOps<R>
        + RuntimeClient<R>,
{
    let dtype = DType::F64;
    let device = labels_true.device();
    let n = labels_true.shape()[0];

    let true_f = client.cast(labels_true, dtype)?;
    let pred_f = client.cast(labels_pred, dtype)?;
    let max_true: f64 = client.max(&true_f, &[0], false)?.item()?;
    let max_pred: f64 = client.max(&pred_f, &[0], false)?.item()?;
    let n_true = (max_true as usize) + 1;
    let n_pred = (max_pred as usize) + 1;

    // Build contingency: for each (t, p) pair, count occurrences
    // Encode as flat index: flat = labels_true * n_pred + labels_pred
    let true_scaled = client.mul_scalar(&true_f, n_pred as f64)?;
    let flat_idx = client.add(&true_scaled, &pred_f)?;
    let flat_idx_i64 = client.cast(&flat_idx, DType::I64)?;
    let ones = Tensor::<R>::ones(&[n], dtype, device);
    let contingency_flat = client.bincount(&flat_idx_i64, Some(&ones), n_true * n_pred)?;
    let contingency = contingency_flat.reshape(&[n_true, n_pred])?;

    Ok((contingency, n_true, n_pred))
}

/// Adjusted Rand Index.
pub fn adjusted_rand_score_impl<R, C>(
    client: &C,
    labels_true: &Tensor<R>,
    labels_pred: &Tensor<R>,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: ReduceOps<R>
        + ScalarOps<R>
        + TensorOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ShapeOps<R>
        + UnaryOps<R>
        + UtilityOps<R>
        + IndexingOps<R>
        + TypeConversionOps<R>
        + RuntimeClient<R>,
{
    let dtype = DType::F64;
    let device = labels_true.device();

    let (contingency, _n_true, _n_pred) = contingency_matrix(client, labels_true, labels_pred)?;

    // n_ij choose 2 = n_ij * (n_ij - 1) / 2
    let c_minus_1 = client.sub_scalar(&contingency, 1.0)?;
    let comb_all = client.mul(&contingency, &c_minus_1)?;
    let comb_all = client.div_scalar(&comb_all, 2.0)?;
    let sum_comb = client.sum(&comb_all, &[0, 1], false)?; // sum of C(n_ij, 2)

    // Row sums (a_i) and col sums (b_j)
    let a = client.sum(&contingency, &[1], false)?; // [n_true]
    let b = client.sum(&contingency, &[0], false)?; // [n_pred]

    // C(a_i, 2) and C(b_j, 2)
    let a_minus_1 = client.sub_scalar(&a, 1.0)?;
    let comb_a = client.div_scalar(&client.mul(&a, &a_minus_1)?, 2.0)?;
    let sum_comb_a = client.sum(&comb_a, &[0], false)?;

    let b_minus_1 = client.sub_scalar(&b, 1.0)?;
    let comb_b = client.div_scalar(&client.mul(&b, &b_minus_1)?, 2.0)?;
    let sum_comb_b = client.sum(&comb_b, &[0], false)?;

    // n choose 2
    let n_f: f64 = labels_true.shape()[0] as f64;
    let n_comb = n_f * (n_f - 1.0) / 2.0;
    let n_comb_t = Tensor::<R>::full_scalar(&[], dtype, n_comb, device);

    // Expected = sum_comb_a * sum_comb_b / n_comb
    let expected = client.div(&client.mul(&sum_comb_a, &sum_comb_b)?, &n_comb_t)?;

    // max_index = (sum_comb_a + sum_comb_b) / 2
    let max_index = client.div_scalar(&client.add(&sum_comb_a, &sum_comb_b)?, 2.0)?;

    // ARI = (sum_comb - expected) / (max_index - expected)
    let numerator = client.sub(&sum_comb, &expected)?;
    let denominator = client.sub(&max_index, &expected)?;
    // Handle edge case: denominator == 0 → ARI = 0
    let denom_safe = client.maximum(
        &client.abs(&denominator)?,
        &Tensor::<R>::full_scalar(&[], dtype, 1e-15, device),
    )?;
    // Preserve sign
    let sign = client.div(&denominator, &denom_safe)?;
    let denom_with_sign = client.mul(&denom_safe, &sign)?;
    client.div(&numerator, &denom_with_sign)
}

/// Entropy of a label distribution from counts.
fn entropy_from_counts<R, C>(client: &C, counts: &Tensor<R>, n: f64) -> Result<Tensor<R>>
where
    R: Runtime,
    C: ReduceOps<R>
        + ScalarOps<R>
        + TensorOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + UnaryOps<R>
        + RuntimeClient<R>,
{
    let dtype = counts.dtype();
    let device = counts.device();
    let n_t = Tensor::<R>::full_scalar(&[1], dtype, n, device);

    // p = counts / n, H = -sum(p * log(p)) for p > 0
    let p = client.div(counts, &n_t)?;
    let zero = Tensor::<R>::zeros(&[1], dtype, device);
    let mask = client.gt(&p, &zero)?;
    let safe_p = client.maximum(&p, &Tensor::<R>::full_scalar(&[1], dtype, 1e-15, device))?;
    let log_p = client.log(&safe_p)?;
    let p_log_p = client.mul(&p, &log_p)?;
    let masked = client.where_cond(&mask, &p_log_p, &zero)?;
    let neg_sum = client.sum(&masked, &[0], false)?;
    client.mul_scalar(&neg_sum, -1.0)
}

/// Normalized Mutual Information.
pub fn normalized_mutual_info_score_impl<R, C>(
    client: &C,
    labels_true: &Tensor<R>,
    labels_pred: &Tensor<R>,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: ReduceOps<R>
        + ScalarOps<R>
        + TensorOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ShapeOps<R>
        + UnaryOps<R>
        + UtilityOps<R>
        + IndexingOps<R>
        + TypeConversionOps<R>
        + RuntimeClient<R>,
{
    let dtype = DType::F64;
    let device = labels_true.device();
    let n = labels_true.shape()[0] as f64;

    let (contingency, _n_true, _n_pred) = contingency_matrix(client, labels_true, labels_pred)?;

    let a = client.sum(&contingency, &[1], false)?; // [n_true]
    let b = client.sum(&contingency, &[0], false)?; // [n_pred]

    let h_true = entropy_from_counts(client, &a, n)?;
    let h_pred = entropy_from_counts(client, &b, n)?;

    // MI = sum_{ij} (n_ij/n) * log(n * n_ij / (a_i * b_j))
    // Only for n_ij > 0
    let n_t = Tensor::<R>::full_scalar(&[1], dtype, n, device);
    let zero = Tensor::<R>::zeros(&[1], dtype, device);
    let mask = client.gt(&contingency, &zero)?;

    // outer product a_i * b_j
    let a_col = a.unsqueeze(1)?;
    let b_row = b.unsqueeze(0)?;
    let outer = client.mul(&a_col, &b_row)?;

    // n * n_ij / (a_i * b_j)
    let numer = client.mul_scalar(&contingency, n)?;
    let outer_safe = client.maximum(
        &outer,
        &Tensor::<R>::full_scalar(&[1], dtype, 1e-15, device),
    )?;
    let ratio = client.div(&numer, &outer_safe)?;
    let safe_ratio = client.maximum(
        &ratio,
        &Tensor::<R>::full_scalar(&[1], dtype, 1e-15, device),
    )?;
    let log_ratio = client.log(&safe_ratio)?;

    let p_ij = client.div(&contingency, &n_t)?;
    let mi_terms = client.mul(&p_ij, &log_ratio)?;
    let mi_masked = client.where_cond(&mask, &mi_terms, &zero)?;
    let mi = client.sum(&mi_masked, &[0, 1], false)?;

    // NMI = 2 * MI / (H_true + H_pred)
    let h_sum = client.add(&h_true, &h_pred)?;
    let h_safe = client.maximum(&h_sum, &Tensor::<R>::full_scalar(&[], dtype, 1e-15, device))?;
    let nmi = client.div(&client.mul_scalar(&mi, 2.0)?, &h_safe)?;
    Ok(nmi)
}

/// Homogeneity, Completeness, V-Measure.
pub fn homogeneity_completeness_v_measure_impl<R, C>(
    client: &C,
    labels_true: &Tensor<R>,
    labels_pred: &Tensor<R>,
) -> Result<HCVScore<R>>
where
    R: Runtime,
    C: ReduceOps<R>
        + ScalarOps<R>
        + TensorOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ShapeOps<R>
        + UnaryOps<R>
        + UtilityOps<R>
        + IndexingOps<R>
        + TypeConversionOps<R>
        + RuntimeClient<R>,
{
    let dtype = DType::F64;
    let device = labels_true.device();
    let n = labels_true.shape()[0] as f64;

    let (contingency, _n_true, _n_pred) = contingency_matrix(client, labels_true, labels_pred)?;

    let a = client.sum(&contingency, &[1], false)?; // [n_true] = class sizes
    let b = client.sum(&contingency, &[0], false)?; // [n_pred] = cluster sizes

    let h_c = entropy_from_counts(client, &a, n)?; // H(C) entropy of classes
    let h_k = entropy_from_counts(client, &b, n)?; // H(K) entropy of clusters

    // H(C|K) = -sum_{ij} (n_ij/n) * log(n_ij / b_j)
    let n_t = Tensor::<R>::full_scalar(&[1], dtype, n, device);
    let zero = Tensor::<R>::zeros(&[1], dtype, device);
    let mask = client.gt(&contingency, &zero)?;

    let b_row = b.unsqueeze(0)?;
    let b_safe = client.maximum(
        &b_row,
        &Tensor::<R>::full_scalar(&[1], dtype, 1e-15, device),
    )?;
    let ratio_ck = client.div(&contingency, &b_safe)?;
    let safe_ratio_ck = client.maximum(
        &ratio_ck,
        &Tensor::<R>::full_scalar(&[1], dtype, 1e-15, device),
    )?;
    let log_ck = client.log(&safe_ratio_ck)?;
    let p_ij = client.div(&contingency, &n_t)?;
    let terms_ck = client.mul(&p_ij, &log_ck)?;
    let masked_ck = client.where_cond(&mask, &terms_ck, &zero)?;
    let h_c_given_k = client.mul_scalar(&client.sum(&masked_ck, &[0, 1], false)?, -1.0)?;

    // H(K|C) = -sum_{ij} (n_ij/n) * log(n_ij / a_i)
    let a_col = a.unsqueeze(1)?;
    let a_safe = client.maximum(
        &a_col,
        &Tensor::<R>::full_scalar(&[1], dtype, 1e-15, device),
    )?;
    let ratio_kc = client.div(&contingency, &a_safe)?;
    let safe_ratio_kc = client.maximum(
        &ratio_kc,
        &Tensor::<R>::full_scalar(&[1], dtype, 1e-15, device),
    )?;
    let log_kc = client.log(&safe_ratio_kc)?;
    let terms_kc = client.mul(&p_ij, &log_kc)?;
    let masked_kc = client.where_cond(&mask, &terms_kc, &zero)?;
    let h_k_given_c = client.mul_scalar(&client.sum(&masked_kc, &[0, 1], false)?, -1.0)?;

    // Homogeneity = 1 - H(C|K) / H(C)
    let h_c_safe = client.maximum(&h_c, &Tensor::<R>::full_scalar(&[], dtype, 1e-15, device))?;
    let homogeneity = client.sub_scalar(&client.div(&h_c_given_k, &h_c_safe)?, 1.0)?;
    let homogeneity = client.mul_scalar(&homogeneity, -1.0)?; // 1 - x = -(x - 1)

    // Completeness = 1 - H(K|C) / H(K)
    let h_k_safe = client.maximum(&h_k, &Tensor::<R>::full_scalar(&[], dtype, 1e-15, device))?;
    let completeness = client.sub_scalar(&client.div(&h_k_given_c, &h_k_safe)?, 1.0)?;
    let completeness = client.mul_scalar(&completeness, -1.0)?;

    // V-Measure = 2 * H * C / (H + C)
    let hc_sum = client.add(&homogeneity, &completeness)?;
    let hc_safe = client.maximum(
        &hc_sum,
        &Tensor::<R>::full_scalar(&[], dtype, 1e-15, device),
    )?;
    let hc_prod = client.mul(&homogeneity, &completeness)?;
    let v_measure = client.div(&client.mul_scalar(&hc_prod, 2.0)?, &hc_safe)?;

    Ok(HCVScore {
        homogeneity,
        completeness,
        v_measure,
    })
}
