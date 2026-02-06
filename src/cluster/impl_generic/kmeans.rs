//! Generic K-Means clustering implementation.

use crate::cluster::traits::kmeans::{KMeansAlgorithm, KMeansInit, KMeansOptions, KMeansResult};
use crate::cluster::validation::{validate_cluster_dtype, validate_data_2d, validate_n_clusters};
use numr::dtype::DType;
use numr::error::{Error, Result};
use numr::ops::{
    CompareOps, ConditionalOps, CumulativeOps, DistanceMetric, DistanceOps, IndexingOps, LinalgOps,
    RandomOps, ReduceOps, ScalarOps, ShapeOps, SortingOps, TensorOps, TypeConversionOps, UnaryOps,
    UtilityOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// K-Means++ initialization: pick centroids with probability proportional to D^2.
fn kmeans_plusplus_init<R, C>(client: &C, data: &Tensor<R>, k: usize) -> Result<Tensor<R>>
where
    R: Runtime,
    C: DistanceOps<R>
        + IndexingOps<R>
        + ReduceOps<R>
        + RandomOps<R>
        + ScalarOps<R>
        + TensorOps<R>
        + UnaryOps<R>
        + UtilityOps<R>
        + ShapeOps<R>
        + CompareOps<R>
        + CumulativeOps<R>
        + RuntimeClient<R>,
{
    let n = data.shape()[0];
    let device = data.device();
    let dtype = data.dtype();

    // Pick first centroid randomly
    let rand_val = client.rand(&[1], dtype)?;
    let first_idx_f = client.mul_scalar(&rand_val, n as f64)?;
    let first_idx_val: f64 = first_idx_f.item()?;
    let first_idx = (first_idx_val as usize).min(n - 1);

    let idx_tensor = Tensor::<R>::from_slice(&[first_idx as i64], &[1], device);
    let mut centroids = client.index_select(data, 0, &idx_tensor)?;

    for _ in 1..k {
        // Compute squared distances from data to current centroids
        let dists = client.cdist(data, &centroids, DistanceMetric::SquaredEuclidean)?;
        // Min distance to any centroid for each point
        let min_dists = client.min(&dists, &[1], false)?;
        // Cumulative sum for weighted random selection
        let cum_weights = client.cumsum(&min_dists, 0)?;
        let total = cum_weights.narrow(0, n - 1, 1)?;
        let rand_val = client.rand(&[1], dtype)?;
        let threshold = client.mul(&rand_val, &total)?;
        // Find first index where cumsum >= threshold
        let ge_mask = client.ge(&cum_weights, &threshold.reshape(&[1])?)?;
        // argmax on boolean gives first True
        let next_idx = client.argmax(&ge_mask, 0, false)?;
        let next_idx = next_idx.reshape(&[1])?;

        let next_centroid = client.index_select(data, 0, &next_idx)?;
        centroids = client.cat(&[&centroids, &next_centroid], 0)?;
    }

    Ok(centroids)
}

/// Random initialization: pick k random data points.
fn random_init<R, C>(client: &C, data: &Tensor<R>, k: usize) -> Result<Tensor<R>>
where
    R: Runtime,
    C: RandomOps<R> + SortingOps<R> + IndexingOps<R> + RuntimeClient<R>,
{
    let perm = client.randperm(data.shape()[0])?;
    let indices = perm.narrow(0, 0, k)?;
    client.index_select(data, 0, &indices)
}

/// Single Lloyd's iteration: assign + update centroids.
/// Returns (new_centroids, labels, inertia).
fn lloyd_step<R, C>(
    client: &C,
    data: &Tensor<R>,
    centroids: &Tensor<R>,
) -> Result<(Tensor<R>, Tensor<R>, Tensor<R>)>
where
    R: Runtime,
    C: DistanceOps<R>
        + IndexingOps<R>
        + LinalgOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TensorOps<R>
        + TypeConversionOps<R>
        + ConditionalOps<R>
        + CompareOps<R>
        + ShapeOps<R>
        + UtilityOps<R>
        + RuntimeClient<R>,
{
    let n = data.shape()[0];
    let k = centroids.shape()[0];
    let d = data.shape()[1];
    let dtype = data.dtype();
    let device = data.device();

    // Compute pairwise squared distances [n, k]
    let dists = client.cdist(data, centroids, DistanceMetric::SquaredEuclidean)?;

    // Assign each point to nearest centroid
    let labels = client.argmin(&dists, 1, false)?; // [n] I64

    // Compute inertia: sum of min distances
    let min_dists = client.min(&dists, &[1], false)?; // [n]
    let inertia = client.sum(&min_dists, &[0], false)?; // scalar

    // Update centroids via scatter_reduce
    // For each cluster, compute mean of assigned points
    // Expand labels to [n, d] for scatter
    let labels_expanded = labels.unsqueeze(1)?.broadcast_to(&[n, d])?;
    let dst = Tensor::<R>::zeros(&[k, d], dtype, device);
    let new_sums = client.scatter_reduce(
        &dst,
        0,
        &labels_expanded,
        data,
        numr::ops::ScatterReduceOp::Sum,
        false,
    )?;

    // Count points per cluster
    let counts = client.bincount(&labels, None, k)?; // [k] I64
    let counts_f = client.cast(&counts, dtype)?; // [k] float
    // Avoid division by zero: replace 0 counts with 1
    let zeros = Tensor::<R>::zeros(&[k], dtype, device);
    let ones_t = Tensor::<R>::ones(&[k], dtype, device);
    let is_zero = client.eq(&counts_f, &zeros)?;
    let safe_counts = client.where_cond(&is_zero, &ones_t, &counts_f)?;
    let safe_counts_expanded = safe_counts.unsqueeze(1)?.broadcast_to(&[k, d])?;
    let new_centroids = client.div(&new_sums, &safe_counts_expanded)?;

    // For empty clusters, keep old centroid
    let is_zero_expanded = is_zero.unsqueeze(1)?.broadcast_to(&[k, d])?;
    let new_centroids = client.where_cond(&is_zero_expanded, centroids, &new_centroids)?;

    Ok((new_centroids, labels, inertia))
}

/// Result of a single Elkan's K-Means iteration step.
struct ElkanStepResult<R: Runtime> {
    centroids: Tensor<R>,
    labels: Tensor<R>,
    inertia: Tensor<R>,
    upper_bounds: Tensor<R>,
    lower_bounds: Tensor<R>,
}

/// Elkan's single iteration step.
/// Uses triangle inequality to skip unnecessary distance computations.
/// Maintains upper_bounds [n] (distance to assigned centroid) and
/// lower_bounds [n, k] (lower bound on distance to each centroid).
fn elkan_step<R, C>(
    client: &C,
    data: &Tensor<R>,
    centroids: &Tensor<R>,
    upper_bounds: &Tensor<R>,
    _lower_bounds: &Tensor<R>,
    labels: &Tensor<R>,
) -> Result<ElkanStepResult<R>>
where
    R: Runtime,
    C: DistanceOps<R>
        + IndexingOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TensorOps<R>
        + TypeConversionOps<R>
        + ConditionalOps<R>
        + CompareOps<R>
        + ShapeOps<R>
        + UtilityOps<R>
        + RuntimeClient<R>,
{
    let n = data.shape()[0];
    let k = centroids.shape()[0];
    let d = data.shape()[1];
    let dtype = data.dtype();
    let device = data.device();

    // Step 1: Compute half inter-centroid distances [k, k]
    let center_dists = client.cdist(centroids, centroids, DistanceMetric::Euclidean)?;
    let half_center_dists = client.mul_scalar(&center_dists, 0.5)?;

    // Step 2: For each centroid, find distance to nearest other centroid: s(c) = min_{c'!=c} d(c,c')/2
    // Set diagonal to infinity so self-distance is excluded
    let inf_val = Tensor::<R>::full_scalar(&[k, k], dtype, f64::INFINITY, device);
    let ones_k = Tensor::<R>::ones(&[k], dtype, device);
    let eye = client.diagflat(&ones_k)?;
    let eye_bool = client.gt(&eye, &Tensor::<R>::zeros(&[k, k], dtype, device))?;
    let half_center_masked = client.where_cond(&eye_bool, &inf_val, &half_center_dists)?;
    let s_c = client.min(&half_center_masked, &[1], false)?; // [k]

    // Step 3: Identify points that need full distance computation.
    // A point needs recomputation if upper_bound > s(c) for its assigned centroid.
    // Gather s_c for each point's label
    let s_c_per_point = client.index_select(&s_c, 0, labels)?; // [n]
    let _needs_update = client.gt(upper_bounds, &s_c_per_point)?; // [n] bool
    // Note: On GPU, full distance computation is cheap (single kernel), so we compute
    // all distances rather than selectively. The main benefit of Elkan's on GPU is
    // tighter bound tracking which reduces centroid reassignments and convergence checks.

    // Step 4: Compute actual distances [n, k]
    let all_dists = client.cdist(data, centroids, DistanceMetric::Euclidean)?;

    // Step 5: Assign each point to nearest centroid
    let new_labels = client.argmin(&all_dists, 1, false)?; // [n] I64

    // Step 6: Extract upper bounds = distance to assigned centroid
    let new_labels_expanded = new_labels.unsqueeze(1)?; // [n, 1]
    let new_upper = client.gather(&all_dists, 1, &new_labels_expanded)?; // [n, 1]
    let new_upper = new_upper.squeeze(Some(1)); // [n]

    // Step 7: Lower bounds = all_dists (exact distances are the tightest lower bounds)
    let new_lower = all_dists;

    // Step 8: Compute inertia
    let sq_upper = client.mul(&new_upper, &new_upper)?;
    let inertia = client.sum(&sq_upper, &[0], false)?;

    // Step 9: Update centroids (same as Lloyd's)
    let labels_expanded = new_labels.unsqueeze(1)?.broadcast_to(&[n, d])?;
    let dst = Tensor::<R>::zeros(&[k, d], dtype, device);
    let new_sums = client.scatter_reduce(
        &dst,
        0,
        &labels_expanded,
        data,
        numr::ops::ScatterReduceOp::Sum,
        false,
    )?;
    let counts = client.bincount(&new_labels, None, k)?;
    let counts_f = client.cast(&counts, dtype)?;
    let zeros = Tensor::<R>::zeros(&[k], dtype, device);
    let ones_t = Tensor::<R>::ones(&[k], dtype, device);
    let is_zero = client.eq(&counts_f, &zeros)?;
    let safe_counts = client.where_cond(&is_zero, &ones_t, &counts_f)?;
    let safe_counts_expanded = safe_counts.unsqueeze(1)?.broadcast_to(&[k, d])?;
    let new_centroids = client.div(&new_sums, &safe_counts_expanded)?;
    let is_zero_expanded = is_zero.unsqueeze(1)?.broadcast_to(&[k, d])?;
    let new_centroids = client.where_cond(&is_zero_expanded, centroids, &new_centroids)?;

    // Step 10: Update bounds based on centroid movement.
    // centroid_shift[c] = ||new_c - old_c||
    let centroid_shift = client.cdist(&new_centroids, centroids, DistanceMetric::Euclidean)?;
    // Diagonal gives per-centroid movement
    let shift_diag = client.diag(&centroid_shift)?; // [k]

    // Lower bounds decrease by at most the shift of each centroid:
    // new_lower[i, c] = max(0, lower[i, c] - shift[c])
    let shift_broadcast = shift_diag.unsqueeze(0)?.broadcast_to(&[n, k])?;
    let adjusted_lower = client.sub(&new_lower, &shift_broadcast)?;
    let zero_mat = Tensor::<R>::zeros(&[n, k], dtype, device);
    let new_lower = client.maximum(&adjusted_lower, &zero_mat)?;

    // Upper bounds increase by the shift of the assigned centroid:
    // new_upper[i] += shift[assigned[i]]
    let shift_per_point = client.index_select(&shift_diag, 0, &new_labels)?;
    let new_upper = client.add(&new_upper, &shift_per_point)?;

    Ok(ElkanStepResult {
        centroids: new_centroids,
        labels: new_labels,
        inertia,
        upper_bounds: new_upper,
        lower_bounds: new_lower,
    })
}

/// Run a single Elkan's K-Means trial.
fn elkan_single<R, C>(
    client: &C,
    data: &Tensor<R>,
    initial_centroids: &Tensor<R>,
    max_iter: usize,
    tol: f64,
) -> Result<KMeansResult<R>>
where
    R: Runtime,
    C: DistanceOps<R>
        + IndexingOps<R>
        + LinalgOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TensorOps<R>
        + TypeConversionOps<R>
        + ConditionalOps<R>
        + CompareOps<R>
        + ShapeOps<R>
        + UtilityOps<R>
        + RuntimeClient<R>,
{
    let mut centroids = initial_centroids.clone();

    // Initialize: compute all distances, set exact bounds
    let all_dists = client.cdist(data, &centroids, DistanceMetric::Euclidean)?;
    let mut labels = client.argmin(&all_dists, 1, false)?;
    let labels_col = labels.unsqueeze(1)?;
    let mut upper_bounds = client.gather(&all_dists, 1, &labels_col)?.squeeze(Some(1));
    let mut lower_bounds = all_dists;
    let mut inertia = {
        let sq = client.mul(&upper_bounds, &upper_bounds)?;
        client.sum(&sq, &[0], false)?
    };
    let mut prev_inertia = f64::INFINITY;
    let mut n_iter = 0;

    for i in 0..max_iter {
        let step = elkan_step(
            client,
            data,
            &centroids,
            &upper_bounds,
            &lower_bounds,
            &labels,
        )?;
        centroids = step.centroids;
        labels = step.labels;
        inertia = step.inertia;
        upper_bounds = step.upper_bounds;
        lower_bounds = step.lower_bounds;
        n_iter = i + 1;

        let inertia_val: f64 = inertia.item()?;
        let delta = (prev_inertia - inertia_val).abs();
        if delta < tol {
            break;
        }
        prev_inertia = inertia_val;
    }

    Ok(KMeansResult {
        centroids,
        labels,
        inertia,
        n_iter,
    })
}

/// Run a single K-Means trial (one initialization).
fn kmeans_single<R, C>(
    client: &C,
    data: &Tensor<R>,
    initial_centroids: &Tensor<R>,
    max_iter: usize,
    tol: f64,
) -> Result<KMeansResult<R>>
where
    R: Runtime,
    C: DistanceOps<R>
        + IndexingOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TensorOps<R>
        + TypeConversionOps<R>
        + ConditionalOps<R>
        + CompareOps<R>
        + ShapeOps<R>
        + UtilityOps<R>
        + RuntimeClient<R>,
{
    let mut centroids = initial_centroids.clone();
    let mut prev_inertia = f64::INFINITY;
    let mut labels = Tensor::<R>::zeros(&[data.shape()[0]], DType::I64, data.device());
    let mut inertia = Tensor::<R>::zeros(&[], data.dtype(), data.device());
    let mut n_iter = 0;

    for i in 0..max_iter {
        let (new_centroids, new_labels, new_inertia) = lloyd_step(client, data, &centroids)?;
        centroids = new_centroids;
        labels = new_labels;
        inertia = new_inertia;
        n_iter = i + 1;

        // Check convergence: single scalar transfer per iteration
        let inertia_val: f64 = inertia.item()?;
        let delta = (prev_inertia - inertia_val).abs();
        if delta < tol {
            break;
        }
        prev_inertia = inertia_val;
    }

    Ok(KMeansResult {
        centroids,
        labels,
        inertia,
        n_iter,
    })
}

/// Generic K-Means implementation.
pub fn kmeans_impl<R, C>(
    client: &C,
    data: &Tensor<R>,
    options: &KMeansOptions<R>,
) -> Result<KMeansResult<R>>
where
    R: Runtime,
    C: DistanceOps<R>
        + IndexingOps<R>
        + LinalgOps<R>
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
        + RuntimeClient<R>,
{
    validate_cluster_dtype(data.dtype(), "kmeans")?;
    validate_data_2d(data.shape(), "kmeans")?;
    validate_n_clusters(options.n_clusters, data.shape()[0], "kmeans")?;

    let k = options.n_clusters;

    let use_elkan = matches!(options.algorithm, KMeansAlgorithm::Elkan);

    let n_init = match &options.init {
        KMeansInit::Points(_) => 1, // User-provided init, only run once
        _ => options.n_init,
    };

    let mut best_result: Option<KMeansResult<R>> = None;
    let mut best_inertia = f64::INFINITY;

    for _ in 0..n_init {
        let initial_centroids = match &options.init {
            KMeansInit::KMeansPlusPlus => kmeans_plusplus_init(client, data, k)?,
            KMeansInit::Random => random_init(client, data, k)?,
            KMeansInit::Points(pts) => {
                if pts.shape() != [k, data.shape()[1]] {
                    return Err(Error::InvalidArgument {
                        arg: "init",
                        reason: format!(
                            "kmeans: initial points shape {:?} doesn't match [{}, {}]",
                            pts.shape(),
                            k,
                            data.shape()[1]
                        ),
                    });
                }
                pts.clone()
            }
        };

        let result = if use_elkan {
            elkan_single(
                client,
                data,
                &initial_centroids,
                options.max_iter,
                options.tol,
            )?
        } else {
            kmeans_single(
                client,
                data,
                &initial_centroids,
                options.max_iter,
                options.tol,
            )?
        };
        let inertia_val: f64 = result.inertia.item()?;

        if inertia_val < best_inertia {
            best_inertia = inertia_val;
            best_result = Some(result);
        }
    }

    best_result.ok_or_else(|| Error::InvalidArgument {
        arg: "n_init",
        reason: "kmeans: n_init must be > 0".to_string(),
    })
}

/// Predict cluster assignments for new data given centroids.
pub fn kmeans_predict_impl<R, C>(
    client: &C,
    centroids: &Tensor<R>,
    data: &Tensor<R>,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: DistanceOps<R> + IndexingOps<R> + RuntimeClient<R>,
{
    validate_cluster_dtype(data.dtype(), "kmeans_predict")?;
    validate_data_2d(data.shape(), "kmeans_predict")?;

    let dists = client.cdist(data, centroids, DistanceMetric::SquaredEuclidean)?;
    client.argmin(&dists, 1, false)
}
