//! Generic Mini-Batch K-Means clustering implementation.

use crate::cluster::traits::kmeans::{KMeansOptions, KMeansResult};
use crate::cluster::traits::mini_batch_kmeans::MiniBatchKMeansOptions;
use crate::cluster::validation::{validate_cluster_dtype, validate_data_2d, validate_n_clusters};
use numr::error::Result;
use numr::ops::{
    CompareOps, ConditionalOps, CumulativeOps, DistanceMetric, DistanceOps, IndexingOps, RandomOps,
    ReduceOps, ScalarOps, ShapeOps, SortingOps, TensorOps, TypeConversionOps, UnaryOps, UtilityOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

pub fn mini_batch_kmeans_impl<R, C>(
    client: &C,
    data: &Tensor<R>,
    options: &MiniBatchKMeansOptions<R>,
) -> Result<KMeansResult<R>>
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
        + RuntimeClient<R>,
{
    validate_cluster_dtype(data.dtype(), "mini_batch_kmeans")?;
    validate_data_2d(data.shape(), "mini_batch_kmeans")?;
    validate_n_clusters(options.n_clusters, data.shape()[0], "mini_batch_kmeans")?;

    let n = data.shape()[0];
    let d = data.shape()[1];
    let k = options.n_clusters;
    let dtype = data.dtype();
    let device = data.device();
    let batch_size = options.batch_size.min(n);

    // Initialize centroids via full kmeans init
    let kmeans_opts = KMeansOptions {
        n_clusters: k,
        max_iter: 1,
        tol: 0.0,
        n_init: 1,
        init: options.init.clone(),
        ..Default::default()
    };
    let init_result = super::kmeans::kmeans_impl(client, data, &kmeans_opts)?;
    let mut centroids = init_result.centroids;

    // Per-centroid counts for learning rate
    let mut counts = Tensor::<R>::ones(&[k], dtype, device);

    let mut best_inertia = f64::INFINITY;
    let mut no_improvement = 0usize;
    let mut n_iter = 0;

    for iter in 0..options.max_iter {
        n_iter = iter + 1;

        // Sample mini-batch
        let perm = client.randperm(n)?;
        let batch_idx = perm.narrow(0, 0, batch_size)?;
        let batch = client.index_select(data, 0, &batch_idx)?; // [batch_size, d]

        // Assign batch to nearest centroid
        let dists = client.cdist(&batch, &centroids, DistanceMetric::SquaredEuclidean)?;
        let labels = client.argmin(&dists, 1, false)?; // [batch_size]

        // Update centroids with streaming average
        // For each point in batch: count[c] += 1, eta = 1/count[c], centroid[c] += eta * (x - centroid[c])
        // Batch version: accumulate sums and counts per cluster, then update
        let labels_exp = labels.unsqueeze(1)?.broadcast_to(&[batch_size, d])?;
        let dst = Tensor::<R>::zeros(&[k, d], dtype, device);
        let batch_sums = client.scatter_reduce(
            &dst,
            0,
            &labels_exp,
            &batch,
            numr::ops::ScatterReduceOp::Sum,
            false,
        )?;
        let batch_counts = client.bincount(&labels, None, k)?;
        let batch_counts_f = client.cast(&batch_counts, dtype)?;

        // Update counts
        counts = client.add(&counts, &batch_counts_f)?;

        // Learning rate per cluster: batch_count / total_count
        let eta = client.div(&batch_counts_f, &counts)?; // [k]
        let eta_exp = eta.unsqueeze(1)?.broadcast_to(&[k, d])?;

        // New centroid positions from batch
        let bc_safe = client.maximum(&batch_counts_f, &Tensor::<R>::ones(&[k], dtype, device))?;
        let bc_exp = bc_safe.unsqueeze(1)?.broadcast_to(&[k, d])?;
        let batch_centroids = client.div(&batch_sums, &bc_exp)?;

        // centroid = centroid + eta * (batch_centroid - centroid)
        // Only update clusters that got assignments
        let has_points = client.gt(&batch_counts_f, &Tensor::<R>::zeros(&[k], dtype, device))?;
        let has_points_exp = has_points.unsqueeze(1)?.broadcast_to(&[k, d])?;
        let has_points_f = client.cast(&has_points_exp, dtype)?;

        let diff = client.sub(&batch_centroids, &centroids)?;
        let update = client.mul(&eta_exp, &diff)?;
        let update = client.mul(&update, &has_points_f)?;
        centroids = client.add(&centroids, &update)?;

        // Check convergence via inertia on batch
        if options.tol > 0.0 || options.max_no_improvement < options.max_iter {
            let min_dists = client.min(&dists, &[1], false)?;
            let inertia: f64 = client.mean(&min_dists, &[0], false)?.item()?;

            if inertia < best_inertia - options.tol {
                best_inertia = inertia;
                no_improvement = 0;
            } else {
                no_improvement += 1;
                if no_improvement >= options.max_no_improvement {
                    break;
                }
            }
        }
    }

    // Final assignment on full data
    let final_dists = client.cdist(data, &centroids, DistanceMetric::SquaredEuclidean)?;
    let labels = client.argmin(&final_dists, 1, false)?;
    let min_dists = client.min(&final_dists, &[1], false)?;
    let inertia = client.sum(&min_dists, &[0], false)?;

    Ok(KMeansResult {
        centroids,
        labels,
        inertia,
        n_iter,
    })
}
