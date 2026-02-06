//! Generic Mean Shift clustering implementation.
//!
//! Batch mode: all points shift simultaneously toward kernel-weighted mean.
//! Uses Gaussian kernel with bandwidth parameter.

use crate::cluster::traits::mean_shift::{MeanShiftOptions, MeanShiftResult};
use crate::cluster::validation::{validate_cluster_dtype, validate_data_2d};
use numr::dtype::DType;
use numr::error::Result;
use numr::ops::{
    CompareOps, ConditionalOps, CumulativeOps, DistanceMetric, DistanceOps, IndexingOps, ReduceOps,
    ScalarOps, ShapeOps, SortingOps, TensorOps, TypeConversionOps, UnaryOps, UtilityOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Generic Mean Shift clustering implementation.
pub fn mean_shift_impl<R, C>(
    client: &C,
    data: &Tensor<R>,
    options: &MeanShiftOptions,
) -> Result<MeanShiftResult<R>>
where
    R: Runtime,
    C: DistanceOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TensorOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + CumulativeOps<R>
        + ShapeOps<R>
        + IndexingOps<R>
        + UnaryOps<R>
        + UtilityOps<R>
        + TypeConversionOps<R>
        + SortingOps<R>
        + RuntimeClient<R>,
{
    validate_cluster_dtype(data.dtype(), "mean_shift")?;
    validate_data_2d(data.shape(), "mean_shift")?;

    let n = data.shape()[0];
    let d = data.shape()[1];
    let dtype = data.dtype();
    let device = data.device();

    // Estimate bandwidth if not provided
    let bandwidth = match options.bandwidth {
        Some(b) => b,
        None => {
            // Simple heuristic: median of pairwise distances / sqrt(d)
            let dists = client.cdist(data, data, DistanceMetric::Euclidean)?;
            let flat = dists.reshape(&[n * n])?;
            let sorted = client.sort(&flat, 0, false)?;
            let mid = n * n / 2;
            let median: f64 = sorted.narrow(0, mid, 1)?.item()?;
            median
        }
    };

    let bw_sq = bandwidth * bandwidth;

    // Initialize seeds = all data points (or bin-seeded subset)
    let mut points = data.clone(); // [n, d] — current positions
    let mut n_iter = 0;

    for iter in 0..options.max_iter {
        n_iter = iter + 1;

        // Compute squared distances from each shifted point to all data points
        let sq_dists = client.cdist(&points, data, DistanceMetric::SquaredEuclidean)?; // [n, n]

        // Gaussian kernel weights: exp(-dist^2 / (2 * bw^2))
        let scale = Tensor::<R>::full_scalar(&[1, 1], dtype, -0.5 / bw_sq, device);
        let scaled = client.mul(&sq_dists, &scale.broadcast_to(&[n, n])?)?;
        let weights = client.exp(&scaled)?; // [n, n]

        // Weighted mean: new_point[i] = sum(weights[i, :] * data) / sum(weights[i, :])
        let weight_sum = client.sum(&weights, &[1], true)?; // [n, 1]
        let weight_sum_safe = client.maximum(
            &weight_sum,
            &Tensor::<R>::full_scalar(&[1, 1], dtype, 1e-32, device),
        )?;

        // weights [n, n] @ data [n, d] = [n, d]
        // But we don't have matmul bound here, so use broadcast multiply + sum
        let weights_exp = weights.unsqueeze(2)?.broadcast_to(&[n, n, d])?; // [n, n, d]
        let data_exp = data.unsqueeze(0)?.broadcast_to(&[n, n, d])?; // [n, n, d]
        let weighted_data = client.mul(&weights_exp, &data_exp)?; // [n, n, d]
        let new_points = client.sum(&weighted_data, &[1], false)?; // [n, d]
        let new_points = client.div(&new_points, &weight_sum_safe.broadcast_to(&[n, d])?)?;

        // Check convergence: max shift distance
        let shift = client.sub(&new_points, &points)?;
        let shift_sq = client.mul(&shift, &shift)?;
        let shift_dist = client.sum(&shift_sq, &[1], false)?; // [n]
        let max_shift: f64 = client.max(&shift_dist, &[0], false)?.item()?;

        points = new_points;

        if max_shift.sqrt() < options.tol {
            break;
        }
    }

    // Merge converged points into cluster centers
    // Points that are within bandwidth of each other belong to the same cluster
    let center_dists = client.cdist(&points, &points, DistanceMetric::SquaredEuclidean)?; // [n, n]
    let threshold = Tensor::<R>::full_scalar(&[n, n], dtype, bw_sq, device);
    let close = client.le(&center_dists, &threshold)?;
    let close_f = client.cast(&close, DType::I64)?;

    // Label propagation to merge close points (same as DBSCAN-style)
    let mut labels = client.arange(0.0, n as f64, 1.0, DType::I64)?;

    for _ in 0..n {
        let labels_row = labels.unsqueeze(0)?.broadcast_to(&[n, n])?;
        let large = Tensor::<R>::full_scalar(&[n, n], DType::I64, n as f64, device);
        let not_close = client.eq(&close_f, &Tensor::<R>::zeros(&[n, n], DType::I64, device))?;
        let masked = client.where_cond(&not_close, &large, &labels_row)?;
        let new_labels = client.min(&masked, &[1], false)?;
        let own_smaller = client.le(&labels, &new_labels)?;
        let new_labels = client.where_cond(&own_smaller, &labels, &new_labels)?;

        let changed = client.ne(&new_labels, &labels)?;
        let changed_f = client.cast(&changed, dtype)?;
        let n_changed: f64 = client.sum(&changed_f, &[0], false)?.item()?;

        labels = new_labels;
        if n_changed == 0.0 {
            break;
        }
    }

    // Remap labels to contiguous 0..k using tensor ops
    let ones_n_f = Tensor::<R>::ones(&[n], dtype, device);

    // Mark which label values are used via scatter_reduce(Max)
    let used = Tensor::<R>::zeros(&[1, n], dtype, device);
    let ones_1n = Tensor::<R>::ones(&[1, n], dtype, device);
    let used = client
        .scatter_reduce(
            &used,
            1,
            &labels.unsqueeze(0)?, // I64 indices
            &ones_1n,
            numr::ops::ScatterReduceOp::Max,
            true,
        )?
        .squeeze(Some(0)); // [n]

    // Cumsum to create contiguous mapping
    let mapping_f = client.sub(&client.cumsum(&used, 0)?, &ones_n_f)?; // [n]
    let mapping = client.cast(&mapping_f, DType::I64)?;

    // Gather new labels from mapping
    let final_labels = client
        .gather(&mapping.unsqueeze(0)?, 1, &labels.unsqueeze(0)?)?
        .squeeze(Some(0)); // [n] I64

    // Count clusters (single scalar transfer)
    let n_clusters: f64 = client.sum(&used, &[0], false)?.item()?;
    let n_clusters = n_clusters as usize;

    // Compute cluster centers as mean of converged points per cluster
    if n_clusters > 0 {
        let labels_exp = final_labels.unsqueeze(1)?.broadcast_to(&[n, d])?;
        let dst = Tensor::<R>::zeros(&[n_clusters, d], dtype, device);
        let sums = client.scatter_reduce(
            &dst,
            0,
            &labels_exp,
            &points,
            numr::ops::ScatterReduceOp::Sum,
            false,
        )?;
        let counts = client.bincount(&final_labels, None, n_clusters)?;
        let counts_f = client.cast(&counts, dtype)?;
        let counts_safe =
            client.maximum(&counts_f, &Tensor::<R>::ones(&[n_clusters], dtype, device))?;
        let counts_exp = counts_safe.unsqueeze(1)?.broadcast_to(&[n_clusters, d])?;
        let cluster_centers = client.div(&sums, &counts_exp)?;

        Ok(MeanShiftResult {
            labels: final_labels,
            cluster_centers,
            n_iter,
        })
    } else {
        Ok(MeanShiftResult {
            labels: final_labels,
            cluster_centers: Tensor::<R>::zeros(&[0, d], dtype, device),
            n_iter,
        })
    }
}
