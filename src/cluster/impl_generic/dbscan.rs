//! Generic DBSCAN clustering implementation.
//!
//! Uses distance matrix + label propagation for connected components, all on-device.

use crate::cluster::traits::dbscan::{DbscanOptions, DbscanResult};
use crate::cluster::validation::{validate_cluster_dtype, validate_data_2d};
use numr::dtype::DType;
use numr::error::Result;
use numr::ops::{
    CompareOps, ConditionalOps, CumulativeOps, DistanceOps, IndexingOps, ReduceOps, ScalarOps,
    ShapeOps, TensorOps, TypeConversionOps, UnaryOps, UtilityOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Generic DBSCAN clustering implementation.
pub fn dbscan_impl<R, C>(
    client: &C,
    data: &Tensor<R>,
    options: &DbscanOptions,
) -> Result<DbscanResult<R>>
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
        + RuntimeClient<R>,
{
    validate_cluster_dtype(data.dtype(), "dbscan")?;
    validate_data_2d(data.shape(), "dbscan")?;

    let n = data.shape()[0];
    let dtype = data.dtype();
    let device = data.device();
    let zeros_n = Tensor::<R>::zeros(&[n], dtype, device);
    let ones_n = Tensor::<R>::ones(&[n], dtype, device);

    // 1. Compute pairwise distance matrix [n, n]
    let dists = client.cdist(data, data, options.metric)?;

    // 2. Neighbor mask: dist <= eps → dtype [n, n] (0.0/1.0)
    let eps_t = Tensor::<R>::full_scalar(&[n, n], dtype, options.eps, device);
    let neighbor_mask = client.le(&dists, &eps_t)?; // [n, n] dtype (0/1)

    // 3. Core points: sum(neighbor_mask, dim=1) >= min_samples
    let neighbor_counts = client.sum(&neighbor_mask, &[1], false)?; // [n]
    let min_samples_t = Tensor::<R>::full_scalar(&[n], dtype, options.min_samples as f64, device);
    let is_core = client.ge(&neighbor_counts, &min_samples_t)?; // [n] dtype (0/1)

    // 4. Core-reachable adjacency: neighbor_mask AND (either endpoint is core)
    // Use arithmetic: OR = max, AND = mul
    let is_core_row = is_core.unsqueeze(1)?.broadcast_to(&[n, n])?; // [n, n]
    let is_core_col = is_core.unsqueeze(0)?.broadcast_to(&[n, n])?; // [n, n]
    let either_core = client.maximum(&is_core_row, &is_core_col)?; // OR
    let adjacency = client.mul(&neighbor_mask, &either_core)?; // AND [n, n]

    // 5. Label propagation for connected components
    // Use F64 labels (not I64) so to_vec() works correctly
    let mut labels = client.arange(0.0, n as f64, 1.0, dtype)?; // [n] F64
    let large_val_nn = Tensor::<R>::full_scalar(&[n, n], dtype, (n + 1) as f64, device);

    for _ in 0..n {
        let labels_row = labels.unsqueeze(0)?.broadcast_to(&[n, n])?;
        // Where not adjacent, use large value; where adjacent, use label
        // not_adj = 1 - adjacency (adjacency is 0/1 dtype)
        let adj_nn = adjacency.broadcast_to(&[n, n])?;
        let ones_nn = Tensor::<R>::ones(&[n, n], dtype, device);
        let not_adj = client.sub(&ones_nn, &adj_nn)?;
        // masked = not_adj * large + adj * labels_row
        let masked_large = client.mul(&not_adj, &large_val_nn)?;
        let masked_labels_part = client.mul(&adj_nn, &labels_row)?;
        let masked_labels = client.add(&masked_large, &masked_labels_part)?;
        let new_labels = client.min(&masked_labels, &[1], false)?;

        // Keep own label if smaller
        let own_smaller = client.le(&labels, &new_labels)?; // dtype 0/1
        // merged = own_smaller * labels + (1 - own_smaller) * new_labels
        let not_own = client.sub(&ones_n, &own_smaller)?;
        let new_labels_merged = client.add(
            &client.mul(&own_smaller, &labels)?,
            &client.mul(&not_own, &new_labels)?,
        )?;

        // Check convergence
        let diff = client.sub(&new_labels_merged, &labels)?;
        let abs_diff = client.abs(&diff)?;
        let total_diff: f64 = client.sum(&abs_diff, &[0], false)?.item()?;

        labels = new_labels_merged;
        if total_diff == 0.0 {
            break;
        }
    }

    // 6. Mark non-core, non-reachable points as noise (-1)
    // reachable[j] = any core point i where neighbor_mask[i,j] = 1
    // is_core_row[i,j] = is_core[i], so neighbor_mask * is_core_row gives core neighbors
    let core_neighbor = client.mul(&neighbor_mask, &is_core_row)?;
    let reachable_from_core = client.sum(&core_neighbor, &[0], false)?; // [n]
    let is_reachable = client.gt(&reachable_from_core, &zeros_n)?; // dtype (0/1)

    // noise = NOT core AND NOT reachable = (1 - is_core) * (1 - is_reachable)
    let not_core = client.sub(&ones_n, &is_core)?;
    let not_reachable = client.sub(&ones_n, &is_reachable)?;
    let is_noise = client.mul(&not_core, &not_reachable)?; // dtype (0/1)

    // Set noise labels to -1.0
    let neg_one = Tensor::<R>::full_scalar(&[n], dtype, -1.0, device);
    // noise_labels = is_noise * (-1) + (1 - is_noise) * labels
    let not_noise = client.sub(&ones_n, &is_noise)?;
    let labels_after_noise = client.add(
        &client.mul(&is_noise, &neg_one)?,
        &client.mul(&not_noise, &labels)?,
    )?;

    // 7. Remap labels to contiguous 0..k using tensor ops
    // Clamp noise labels (-1) to 0 for safe indexing
    let safe_labels = client.maximum(&labels_after_noise, &zeros_n)?;
    let safe_labels_i64 = client.cast(&safe_labels, DType::I64)?;

    // Mark which label values are used by non-noise points via scatter_reduce(Max)
    let used = Tensor::<R>::zeros(&[1, n], dtype, device);
    let non_noise_2d = client.sub(&ones_n, &is_noise)?.unsqueeze(0)?; // [1, n]
    let used = client
        .scatter_reduce(
            &used,
            1,
            &safe_labels_i64.unsqueeze(0)?,
            &non_noise_2d,
            numr::ops::ScatterReduceOp::Max,
            true,
        )?
        .squeeze(Some(0)); // [n] — 1.0 where label is used, 0.0 otherwise

    // Cumsum to create contiguous mapping: mapping[i] = (# used labels <= i) - 1
    let mapping = client.sub(&client.cumsum(&used, 0)?, &ones_n)?; // [n]

    // Gather new labels from mapping using original labels as indices
    let new_labels_f = client
        .gather(&mapping.unsqueeze(0)?, 1, &safe_labels_i64.unsqueeze(0)?)?
        .squeeze(Some(0)); // [n]

    // Restore -1 for noise points
    let final_labels = client.where_cond(&is_noise, &neg_one, &new_labels_f)?;

    // Count clusters (single scalar transfer — acceptable)
    let n_clusters: f64 = client.sum(&used, &[0], false)?.item()?;

    // 8. Core sample indices
    let is_core_u8 = client.cast(&is_core, DType::U8)?;
    let all_indices = client.arange(0.0, n as f64, 1.0, DType::I64)?;
    let core_sample_indices = client.masked_select(&all_indices, &is_core_u8)?;

    Ok(DbscanResult {
        labels: final_labels,
        core_sample_indices,
        n_clusters: n_clusters as usize,
    })
}
