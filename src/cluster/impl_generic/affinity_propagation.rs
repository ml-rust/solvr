//! Generic Affinity Propagation clustering implementation.
//!
//! Message passing between data points using responsibility and availability matrices.
//! All computation on-device with damping for stability.

use crate::cluster::traits::affinity_propagation::{
    AffinityPropagationOptions, AffinityPropagationResult,
};
use crate::cluster::validation::validate_cluster_dtype;
use numr::dtype::DType;
use numr::error::Result;
use numr::ops::{
    CompareOps, ConditionalOps, DistanceOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps,
    SortingOps, TensorOps, TypeConversionOps, UnaryOps, UtilityOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Generic Affinity Propagation implementation.
pub fn affinity_propagation_impl<R, C>(
    client: &C,
    similarities: &Tensor<R>,
    options: &AffinityPropagationOptions,
) -> Result<AffinityPropagationResult<R>>
where
    R: Runtime,
    C: DistanceOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TensorOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ShapeOps<R>
        + IndexingOps<R>
        + UnaryOps<R>
        + UtilityOps<R>
        + TypeConversionOps<R>
        + SortingOps<R>
        + RuntimeClient<R>,
{
    validate_cluster_dtype(similarities.dtype(), "affinity_propagation")?;

    let n = similarities.shape()[0];
    let dtype = similarities.dtype();
    let device = similarities.device();
    let damping = options.damping;

    // Set preference (diagonal of similarity matrix)
    let pref = match options.preference {
        Some(p) => p,
        None => {
            // Median of similarities
            let flat = similarities.reshape(&[n * n])?;
            let sorted = client.sort(&flat, 0, false)?;
            let mid = n * n / 2;
            let median: f64 = sorted.narrow(0, mid, 1)?.item()?;
            median
        }
    };

    // Set diagonal to preference using mask
    let indices = client.arange(0.0, n as f64, 1.0, DType::I64)?;
    let eye_idx = indices.unsqueeze(1)?.broadcast_to(&[n, n])?;
    let col_idx = client.arange(0.0, n as f64, 1.0, DType::I64)?;
    let col_idx_exp = col_idx.unsqueeze(0)?.broadcast_to(&[n, n])?;
    let is_diag = client.eq(&eye_idx, &col_idx_exp)?;
    let pref_matrix = Tensor::<R>::full_scalar(&[n, n], dtype, pref, device);
    let s = client.where_cond(&is_diag, &pref_matrix, similarities)?;

    // Initialize responsibility and availability matrices
    let mut r = Tensor::<R>::zeros(&[n, n], dtype, device);
    let mut a = Tensor::<R>::zeros(&[n, n], dtype, device);

    let mut n_iter = 0;
    let mut no_change_count = 0;
    let mut prev_exemplar_f: Option<Tensor<R>> = None;

    let damp_t = Tensor::<R>::full_scalar(&[n, n], dtype, damping, device);
    let one_minus_damp_t = Tensor::<R>::full_scalar(&[n, n], dtype, 1.0 - damping, device);
    let zeros_nn = Tensor::<R>::zeros(&[n, n], dtype, device);

    for iter in 0..options.max_iter {
        n_iter = iter + 1;

        // Update responsibilities: r(i,k) = s(i,k) - max_{k' != k}(a(i,k') + s(i,k'))
        let as_sum = client.add(&a, &s)?; // [n, n]

        // For each row, get top-2 values to compute "max excluding k"
        let sorted_row = client.sort(&as_sum, 1, true)?; // descending [n, n]
        let max1 = sorted_row.narrow(1, 0, 1)?; // [n, 1]
        let max2 = sorted_row.narrow(1, 1, 1)?; // [n, 1]

        let is_max = client.eq(&as_sum, &max1.broadcast_to(&[n, n])?)?;
        let exclude_max = client.where_cond(
            &is_max,
            &max2.broadcast_to(&[n, n])?,
            &max1.broadcast_to(&[n, n])?,
        )?;

        let r_new = client.sub(&s, &exclude_max)?;

        // Damping
        r = client.add(
            &client.mul(&damp_t, &r)?,
            &client.mul(&one_minus_damp_t, &r_new)?,
        )?;

        // Update availabilities
        let r_pos = client.maximum(&r, &zeros_nn)?;

        // Sum of positive responsibilities per column (over rows)
        let sum_r_pos = client.sum(&r_pos, &[0], false)?; // [n]

        // Diagonal of r and r_pos
        let idx_gather = indices.unsqueeze(1)?; // [n, 1]
        let r_diag = client.gather(&r, 1, &idx_gather)?.reshape(&[n])?;
        let r_pos_diag = client.gather(&r_pos, 1, &idx_gather)?.reshape(&[n])?;

        // a(i,k) for i != k: min(0, r(k,k) + sum_excluding)
        // sum_excluding = sum_r_pos[k] - r_pos[i,k] - r_pos[k,k]
        let sum_r_pos_exp = sum_r_pos.unsqueeze(0)?.broadcast_to(&[n, n])?;
        let r_pos_diag_exp = r_pos_diag.unsqueeze(0)?.broadcast_to(&[n, n])?;
        let r_diag_exp = r_diag.unsqueeze(0)?.broadcast_to(&[n, n])?;

        let a_raw = client.add(
            &r_diag_exp,
            &client.sub(&client.sub(&sum_r_pos_exp, &r_pos)?, &r_pos_diag_exp)?,
        )?;
        let a_non_diag = client.minimum(&a_raw, &zeros_nn)?;

        // a(k,k) = sum_{i' != k} max(0, r(i',k))
        let a_diag_vals = client.sub(&sum_r_pos, &r_pos_diag)?; // [n]
        let a_diag_exp = a_diag_vals.unsqueeze(0)?.broadcast_to(&[n, n])?;
        let a_new = client.where_cond(&is_diag, &a_diag_exp, &a_non_diag)?;

        // Damping
        a = client.add(
            &client.mul(&damp_t, &a)?,
            &client.mul(&one_minus_damp_t, &a_new)?,
        )?;

        // Check convergence
        let ar_diag = client.add(&client.gather(&a, 1, &idx_gather)?.reshape(&[n])?, &r_diag)?;
        let exemplar_mask = client.gt(&ar_diag, &Tensor::<R>::zeros(&[n], dtype, device))?;
        let exemplar_f = client.cast(&exemplar_mask, dtype)?;

        // Convergence: compare exemplar mask with previous via single scalar transfer
        if let Some(ref prev) = prev_exemplar_f {
            let diff = client.sub(&exemplar_f, prev)?;
            let abs_diff = client.abs(&diff)?;
            let total_diff: f64 = client.sum(&abs_diff, &[0], false)?.item()?;
            if total_diff == 0.0 {
                no_change_count += 1;
                if no_change_count >= options.convergence_iter {
                    break;
                }
            } else {
                no_change_count = 0;
            }
        }
        prev_exemplar_f = Some(exemplar_f);
    }

    // Extract exemplars and assign labels
    let ar = client.add(&a, &r)?;
    let idx_gather = indices.unsqueeze(1)?;
    let ar_diag = client.add(
        &client.gather(&ar, 1, &idx_gather)?.reshape(&[n])?,
        &Tensor::<R>::zeros(&[n], dtype, device), // identity add to ensure shape
    )?;
    let exemplar_mask = client.gt(&ar_diag, &Tensor::<R>::zeros(&[n], dtype, device))?;
    let all_indices = client.arange(0.0, n as f64, 1.0, DType::I64)?;
    let exemplar_mask_u8 = client.cast(&exemplar_mask, DType::U8)?;
    let cluster_centers_indices = client.masked_select(&all_indices, &exemplar_mask_u8)?;

    let n_clusters = cluster_centers_indices.shape()[0];

    if n_clusters == 0 {
        let labels = Tensor::<R>::full_scalar(&[n], DType::I64, -1.0, device);
        return Ok(AffinityPropagationResult {
            labels,
            cluster_centers_indices,
            n_iter,
        });
    }

    // Assign each point to nearest exemplar based on similarity
    let exemplar_rows = client.index_select(&s, 0, &cluster_centers_indices)?; // [k, n]
    let exemplar_cols = exemplar_rows.transpose(0, 1)?; // [n, k]
    let labels = client.argmax(&exemplar_cols, 1, false)?;

    Ok(AffinityPropagationResult {
        labels,
        cluster_centers_indices,
        n_iter,
    })
}
