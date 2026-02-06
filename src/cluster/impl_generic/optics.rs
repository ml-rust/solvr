//! Generic OPTICS clustering implementation.
//!
//! Sequential ordering algorithm with on-device distance computation.
//! Each of n iterations extracts one scalar (argmin of reachability among unprocessed).

use crate::cluster::traits::optics::{OpticsOptions, OpticsResult};
use crate::cluster::validation::{validate_cluster_dtype, validate_data_2d};
use numr::dtype::DType;
use numr::error::Result;
use numr::ops::{
    CompareOps, ConditionalOps, DistanceOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps,
    SortingOps, TensorOps, TypeConversionOps, UnaryOps, UtilityOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Generic OPTICS clustering implementation.
pub fn optics_impl<R, C>(
    client: &C,
    data: &Tensor<R>,
    options: &OpticsOptions,
) -> Result<OpticsResult<R>>
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
    validate_cluster_dtype(data.dtype(), "optics")?;
    validate_data_2d(data.shape(), "optics")?;

    let n = data.shape()[0];
    let dtype = data.dtype();
    let device = data.device();
    let inf = f64::INFINITY;

    // 1. Compute full distance matrix [n, n]
    let dists = client.cdist(data, data, options.metric)?;

    // 2. Core distances: distance to min_samples-th nearest neighbor
    // Sort each row, take the min_samples-th element (0-indexed: min_samples - 1)
    let sorted_dists = client.sort(&dists, 1, false)?; // [n, n] sorted ascending per row
    let ms = options.min_samples;
    let core_distances = if ms <= n {
        sorted_dists.narrow(1, ms, 1)?.contiguous().reshape(&[n])? // [n] — ms-th nearest (including self at 0)
    } else {
        Tensor::<R>::full_scalar(&[n], dtype, inf, device)
    };

    // 3. Apply max_eps filter: core_distance > max_eps → infinity
    let max_eps_t = Tensor::<R>::full_scalar(&[n], dtype, options.max_eps, device);
    let exceeds_max = client.gt(&core_distances, &max_eps_t)?;
    let inf_t = Tensor::<R>::full_scalar(&[n], dtype, inf, device);
    let core_distances = client.where_cond(&exceeds_max, &inf_t, &core_distances)?;

    // 4. Sequential ordering
    // reachability[i] = inf initially
    // processed[i] = false
    let mut reachability = Tensor::<R>::full_scalar(&[n], dtype, inf, device);
    let mut processed = Tensor::<R>::zeros(&[n], dtype, device); // 0 = unprocessed
    let ones = Tensor::<R>::ones(&[n], dtype, device);

    let mut ordering_vec = Vec::with_capacity(n);

    for _step in 0..n {
        // Pick next point: unprocessed point with smallest reachability
        // For first point, all reachabilities are inf, so pick any unprocessed (first one)
        let large = Tensor::<R>::full_scalar(&[n], dtype, inf + 1.0, device);
        let proc_bool = client.gt(&processed, &Tensor::<R>::zeros(&[n], dtype, device))?;
        let masked_reach = client.where_cond(&proc_bool, &large, &reachability)?;

        let current_idx: i64 = client
            .argmin(&masked_reach, 0, false)?
            .reshape(&[1])?
            .item()?;
        ordering_vec.push(current_idx);

        // Mark as processed
        let idx_t = Tensor::<R>::from_slice(&[current_idx], &[1], device);
        let one_val = Tensor::<R>::ones(&[1], dtype, device);
        // scatter 1 into processed at current_idx
        let proc_2d = processed.unsqueeze(0)?;
        let idx_2d = idx_t.unsqueeze(0)?;
        let one_2d = one_val.unsqueeze(0)?;
        processed = client
            .scatter(&proc_2d, 1, &idx_2d, &one_2d)?
            .squeeze(Some(0));

        // Get distances from current point to all others: dists[current_idx, :]
        let current_dists = client.index_select(&dists, 0, &idx_t)?.reshape(&[n])?; // [n]

        // Get core distance of current point
        let current_core = client
            .index_select(&core_distances, 0, &idx_t)?
            .reshape(&[1])?;
        let current_core_broadcast = current_core.broadcast_to(&[n])?;

        // new_reachability[j] = max(core_dist[current], dist[current, j])
        let new_reach = client.maximum(&current_core_broadcast, &current_dists)?;

        // Only update unprocessed neighbors within max_eps
        let within_eps = client.le(&current_dists, &max_eps_t)?;
        // NOT processed = 1 - processed (processed is 0/1 F64)
        let not_processed = client.sub(&ones, &processed)?;
        // AND via mul (all values are 0/1)
        let update_mask = client.mul(&within_eps, &not_processed)?;

        // Update reachability: take min of existing and new
        let better = client.lt(&new_reach, &reachability)?;
        let should_update = client.mul(&update_mask, &better)?;
        reachability = client.where_cond(&should_update, &new_reach, &reachability)?;
    }

    // Build ordering tensor
    let ordering = Tensor::<R>::from_slice(&ordering_vec, &[n], device);

    // Reorder reachability to match ordering
    let reachability_ordered = client.index_select(&reachability, 0, &ordering)?;
    let core_distances_ordered = client.index_select(&core_distances, 0, &ordering)?;

    // 5. Xi cluster extraction (if xi provided)
    let labels = if let Some(xi) = options.xi {
        xi_cluster_extraction(
            client,
            &reachability_ordered,
            &ordering,
            n,
            xi,
            dtype,
            device,
        )?
    } else {
        // No cluster extraction, all labels = -1
        Tensor::<R>::full_scalar(&[n], DType::I64, -1.0, device)
    };

    Ok(OpticsResult {
        ordering,
        reachability: reachability_ordered,
        core_distances: core_distances_ordered,
        labels,
    })
}

/// Xi-based cluster extraction from OPTICS reachability plot.
fn xi_cluster_extraction<R, C>(
    _client: &C,
    reachability: &Tensor<R>,
    ordering: &Tensor<R>,
    n: usize,
    xi: f64,
    _dtype: numr::dtype::DType,
    device: &R::Device,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: CompareOps<R>
        + ConditionalOps<R>
        + ScalarOps<R>
        + TensorOps<R>
        + IndexingOps<R>
        + ReduceOps<R>
        + TypeConversionOps<R>
        + UtilityOps<R>
        + RuntimeClient<R>,
{
    // Xi steep-down/steep-up extraction is inherently sequential (each point's
    // classification depends on whether we're inside an open steep-down region).
    // Transfer the O(n) reachability plot for CPU-side processing.
    let reach_vec: Vec<f64> = reachability.to_vec();
    let order_vec: Vec<f64> = ordering.to_vec();

    // Xi steep-down/steep-up detection
    let mut labels_vec = vec![-1i64; n];
    let mut cluster_id = 0i64;

    // Simplified xi extraction: find steep-down areas followed by steep-up areas
    // A point is steep-down if reach[i] * (1 - xi) >= reach[i+1]
    // A point is steep-up if reach[i] <= reach[i+1] * (1 - xi)
    let factor = 1.0 - xi;

    let mut steep_down_start: Option<usize> = None;

    for i in 0..n.saturating_sub(1) {
        let r_curr = reach_vec[i];
        let r_next = reach_vec[i + 1];

        if r_curr.is_infinite() || r_next.is_infinite() {
            // End any open cluster
            if let Some(start) = steep_down_start.take() {
                for &ov in &order_vec[start..=i] {
                    let orig_idx = ov as usize;
                    if orig_idx < n {
                        labels_vec[orig_idx] = cluster_id;
                    }
                }
                cluster_id += 1;
            }
            continue;
        }

        if r_curr * factor >= r_next {
            // Steep down
            if steep_down_start.is_none() {
                steep_down_start = Some(i);
            }
        } else if r_curr <= r_next * factor {
            // Steep up — close cluster if we had a steep down
            if let Some(start) = steep_down_start.take() {
                for &ov in &order_vec[start..=i] {
                    let orig_idx = ov as usize;
                    if orig_idx < n {
                        labels_vec[orig_idx] = cluster_id;
                    }
                }
                cluster_id += 1;
            }
        }
    }

    // Close any remaining open cluster
    if let Some(start) = steep_down_start {
        for &ov in &order_vec[start..n] {
            let orig_idx = ov as usize;
            if orig_idx < n {
                labels_vec[orig_idx] = cluster_id;
            }
        }
    }

    Ok(Tensor::<R>::from_slice(&labels_vec, &[n], device))
}
