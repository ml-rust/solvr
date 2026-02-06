//! Generic hierarchical clustering implementation.
//!
//! Fully on-device: distance matrix, argmin, updates all via tensor ops.
//! The linkage matrix is pre-allocated on device and filled via scatter.

use crate::cluster::traits::hierarchy::{FClusterCriterion, LinkageMatrix, LinkageMethod};
use crate::cluster::validation::validate_cluster_dtype;
use numr::dtype::DType;
use numr::error::{Error, Result};
use numr::ops::{
    CompareOps, ConditionalOps, CumulativeOps, DistanceMetric, DistanceOps, IndexingOps, ReduceOps,
    ScalarOps, ScatterReduceOp, ShapeOps, TensorOps, TypeConversionOps, UnaryOps, UtilityOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Compute linkage from condensed distance vector.
pub fn linkage_impl<R, C>(
    client: &C,
    distances: &Tensor<R>,
    n: usize,
    method: LinkageMethod,
) -> Result<LinkageMatrix<R>>
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
        + RuntimeClient<R>,
{
    let dtype = distances.dtype();
    let device = distances.device();

    // Convert condensed to square distance matrix
    let sq = client.squareform(distances, n)?;
    linkage_from_square(client, &sq, n, method, dtype, device)
}

/// Compute linkage from data points.
pub fn linkage_from_data_impl<R, C>(
    client: &C,
    data: &Tensor<R>,
    method: LinkageMethod,
    metric: DistanceMetric,
) -> Result<LinkageMatrix<R>>
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
        + RuntimeClient<R>,
{
    validate_cluster_dtype(data.dtype(), "linkage")?;

    let n = data.shape()[0];
    let dtype = data.dtype();
    let device = data.device();

    let sq = client.cdist(data, data, metric)?;
    linkage_from_square(client, &sq, n, method, dtype, device)
}

/// Core linkage on a square distance matrix. Fully on-device.
fn linkage_from_square<R, C>(
    client: &C,
    dist_matrix: &Tensor<R>,
    n: usize,
    method: LinkageMethod,
    dtype: DType,
    device: &R::Device,
) -> Result<LinkageMatrix<R>>
where
    R: Runtime,
    C: ReduceOps<R>
        + ScalarOps<R>
        + TensorOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ShapeOps<R>
        + IndexingOps<R>
        + UnaryOps<R>
        + UtilityOps<R>
        + TypeConversionOps<R>
        + RuntimeClient<R>,
{
    if n < 2 {
        return Err(Error::InvalidArgument {
            arg: "n",
            reason: "linkage requires at least 2 points".to_string(),
        });
    }

    // Work with the distance matrix, mask diagonal + inactive with inf
    let mut dists = dist_matrix.clone();
    // Set diagonal to inf
    let idx = client.arange(0.0, n as f64, 1.0, dtype)?;
    let idx_row = idx.unsqueeze(1)?;
    let idx_col = idx.unsqueeze(0)?;
    let diag_mask = client.cast(&client.eq(&idx_row, &idx_col)?, DType::U8)?;
    dists = client.masked_fill(&dists, &diag_mask, f64::INFINITY)?;

    // active[i] = 1.0 if cluster i is still active, 0.0 otherwise
    let mut active = Tensor::<R>::ones(&[n], dtype, device);
    // sizes[i] = number of points in cluster i
    let mut sizes = Tensor::<R>::ones(&[n], dtype, device);

    // Pre-allocate linkage output: we'll collect rows and stack at end
    let mut z_rows: Vec<Tensor<R>> = Vec::with_capacity(n - 1);

    for _step in 0..n - 1 {
        // Mask inactive rows/cols: set to inf
        let active_row = active.unsqueeze(1)?; // [n, 1]
        let active_col = active.unsqueeze(0)?; // [1, n]
        let active_mask = client.mul(&active_row, &active_col)?; // [n, n]
        let inactive = client.cast(
            &client.eq(&active_mask, &Tensor::<R>::zeros(&[1], dtype, device))?,
            DType::U8,
        )?;
        let masked_dists = client.masked_fill(&dists, &inactive, f64::INFINITY)?;
        // Also mask diagonal
        let masked_dists = client.masked_fill(&masked_dists, &diag_mask, f64::INFINITY)?;

        // Find minimum: flatten → argmin
        let flat = masked_dists.reshape(&[n * n])?;
        let flat_argmin = client.argmin(&flat, 0, false)?;
        let flat_argmin = flat_argmin.reshape(&[1])?;

        // Convert flat index to row, col
        let flat_f = client.cast(&flat_argmin, dtype)?;
        let n_f = n as f64;
        let row_f = client.div_scalar(&flat_f, n_f)?;
        // floor
        let row_f = client.sub(&row_f, &Tensor::<R>::full_scalar(&[1], dtype, 0.5, device))?;
        // Use cast to I64 to floor
        let row_i64 = client.cast(&row_f, DType::I64)?;
        let row_f = client.cast(&row_i64, dtype)?;

        let row_times_n = client.mul_scalar(&row_f, n_f)?;
        let col_f = client.sub(&flat_f, &row_times_n)?;
        let col_i64 = client.cast(&col_f, DType::I64)?;

        // Get the merge distance
        let min_dist = client.index_select(&flat, 0, &flat_argmin)?;

        // Get sizes of merging clusters
        let size_i = client.index_select(&sizes, 0, &row_i64)?;
        let size_j = client.index_select(&sizes, 0, &col_i64)?;
        let new_size = client.add(&size_i, &size_j)?;

        // Build linkage row: [min(i,j), max(i,j), dist, new_size]
        // Ensure row < col ordering
        let min_ij = client.minimum(&row_f, &col_f)?;
        let max_ij = client.maximum(&row_f, &col_f)?;

        // Offset cluster IDs: original clusters are 0..n, merged clusters are n..2n-1
        // But linkage convention: IDs in Z can reference both original (0..n) and merged (n, n+1, ...)
        // For step s, the new cluster ID is n + s
        let z_row = client.cat(&[&min_ij, &max_ij, &min_dist, &new_size], 0)?;
        z_rows.push(z_row);

        // Update distance matrix for the surviving cluster (row_i64)
        // Deactivate col cluster
        // Update row cluster with merged distances
        let dist_row_i = client.index_select(&dists, 0, &row_i64)?.reshape(&[n])?;
        let dist_row_j = client.index_select(&dists, 0, &col_i64)?.reshape(&[n])?;

        let new_dists = match method {
            LinkageMethod::Single => client.minimum(&dist_row_i, &dist_row_j)?,
            LinkageMethod::Complete => client.maximum(&dist_row_i, &dist_row_j)?,
            LinkageMethod::Average => {
                let w_i = client.div(&size_i, &new_size)?;
                let w_j = client.div(&size_j, &new_size)?;
                let term_i = client.mul(&dist_row_i, &w_i)?;
                let term_j = client.mul(&dist_row_j, &w_j)?;
                client.add(&term_i, &term_j)?
            }
            LinkageMethod::Weighted => {
                let sum = client.add(&dist_row_i, &dist_row_j)?;
                client.div_scalar(&sum, 2.0)?
            }
            LinkageMethod::Ward => {
                // Lance-Williams: d(ij,k) = sqrt(((n_i+n_k)*d(i,k)^2 + (n_j+n_k)*d(j,k)^2 - n_k*d(i,j)^2) / (n_i+n_j+n_k))
                let all_sizes = sizes.clone();
                let d_ij_sq = client.mul(&min_dist, &min_dist)?;

                let d_ik_sq = client.mul(&dist_row_i, &dist_row_i)?;
                let d_jk_sq = client.mul(&dist_row_j, &dist_row_j)?;

                let si_plus_sk = client.add(&size_i, &all_sizes)?;
                let sj_plus_sk = client.add(&size_j, &all_sizes)?;
                let total = client.add(&new_size, &all_sizes)?;

                let term1 = client.mul(&si_plus_sk, &d_ik_sq)?;
                let term2 = client.mul(&sj_plus_sk, &d_jk_sq)?;
                let term3 = client.mul(&all_sizes, &d_ij_sq)?;

                let numer = client.sub(&client.add(&term1, &term2)?, &term3)?;
                let result = client.div(&numer, &total)?;
                let result = client.maximum(&result, &Tensor::<R>::zeros(&[1], dtype, device))?;
                client.sqrt(&result)?
            }
            LinkageMethod::Centroid | LinkageMethod::Median => {
                // Simplified: use weighted average (same as Average for Centroid)
                let w_i = client.div(&size_i, &new_size)?;
                let w_j = client.div(&size_j, &new_size)?;
                let term_i = client.mul(&dist_row_i, &w_i)?;
                let term_j = client.mul(&dist_row_j, &w_j)?;
                client.add(&term_i, &term_j)?
            }
        };

        // Scatter new distances into row and column of surviving cluster
        // Update row: dists[row, :] = new_dists
        let new_dists_2d = new_dists.unsqueeze(0)?; // [1, n]
        let row_idx_exp = row_i64.unsqueeze(1)?.broadcast_to(&[1, n])?;
        dists = client.scatter(&dists, 0, &row_idx_exp, &new_dists_2d)?;
        // Update column: dists[:, row] = new_dists
        let new_dists_col = new_dists.unsqueeze(1)?; // [n, 1]
        let row_idx_col = row_i64.unsqueeze(0)?.broadcast_to(&[n, 1])?;
        dists = client.scatter(&dists, 1, &row_idx_col, &new_dists_col)?;

        // Deactivate col cluster
        let col_zero = Tensor::<R>::zeros(&[1], dtype, device);
        active = client
            .scatter(
                &active.unsqueeze(0)?,
                1,
                &col_i64.unsqueeze(0)?,
                &col_zero.unsqueeze(0)?,
            )?
            .squeeze(Some(0));

        // Update size of surviving cluster
        sizes = client
            .scatter(
                &sizes.unsqueeze(0)?,
                1,
                &row_i64.unsqueeze(0)?,
                &new_size.unsqueeze(0)?,
            )?
            .squeeze(Some(0));

        // Set diagonal entries to inf for surviving cluster
        dists = client.masked_fill(&dists, &diag_mask, f64::INFINITY)?;
    }

    // Stack all rows into linkage matrix [n-1, 4]
    let z_refs: Vec<&Tensor<R>> = z_rows.iter().collect();
    let z = client.stack(&z_refs, 0)?;

    Ok(LinkageMatrix { z })
}

/// Cut dendrogram to form flat clusters. Fully on-device via label propagation.
pub fn fcluster_impl<R, C>(
    client: &C,
    z: &LinkageMatrix<R>,
    criterion: FClusterCriterion,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: ReduceOps<R>
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
    let n_merges = z.z.shape()[0];
    let n = n_merges + 1;
    let dtype = z.z.dtype();
    let device = z.z.device();

    let threshold = match criterion {
        FClusterCriterion::Distance(t) => t,
        FClusterCriterion::MaxClust(k) => {
            if k >= n {
                return client.arange(0.0, n as f64, 1.0, DType::I64);
            }
            let dists_col = z.z.narrow(1, 2, 1)?.squeeze(Some(1)); // [n-1]
            let idx = n - k;
            if idx == 0 {
                0.0
            } else {
                // Single scalar transfers for threshold computation (acceptable)
                let val: f64 = dists_col.narrow(0, idx - 1, 1)?.item()?;
                if idx < n_merges {
                    let next: f64 = dists_col.narrow(0, idx, 1)?.item()?;
                    (val + next) / 2.0
                } else {
                    val + 1.0
                }
            }
        }
    };

    // On-device label propagation (same pattern as dbscan).
    //
    // Strategy:
    // 1. Build [n, n] adjacency from linkage merges where dist <= threshold.
    //    The linkage stores original matrix indices (0..n-1), so adjacency is [n, n].
    // 2. Label propagation: labels[j] = min(labels[neighbors]) until convergence.
    // 3. Remap to contiguous 0..k.

    // Extract linkage columns
    let id1_f = z.z.narrow(1, 0, 1)?.squeeze(Some(1)); // [n-1]
    let id2_f = z.z.narrow(1, 1, 1)?.squeeze(Some(1)); // [n-1]
    let dists_col = z.z.narrow(1, 2, 1)?.squeeze(Some(1)); // [n-1]

    // Merge mask: which merges are below threshold [n-1] (0/1)
    let thresh_t = Tensor::<R>::full_scalar(&[n_merges], dtype, threshold, device);
    let merge_mask = client.le(&dists_col, &thresh_t)?;

    // Build adjacency [n, n] via scatter: for valid merges, set adj[id1, id2] = 1, adj[id2, id1] = 1
    let id1_i64 = client.cast(&id1_f, DType::I64)?;
    let id2_i64 = client.cast(&id2_f, DType::I64)?;

    let mut adjacency = Tensor::<R>::zeros(&[n, n], dtype, device);

    // Scatter merge_mask into adjacency at (id1, id2) and (id2, id1)
    // For each merge i: adj[id1[i], id2[i]] = merge_mask[i]
    // We need 2D indices. Use id1 as row selector, id2 as column within that row.
    // scatter along dim=1: adj[id1[i], id2[i]] = merge_mask[i]
    // First, select rows by id1, then scatter into columns by id2
    // Simpler: build [n_merges, n] sparse rows, then scatter_reduce into [n, n]

    // Approach: create [1, n_merges] index tensors and scatter into [n, n]
    // adj_sparse[i, id2[i]] = merge_mask[i] for each merge i (row = merge index)
    let merge_row = Tensor::<R>::zeros(&[n_merges, n], dtype, device);
    let merge_row = client.scatter(
        &merge_row,
        1,
        &id2_i64.unsqueeze(1)?,
        &merge_mask.unsqueeze(1)?,
    )?; // [n_merges, n]: row i has merge_mask[i] at column id2[i]

    // Now scatter_reduce these rows into adjacency using id1 as row indices
    // adj[id1[i], :] = max(adj[id1[i], :], merge_row[i, :])
    let id1_exp = id1_i64.unsqueeze(1)?.broadcast_to(&[n_merges, n])?;
    adjacency = client.scatter_reduce(
        &adjacency,
        0,
        &id1_exp,
        &merge_row,
        ScatterReduceOp::Max,
        true,
    )?;

    // Symmetrize: adj[id2, id1] too
    let merge_col = Tensor::<R>::zeros(&[n_merges, n], dtype, device);
    let merge_col = client.scatter(
        &merge_col,
        1,
        &id1_i64.unsqueeze(1)?,
        &merge_mask.unsqueeze(1)?,
    )?;
    let id2_exp = id2_i64.unsqueeze(1)?.broadcast_to(&[n_merges, n])?;
    adjacency = client.scatter_reduce(
        &adjacency,
        0,
        &id2_exp,
        &merge_col,
        ScatterReduceOp::Max,
        true,
    )?;

    // Add self-connections (each point is adjacent to itself)
    let eye_mask = client.cast(
        &client.eq(
            &client.arange(0.0, n as f64, 1.0, dtype)?.unsqueeze(1)?,
            &client.arange(0.0, n as f64, 1.0, dtype)?.unsqueeze(0)?,
        )?,
        dtype,
    )?;
    adjacency = client.maximum(&adjacency, &eye_mask)?;

    // Label propagation: find connected components
    let ones_n = Tensor::<R>::ones(&[n], dtype, device);
    let large_val = Tensor::<R>::full_scalar(&[n, n], dtype, (n + 1) as f64, device);
    let mut labels = client.arange(0.0, n as f64, 1.0, dtype)?;

    for _ in 0..n {
        let labels_row = labels.unsqueeze(0)?.broadcast_to(&[n, n])?;
        // Where not adjacent, use large value; where adjacent, use label
        let not_adj = client.sub(&Tensor::<R>::ones(&[n, n], dtype, device), &adjacency)?;
        let masked = client.add(
            &client.mul(&not_adj, &large_val)?,
            &client.mul(&adjacency, &labels_row)?,
        )?;
        let new_labels = client.min(&masked, &[1], false)?;

        // Keep own label if smaller
        let own_smaller = client.le(&labels, &new_labels)?;
        let not_own = client.sub(&ones_n, &own_smaller)?;
        let merged = client.add(
            &client.mul(&own_smaller, &labels)?,
            &client.mul(&not_own, &new_labels)?,
        )?;

        // Convergence check
        let diff = client.sub(&merged, &labels)?;
        let abs_diff = client.abs(&diff)?;
        let total_diff: f64 = client.sum(&abs_diff, &[0], false)?.item()?;

        labels = merged;
        if total_diff == 0.0 {
            break;
        }
    }

    // Remap to contiguous 0..k using scatter_reduce + cumsum + gather
    let labels_i64 = client.cast(&labels, DType::I64)?;

    let used = Tensor::<R>::zeros(&[1, n], dtype, device);
    let used = client
        .scatter_reduce(
            &used,
            1,
            &labels_i64.unsqueeze(0)?,
            &ones_n.unsqueeze(0)?,
            ScatterReduceOp::Max,
            true,
        )?
        .squeeze(Some(0)); // [n]

    let mapping = client.sub(&client.cumsum(&used, 0)?, &ones_n)?; // [n]

    let new_labels = client
        .gather(&mapping.unsqueeze(0)?, 1, &labels_i64.unsqueeze(0)?)?
        .squeeze(Some(0)); // [n]

    Ok(new_labels)
}

/// Cluster data directly (linkage + fcluster).
pub fn fclusterdata_impl<R, C>(
    client: &C,
    data: &Tensor<R>,
    criterion: FClusterCriterion,
    method: LinkageMethod,
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
        + CumulativeOps<R>
        + ShapeOps<R>
        + IndexingOps<R>
        + UnaryOps<R>
        + UtilityOps<R>
        + TypeConversionOps<R>
        + RuntimeClient<R>,
{
    let z = linkage_from_data_impl(client, data, method, metric)?;
    fcluster_impl(client, &z, criterion)
}

/// Get leaves in dendrogram order.
pub fn leaves_list_impl<R, C>(_client: &C, z: &LinkageMatrix<R>) -> Result<Tensor<R>>
where
    R: Runtime,
    C: RuntimeClient<R>,
{
    let n_merges = z.z.shape()[0];
    let n = n_merges + 1;
    let device = z.z.device();
    let z_data: Vec<f64> = z.z.to_vec();

    // DFS traversal: inherently sequential, operates on O(n) linkage rows.
    let mut order = Vec::with_capacity(n);
    let mut stack = vec![2 * n - 2]; // root node

    while let Some(node) = stack.pop() {
        if node < n {
            order.push(node as i64);
        } else {
            let merge_idx = node - n;
            let right = z_data[merge_idx * 4 + 1] as usize;
            let left = z_data[merge_idx * 4] as usize;
            stack.push(right);
            stack.push(left);
        }
    }

    Ok(Tensor::<R>::from_slice(&order, &[n], device))
}

/// Cut tree at multiple cluster counts.
/// Returns [n, len(n_clusters)] I64 tensor.
pub fn cut_tree_impl<R, C>(
    client: &C,
    z: &LinkageMatrix<R>,
    n_clusters: &[usize],
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: ReduceOps<R>
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
    // Compute labels for each cluster count, stack, and transpose on-device
    let mut label_tensors: Vec<Tensor<R>> = Vec::with_capacity(n_clusters.len());
    for &k in n_clusters {
        let labels = fcluster_impl(client, z, FClusterCriterion::MaxClust(k))?;
        label_tensors.push(labels.unsqueeze(1)?); // [n, 1]
    }
    let refs: Vec<&Tensor<R>> = label_tensors.iter().collect();
    client.cat(&refs, 1) // [n, m]
}
