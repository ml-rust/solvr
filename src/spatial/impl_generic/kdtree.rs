//! Generic KDTree implementation.
//!
//! KDTree construction and query using tensor operations.
//! Data stays on device throughout all operations.

use crate::spatial::traits::kdtree::{KDTree, KDTreeOptions, KNNResult, RadiusResult};
use crate::spatial::{validate_k, validate_points_2d, validate_points_dtype, validate_radius};
use numr::dtype::DType;
use numr::error::{Error, Result};
use numr::ops::{
    CompareOps, CumulativeOps, DistanceOps, IndexingOps, ReduceOps, ScalarOps, SortingOps,
    TensorOps, TypeConversionOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Build a KDTree from points.
///
/// # Algorithm
///
/// Uses tensor operations for all heavy computation:
/// 1. Find split dimension via min/max/argmax tensor ops
/// 2. Sort points along split dimension via argsort
/// 3. Find median and partition using tensor slicing
/// 4. Recurse until leaf size threshold
///
/// Tree structure stored as flat tensors for efficient parallel queries.
pub fn kdtree_build_impl<R, C>(
    client: &C,
    points: &Tensor<R>,
    options: KDTreeOptions,
) -> Result<KDTree<R>>
where
    R: Runtime,
    C: TensorOps<R>
        + ScalarOps<R>
        + ReduceOps<R>
        + SortingOps<R>
        + IndexingOps<R>
        + RuntimeClient<R>,
{
    validate_points_dtype(points.dtype(), "kdtree_build")?;
    validate_points_2d(points.shape(), "kdtree_build")?;

    let n_points = points.shape()[0];
    let n_dims = points.shape()[1];

    if n_points == 0 {
        return Err(Error::InvalidArgument {
            arg: "points",
            reason: "KDTree requires at least 1 point".to_string(),
        });
    }

    let device = points.device();
    let dtype = points.dtype();

    // Initial indices tensor [0, 1, 2, ..., n_points-1]
    let all_indices: Vec<i64> = (0..n_points as i64).collect();
    let initial_indices = Tensor::<R>::from_slice(&all_indices, &[n_points], device);

    // For small point sets, just create a single leaf node
    if n_points <= options.leaf_size {
        let empty_internal = Tensor::<R>::zeros(&[0], dtype, device);

        return Ok(KDTree {
            data: points.clone(),
            split_dims: empty_internal.clone(),
            split_values: empty_internal.clone(),
            left_children: empty_internal.clone(),
            right_children: empty_internal,
            point_indices: initial_indices,
            leaf_starts: Tensor::<R>::from_slice(&[0i64], &[1], device),
            leaf_sizes: Tensor::<R>::from_slice(&[n_points as i64], &[1], device),
            options,
        });
    }

    // Build tree structure using tensor operations
    // We collect metadata as Vecs (small) while keeping point data on device
    let mut split_dims_vec: Vec<i64> = Vec::new();
    let mut split_values_vec: Vec<f64> = Vec::new();
    let mut left_children_vec: Vec<i64> = Vec::new();
    let mut right_children_vec: Vec<i64> = Vec::new();
    let mut leaf_starts_vec: Vec<i64> = Vec::new();
    let mut leaf_sizes_vec: Vec<i64> = Vec::new();
    let mut point_indices_ordered: Vec<i64> = Vec::new();

    // Recursive build using tensor ops for heavy computation
    let mut node_id = 0i64;
    build_node_tensor(
        client,
        points,
        &initial_indices,
        n_dims,
        options.leaf_size,
        &mut node_id,
        &mut split_dims_vec,
        &mut split_values_vec,
        &mut left_children_vec,
        &mut right_children_vec,
        &mut leaf_starts_vec,
        &mut leaf_sizes_vec,
        &mut point_indices_ordered,
    )?;

    // Convert tree metadata to tensors
    let n_nodes = split_dims_vec.len();
    let n_leaves = leaf_starts_vec.len();

    Ok(KDTree {
        data: points.clone(),
        split_dims: Tensor::<R>::from_slice(&split_dims_vec, &[n_nodes], device),
        split_values: Tensor::<R>::from_slice(&split_values_vec, &[n_nodes], device),
        left_children: Tensor::<R>::from_slice(&left_children_vec, &[n_nodes], device),
        right_children: Tensor::<R>::from_slice(&right_children_vec, &[n_nodes], device),
        point_indices: Tensor::<R>::from_slice(
            &point_indices_ordered,
            &[point_indices_ordered.len()],
            device,
        ),
        leaf_starts: Tensor::<R>::from_slice(&leaf_starts_vec, &[n_leaves], device),
        leaf_sizes: Tensor::<R>::from_slice(&leaf_sizes_vec, &[n_leaves], device),
        options,
    })
}

/// Build a node using tensor operations for heavy computation.
/// Only extracts small scalar values (split_dim, split_value) for tree metadata.
#[allow(clippy::too_many_arguments, clippy::only_used_in_recursion)]
fn build_node_tensor<R, C>(
    client: &C,
    points: &Tensor<R>,
    indices: &Tensor<R>,
    n_dims: usize,
    leaf_size: usize,
    node_id: &mut i64,
    split_dims: &mut Vec<i64>,
    split_values: &mut Vec<f64>,
    left_children: &mut Vec<i64>,
    right_children: &mut Vec<i64>,
    leaf_starts: &mut Vec<i64>,
    leaf_sizes: &mut Vec<i64>,
    point_indices: &mut Vec<i64>,
) -> Result<i64>
where
    R: Runtime,
    C: TensorOps<R>
        + ScalarOps<R>
        + ReduceOps<R>
        + SortingOps<R>
        + IndexingOps<R>
        + RuntimeClient<R>,
{
    let n = indices.shape()[0];
    let current_node = *node_id;
    *node_id += 1;
    let device = points.device();

    // Leaf node - extract indices (small transfer for leaf points only)
    if n <= leaf_size {
        split_dims.push(-1); // Mark as leaf
        split_values.push(0.0);
        left_children.push(-1);
        right_children.push(-1);

        leaf_starts.push(point_indices.len() as i64);
        leaf_sizes.push(n as i64);

        // Extract leaf indices (this is unavoidable - we need the final ordering)
        let indices_vec: Vec<i64> = indices.to_vec();
        point_indices.extend(indices_vec);

        return Ok(current_node);
    }

    // Get subset of points for this node using tensor index_select
    let subset_points = client.index_select(points, 0, indices)?;

    // Find dimension with maximum range using tensor ops
    let mins = client.min(&subset_points, &[0], false)?; // [n_dims]
    let maxs = client.max(&subset_points, &[0], false)?; // [n_dims]
    let ranges = client.sub(&maxs, &mins)?; // [n_dims]
    let split_dim_tensor = client.argmax(&ranges, 0, false)?; // scalar

    // Extract split dimension (small scalar transfer)
    let split_dim_vec: Vec<i64> = split_dim_tensor.to_vec();
    let split_dim = split_dim_vec[0] as usize;

    // Get values along split dimension
    let split_col = subset_points.narrow(1, split_dim, 1)?.contiguous();
    let split_col = split_col.reshape(&[n])?;

    // Argsort to get sorted order within subset
    let sorted_order = client.argsort(&split_col, 0, false)?; // [n] indices into subset

    // Map sorted order back to original point indices
    let sorted_indices = client.index_select(indices, 0, &sorted_order)?;

    // Find median value - get the middle element
    let mid = n / 2;
    let mid_idx = Tensor::<R>::from_slice(&[mid as i64], &[1], device);
    let median_pos = client.index_select(&sorted_order, 0, &mid_idx)?;
    let median_value_tensor = client.index_select(&split_col, 0, &median_pos)?;

    // Extract median value (small scalar transfer)
    let split_value_vec: Vec<f64> = median_value_tensor.to_vec();
    let split_value = split_value_vec[0];

    // Record node info
    split_dims.push(split_dim as i64);
    split_values.push(split_value);

    // Placeholder for children indices
    let left_idx = left_children.len();
    let right_idx = right_children.len();
    left_children.push(-1);
    right_children.push(-1);

    // Split indices using tensor narrow (stays on device)
    let left_indices = sorted_indices.narrow(0, 0, mid)?.contiguous();
    let right_indices = sorted_indices.narrow(0, mid, n - mid)?.contiguous();

    // Recurse on children
    let left_child = build_node_tensor(
        client,
        points,
        &left_indices,
        n_dims,
        leaf_size,
        node_id,
        split_dims,
        split_values,
        left_children,
        right_children,
        leaf_starts,
        leaf_sizes,
        point_indices,
    )?;

    let right_child = build_node_tensor(
        client,
        points,
        &right_indices,
        n_dims,
        leaf_size,
        node_id,
        split_dims,
        split_values,
        left_children,
        right_children,
        leaf_starts,
        leaf_sizes,
        point_indices,
    )?;

    left_children[left_idx] = left_child;
    right_children[right_idx] = right_child;

    Ok(current_node)
}

/// Query k nearest neighbors.
///
/// Uses tensor operations throughout - data stays on device.
pub fn kdtree_query_impl<R, C>(
    client: &C,
    tree: &KDTree<R>,
    query: &Tensor<R>,
    k: usize,
) -> Result<KNNResult<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + DistanceOps<R> + SortingOps<R> + RuntimeClient<R>,
{
    validate_points_dtype(query.dtype(), "kdtree_query")?;
    validate_points_2d(query.shape(), "kdtree_query")?;

    let n_tree_points = tree.data.shape()[0];
    validate_k(k, n_tree_points, "kdtree_query")?;

    // Compute all pairwise distances using tensor ops (stays on device)
    let distances = client.cdist(query, &tree.data, tree.options.metric)?;

    // Get k smallest distances and their indices using topk (stays on device)
    let (topk_distances, topk_indices) = client.topk(&distances, k, 1, false, true)?;

    Ok(KNNResult {
        distances: topk_distances,
        indices: topk_indices,
    })
}

/// Query all neighbors within radius.
///
/// Uses tensor operations throughout - data stays on device.
pub fn kdtree_query_radius_impl<R, C>(
    client: &C,
    tree: &KDTree<R>,
    query: &Tensor<R>,
    radius: f64,
) -> Result<RadiusResult<R>>
where
    R: Runtime,
    C: TensorOps<R>
        + ScalarOps<R>
        + DistanceOps<R>
        + ReduceOps<R>
        + CompareOps<R>
        + IndexingOps<R>
        + CumulativeOps<R>
        + TypeConversionOps<R>
        + RuntimeClient<R>,
{
    validate_points_dtype(query.dtype(), "kdtree_query_radius")?;
    validate_points_2d(query.shape(), "kdtree_query_radius")?;
    validate_radius(radius, "kdtree_query_radius")?;

    let n_queries = query.shape()[0];
    let n_points = tree.data.shape()[0];
    let device = query.device();
    let dtype = query.dtype();

    // Compute all pairwise distances using tensor ops
    let all_distances = client.cdist(query, &tree.data, tree.options.metric)?;

    // Create mask for distances within radius
    let threshold = Tensor::<R>::full_scalar(&[], dtype, radius, device);
    let within_radius_raw = client.le(&all_distances, &threshold)?; // [n_queries, n_points]
    // Cast to U8 for boolean mask (comparison ops may return same dtype as input)
    let within_radius = client.cast(&within_radius_raw, DType::U8)?;

    // Count neighbors per query using tensor ops
    let within_radius_f = client.cast(&within_radius, dtype)?;
    let counts_f = client.sum(&within_radius_f, &[1], false)?; // [n_queries]
    let counts = client.cast(&counts_f, DType::I64)?;

    // Compute offsets using cumsum (stays on device)
    let zero = Tensor::<R>::zeros(&[1], DType::I64, device);
    let cumsum = client.cumsum(&counts, 0)?; // [n_queries]

    // Concatenate zero with cumsum to get offsets [0, c0, c0+c1, ...]
    let offsets = client.cat(&[&zero, &cumsum], 0)?; // [n_queries + 1]

    // Flatten distances and mask for masked_select
    let flat_distances = all_distances.reshape(&[n_queries * n_points])?;
    let flat_mask = within_radius.reshape(&[n_queries * n_points])?;

    // Use masked_select to get distances within radius (stays on device)
    let result_distances = client.masked_select(&flat_distances, &flat_mask)?;

    // Create point index tensor [0,1,2,...,n_points-1] repeated for each query
    // Shape: [n_queries, n_points] where each row is [0, 1, 2, ..., n_points-1]
    let point_indices_row: Vec<i64> = (0..n_points as i64).collect();
    let point_indices_1d = Tensor::<R>::from_slice(&point_indices_row, &[n_points], device);
    let point_indices_2d = point_indices_1d
        .unsqueeze(0)?
        .broadcast_to(&[n_queries, n_points])?
        .contiguous();
    let flat_indices = point_indices_2d.reshape(&[n_queries * n_points])?;

    // Use masked_select to get point indices within radius (stays on device)
    let result_indices = client.masked_select(&flat_indices, &flat_mask)?;

    // Handle empty results
    let total_neighbors = result_distances.shape()[0];
    let final_distances = if total_neighbors == 0 {
        Tensor::<R>::zeros(&[1], dtype, device)
    } else {
        result_distances
    };
    let final_indices = if total_neighbors == 0 {
        Tensor::<R>::zeros(&[1], DType::I64, device)
    } else {
        result_indices
    };

    Ok(RadiusResult {
        distances: final_distances,
        indices: final_indices,
        counts,
        offsets,
    })
}
