//! Generic BallTree implementation.
//!
//! BallTree construction and query using tensor operations.
//! Data stays on device throughout all operations.

use crate::spatial::traits::balltree::{BallTree, BallTreeOptions};
use crate::spatial::traits::kdtree::{KNNResult, RadiusResult};
use crate::spatial::{validate_k, validate_points_2d, validate_points_dtype, validate_radius};
use numr::dtype::DType;
use numr::error::{Error, Result};
use numr::ops::{
    CompareOps, CumulativeOps, DistanceOps, IndexingOps, ReduceOps, ScalarOps, SortingOps,
    TensorOps, TypeConversionOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Build a BallTree from points.
///
/// # Algorithm
///
/// Uses tensor operations for all heavy computation:
/// 1. Compute bounding ball using mean (centroid) and max distance
/// 2. Choose split along dimension with largest spread via min/max/argmax
/// 3. Partition points using argsort
/// 4. Recurse until leaf size threshold
pub fn balltree_build_impl<R, C>(
    client: &C,
    points: &Tensor<R>,
    options: BallTreeOptions,
) -> Result<BallTree<R>>
where
    R: Runtime,
    C: TensorOps<R>
        + ScalarOps<R>
        + ReduceOps<R>
        + DistanceOps<R>
        + SortingOps<R>
        + IndexingOps<R>
        + RuntimeClient<R>,
{
    validate_points_dtype(points.dtype(), "balltree_build")?;
    validate_points_2d(points.shape(), "balltree_build")?;

    let n_points = points.shape()[0];
    let n_dims = points.shape()[1];

    if n_points == 0 {
        return Err(Error::InvalidArgument {
            arg: "points",
            reason: "BallTree requires at least 1 point".to_string(),
        });
    }

    let device = points.device();
    let _dtype = points.dtype();

    // Initial indices tensor [0, 1, 2, ..., n_points-1]
    let all_indices: Vec<i64> = (0..n_points as i64).collect();
    let initial_indices = Tensor::<R>::from_slice(&all_indices, &[n_points], device);

    // For small point sets, create single leaf
    if n_points <= options.leaf_size {
        // Compute ball for all points using tensor ops
        let center = client.mean(points, &[0], false)?; // [n_dims]
        let center_broadcast = center.unsqueeze(0)?.broadcast_to(&[n_points, n_dims])?;
        let diffs = client.sub(points, &center_broadcast)?;
        let diffs_sq = client.mul(&diffs, &diffs)?;
        let dist_sq = client.sum(&diffs_sq, &[1], false)?; // [n_points]
        let max_dist_sq = client.max(&dist_sq, &[0], false)?; // scalar
        let radius = client.sqrt(&max_dist_sq)?;

        // Extract center and radius (small transfers)
        let center_vec: Vec<f64> = center.to_vec();
        let radius_vec: Vec<f64> = radius.to_vec();

        return Ok(BallTree {
            data: points.clone(),
            centers: Tensor::<R>::from_slice(&center_vec, &[1, n_dims], device),
            radii: Tensor::<R>::from_slice(&radius_vec, &[1], device),
            left_children: Tensor::<R>::from_slice(&[-1i64], &[1], device),
            right_children: Tensor::<R>::from_slice(&[-1i64], &[1], device),
            point_indices: initial_indices,
            leaf_starts: Tensor::<R>::from_slice(&[0i64], &[1], device),
            leaf_sizes: Tensor::<R>::from_slice(&[n_points as i64], &[1], device),
            options,
        });
    }

    // Build tree structure using tensor operations
    let mut centers_vec: Vec<f64> = Vec::new();
    let mut radii_vec: Vec<f64> = Vec::new();
    let mut left_children_vec: Vec<i64> = Vec::new();
    let mut right_children_vec: Vec<i64> = Vec::new();
    let mut leaf_starts_vec: Vec<i64> = Vec::new();
    let mut leaf_sizes_vec: Vec<i64> = Vec::new();
    let mut point_indices_ordered: Vec<i64> = Vec::new();

    let mut node_id = 0i64;
    build_ball_node_tensor(
        client,
        points,
        &initial_indices,
        n_dims,
        options.leaf_size,
        &mut node_id,
        &mut centers_vec,
        &mut radii_vec,
        &mut left_children_vec,
        &mut right_children_vec,
        &mut leaf_starts_vec,
        &mut leaf_sizes_vec,
        &mut point_indices_ordered,
    )?;

    let n_nodes = radii_vec.len();
    let n_leaves = leaf_starts_vec.len();

    Ok(BallTree {
        data: points.clone(),
        centers: Tensor::<R>::from_slice(&centers_vec, &[n_nodes, n_dims], device),
        radii: Tensor::<R>::from_slice(&radii_vec, &[n_nodes], device),
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

/// Build a ball tree node using tensor operations.
#[allow(clippy::too_many_arguments)]
fn build_ball_node_tensor<R, C>(
    client: &C,
    points: &Tensor<R>,
    indices: &Tensor<R>,
    n_dims: usize,
    leaf_size: usize,
    node_id: &mut i64,
    centers: &mut Vec<f64>,
    radii: &mut Vec<f64>,
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
    let _device = points.device();

    // Get subset of points for this node
    let subset_points = client.index_select(points, 0, indices)?;

    // Compute bounding ball using tensor ops
    let center = client.mean(&subset_points, &[0], false)?; // [n_dims]
    let center_broadcast = center.unsqueeze(0)?.broadcast_to(&[n, n_dims])?;
    let diffs = client.sub(&subset_points, &center_broadcast)?;
    let diffs_sq = client.mul(&diffs, &diffs)?;
    let dist_sq = client.sum(&diffs_sq, &[1], false)?; // [n]
    let max_dist_sq = client.max(&dist_sq, &[0], false)?; // scalar
    let radius_tensor = client.sqrt(&max_dist_sq)?;

    // Extract center and radius (small transfers)
    let center_vec: Vec<f64> = center.to_vec();
    let radius_val: Vec<f64> = radius_tensor.to_vec();
    centers.extend_from_slice(&center_vec);
    radii.push(radius_val[0]);

    // Leaf node
    if n <= leaf_size {
        left_children.push(-1);
        right_children.push(-1);
        leaf_starts.push(point_indices.len() as i64);
        leaf_sizes.push(n as i64);

        // Extract leaf indices
        let indices_vec: Vec<i64> = indices.to_vec();
        point_indices.extend(indices_vec);

        return Ok(current_node);
    }

    // Find dimension with maximum spread using tensor ops
    let mins = client.min(&subset_points, &[0], false)?; // [n_dims]
    let maxs = client.max(&subset_points, &[0], false)?; // [n_dims]
    let spreads = client.sub(&maxs, &mins)?; // [n_dims]
    let best_dim_tensor = client.argmax(&spreads, 0, false)?; // scalar

    // Extract split dimension (small transfer)
    let best_dim_vec: Vec<i64> = best_dim_tensor.to_vec();
    let best_dim = best_dim_vec[0] as usize;

    // Get values along split dimension
    let split_col = subset_points.narrow(1, best_dim, 1)?.contiguous();
    let split_col = split_col.reshape(&[n])?;

    // Argsort to get sorted order
    let sorted_order = client.argsort(&split_col, 0, false)?;

    // Map back to original point indices
    let sorted_indices = client.index_select(indices, 0, &sorted_order)?;

    // Split indices
    let mid = n / 2;
    let left_indices = sorted_indices.narrow(0, 0, mid)?.contiguous();
    let right_indices = sorted_indices.narrow(0, mid, n - mid)?.contiguous();

    // Placeholders for children
    let left_idx = left_children.len();
    let right_idx = right_children.len();
    left_children.push(-1);
    right_children.push(-1);

    // Recurse
    let left_child = build_ball_node_tensor(
        client,
        points,
        &left_indices,
        n_dims,
        leaf_size,
        node_id,
        centers,
        radii,
        left_children,
        right_children,
        leaf_starts,
        leaf_sizes,
        point_indices,
    )?;

    let right_child = build_ball_node_tensor(
        client,
        points,
        &right_indices,
        n_dims,
        leaf_size,
        node_id,
        centers,
        radii,
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

/// Query k nearest neighbors using BallTree.
///
/// Uses tensor operations throughout - data stays on device.
pub fn balltree_query_impl<R, C>(
    client: &C,
    tree: &BallTree<R>,
    query: &Tensor<R>,
    k: usize,
) -> Result<KNNResult<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + DistanceOps<R> + SortingOps<R> + RuntimeClient<R>,
{
    validate_points_dtype(query.dtype(), "balltree_query")?;
    validate_points_2d(query.shape(), "balltree_query")?;

    let n_tree_points = tree.data.shape()[0];
    validate_k(k, n_tree_points, "balltree_query")?;

    // Compute all pairwise distances using tensor ops (stays on device)
    let distances = client.cdist(query, &tree.data, tree.options.metric)?;

    // Get k smallest using topk (stays on device)
    let (topk_distances, topk_indices) = client.topk(&distances, k, 1, false, true)?;

    Ok(KNNResult {
        distances: topk_distances,
        indices: topk_indices,
    })
}

/// Query all neighbors within radius using BallTree.
///
/// Uses tensor operations throughout - data stays on device.
pub fn balltree_query_radius_impl<R, C>(
    client: &C,
    tree: &BallTree<R>,
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
    validate_points_dtype(query.dtype(), "balltree_query_radius")?;
    validate_points_2d(query.shape(), "balltree_query_radius")?;
    validate_radius(radius, "balltree_query_radius")?;

    let n_queries = query.shape()[0];
    let n_points = tree.data.shape()[0];
    let device = query.device();
    let dtype = query.dtype();

    // Compute all pairwise distances using tensor ops
    let all_distances = client.cdist(query, &tree.data, tree.options.metric)?;

    // Create mask for distances within radius
    let threshold = Tensor::<R>::full_scalar(&[], dtype, radius, device);
    let within_radius_raw = client.le(&all_distances, &threshold)?;
    // Cast to U8 for boolean mask (comparison ops may return same dtype as input)
    let within_radius = client.cast(&within_radius_raw, DType::U8)?;

    // Count neighbors per query using tensor ops
    let within_radius_f = client.cast(&within_radius, dtype)?;
    let counts_f = client.sum(&within_radius_f, &[1], false)?;
    let counts = client.cast(&counts_f, DType::I64)?;

    // Compute offsets using cumsum (stays on device)
    let zero = Tensor::<R>::zeros(&[1], DType::I64, device);
    let cumsum = client.cumsum(&counts, 0)?;
    let offsets = client.cat(&[&zero, &cumsum], 0)?;

    // Flatten for masked_select
    let flat_distances = all_distances.reshape(&[n_queries * n_points])?;
    let flat_mask = within_radius.reshape(&[n_queries * n_points])?;

    // Use masked_select to get distances within radius (stays on device)
    let result_distances = client.masked_select(&flat_distances, &flat_mask)?;

    // Create point index tensor and use masked_select
    let point_indices_row: Vec<i64> = (0..n_points as i64).collect();
    let point_indices_1d = Tensor::<R>::from_slice(&point_indices_row, &[n_points], device);
    let point_indices_2d = point_indices_1d
        .unsqueeze(0)?
        .broadcast_to(&[n_queries, n_points])?
        .contiguous();
    let flat_indices = point_indices_2d.reshape(&[n_queries * n_points])?;
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
