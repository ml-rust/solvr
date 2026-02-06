//! Generic HDBSCAN clustering implementation.
//!
//! Steps: core distances → mutual reachability → MST (Prim's on device) →
//! condensed tree → cluster extraction (EOM/Leaf).

use crate::cluster::traits::hdbscan::{ClusterSelectionMethod, HdbscanOptions, HdbscanResult};
use crate::cluster::validation::{validate_cluster_dtype, validate_data_2d};
use numr::dtype::DType;
use numr::error::Result;
use numr::ops::{
    CompareOps, ConditionalOps, DistanceOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps,
    SortingOps, TensorOps, TypeConversionOps, UnaryOps, UtilityOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Generic HDBSCAN clustering implementation.
pub fn hdbscan_impl<R, C>(
    client: &C,
    data: &Tensor<R>,
    options: &HdbscanOptions,
) -> Result<HdbscanResult<R>>
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
    validate_cluster_dtype(data.dtype(), "hdbscan")?;
    validate_data_2d(data.shape(), "hdbscan")?;

    let n = data.shape()[0];
    let dtype = data.dtype();
    let device = data.device();
    let min_samples = options.min_samples.unwrap_or(options.min_cluster_size);

    // 1. Compute distance matrix [n, n]
    let dists = client.cdist(data, data, options.metric)?;

    // 2. Core distances: k-th nearest neighbor distance (k = min_samples)
    // Sort each row, take min_samples-th element
    let sorted = client.sort(&dists, 1, false)?; // [n, n]
    let core_distances = if min_samples < n {
        sorted
            .narrow(1, min_samples, 1)?
            .contiguous()
            .reshape(&[n])?
    } else {
        Tensor::<R>::full_scalar(&[n], dtype, f64::INFINITY, device)
    };

    // 3. Mutual reachability distance: max(core[i], core[j], dist[i,j])
    let core_row = core_distances.unsqueeze(1)?.broadcast_to(&[n, n])?; // [n, n]
    let core_col = core_distances.unsqueeze(0)?.broadcast_to(&[n, n])?; // [n, n]
    let mr_dist = client.maximum(&dists, &core_row)?;
    let mr_dist = client.maximum(&mr_dist, &core_col)?;

    // 4. MST via Prim's algorithm on mutual reachability matrix
    // Sequential: n-1 iterations, 1 scalar extraction per iter
    let inf = f64::INFINITY;
    let mut in_mst = Tensor::<R>::zeros(&[n], dtype, device); // 0 = not in MST
    let mut min_cost = Tensor::<R>::full_scalar(&[n], dtype, inf, device);

    // Start from node 0
    let zero_idx = Tensor::<R>::from_slice(&[0i64], &[1], device);
    let one_val = Tensor::<R>::ones(&[1], dtype, device);
    let in_mst_2d = in_mst.unsqueeze(0)?;
    let zero_idx_2d = zero_idx.unsqueeze(0)?;
    let one_2d = one_val.unsqueeze(0)?;
    in_mst = client
        .scatter(&in_mst_2d, 1, &zero_idx_2d, &one_2d)?
        .squeeze(Some(0));

    // Update min_cost from node 0's row
    let row0 = client.index_select(&mr_dist, 0, &zero_idx)?.reshape(&[n])?;
    min_cost = client.minimum(&min_cost, &row0)?;

    // MST edges: (from, to, weight) — collect on CPU
    let mut mst_from = Vec::with_capacity(n - 1);
    let mut mst_to = Vec::with_capacity(n - 1);
    let mut mst_weight = Vec::with_capacity(n - 1);

    // Track which node connected each point (for MST edge recording)
    // parent[j] = the MST node that gave j its current min_cost
    let mut parent = Tensor::<R>::zeros(&[n], DType::I64, device);

    for _ in 0..(n - 1) {
        // Mask out nodes already in MST
        let in_mst_bool = client.gt(&in_mst, &Tensor::<R>::zeros(&[n], dtype, device))?;
        let large = Tensor::<R>::full_scalar(&[n], dtype, inf + 1.0, device);
        let masked_cost = client.where_cond(&in_mst_bool, &large, &min_cost)?;

        // Find cheapest edge to add
        let next_idx: i64 = client
            .argmin(&masked_cost, 0, false)?
            .reshape(&[1])?
            .item()?;
        let next_weight: f64 = min_cost.narrow(0, next_idx as usize, 1)?.item()?;
        let parent_idx: i64 = parent.narrow(0, next_idx as usize, 1)?.item()?;

        mst_from.push(parent_idx);
        mst_to.push(next_idx);
        mst_weight.push(next_weight);

        // Add to MST
        let next_t = Tensor::<R>::from_slice(&[next_idx], &[1], device);
        let in_mst_2d = in_mst.unsqueeze(0)?;
        let next_2d = next_t.unsqueeze(0)?;
        let one_2d = Tensor::<R>::ones(&[1], dtype, device).unsqueeze(0)?;
        in_mst = client
            .scatter(&in_mst_2d, 1, &next_2d, &one_2d)?
            .squeeze(Some(0));

        // Update min_cost from new node's row
        let new_row = client.index_select(&mr_dist, 0, &next_t)?.reshape(&[n])?;
        let improved = client.lt(&new_row, &min_cost)?;
        min_cost = client.where_cond(&improved, &new_row, &min_cost)?;

        // Update parent for improved nodes
        let next_i64 = Tensor::<R>::full_scalar(&[n], DType::I64, next_idx as f64, device);
        parent = client.where_cond(&improved, &next_i64, &parent)?;
    }

    // 5. Build condensed tree and extract clusters (CPU — operates on n-1 MST edges)
    let min_cluster_size = options.min_cluster_size;
    let (labels_vec, probabilities_vec, persistence_vec) = extract_clusters(
        n,
        &mst_from,
        &mst_to,
        &mst_weight,
        min_cluster_size,
        options.cluster_selection_method,
        options.allow_single_cluster,
    );

    let labels = Tensor::<R>::from_slice(&labels_vec, &[n], device);
    let probabilities_f: Vec<f64> = probabilities_vec.iter().map(|&x| x as f64).collect();
    let probabilities = Tensor::<R>::from_slice(&probabilities_f, &[n], device);
    let n_clusters = persistence_vec.len();
    let persistence_f: Vec<f64> = persistence_vec.iter().map(|&x| x as f64).collect();
    let cluster_persistence = if n_clusters > 0 {
        Tensor::<R>::from_slice(&persistence_f, &[n_clusters], device)
    } else {
        Tensor::<R>::zeros(&[0], dtype, device)
    };

    Ok(HdbscanResult {
        labels,
        probabilities,
        cluster_persistence,
    })
}

/// Extract clusters from MST using single-linkage hierarchy + EOM/Leaf.
fn extract_clusters(
    n: usize,
    mst_from: &[i64],
    mst_to: &[i64],
    mst_weight: &[f64],
    min_cluster_size: usize,
    method: ClusterSelectionMethod,
    allow_single_cluster: bool,
) -> (Vec<i64>, Vec<f32>, Vec<f32>) {
    // Sort MST edges by weight (ascending) for single-linkage dendrogram
    let mut edge_order: Vec<usize> = (0..mst_from.len()).collect();
    edge_order.sort_by(|&a, &b| {
        mst_weight[a]
            .partial_cmp(&mst_weight[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Union-Find for building hierarchy
    let mut uf_parent: Vec<usize> = (0..2 * n).collect();
    let mut uf_size: Vec<usize> = vec![1; 2 * n];
    let mut next_cluster = n;

    // Condensed tree: (parent_cluster, child, lambda, child_size)
    let mut condensed: Vec<(usize, usize, f64, usize)> = Vec::new();

    fn find(uf: &mut [usize], x: usize) -> usize {
        let mut root = x;
        while uf[root] != root {
            root = uf[root];
        }
        let mut cur = x;
        while uf[cur] != root {
            let next = uf[cur];
            uf[cur] = root;
            cur = next;
        }
        root
    }

    for &ei in &edge_order {
        let a = find(&mut uf_parent, mst_from[ei] as usize);
        let b = find(&mut uf_parent, mst_to[ei] as usize);
        if a == b {
            continue;
        }

        let w = mst_weight[ei];
        let lambda = if w > 0.0 { 1.0 / w } else { f64::INFINITY };
        let new_cluster = next_cluster;
        next_cluster += 1;
        if next_cluster > 2 * n - 1 {
            break;
        }

        let size_a = uf_size[a];
        let size_b = uf_size[b];

        // Add to condensed tree
        if size_a >= min_cluster_size && size_b >= min_cluster_size {
            // Both large enough → both become children of new cluster
            condensed.push((new_cluster, a, lambda, size_a));
            condensed.push((new_cluster, b, lambda, size_b));
        } else if size_a >= min_cluster_size {
            // Only a is big → b's points fall out as noise from a
            condensed.push((new_cluster, a, lambda, size_a));
            // Individual points from b fall out
            condensed.push((new_cluster, b, lambda, size_b));
        } else if size_b >= min_cluster_size {
            condensed.push((new_cluster, b, lambda, size_b));
            condensed.push((new_cluster, a, lambda, size_a));
        } else {
            // Both small → merge
            condensed.push((new_cluster, a, lambda, size_a));
            condensed.push((new_cluster, b, lambda, size_b));
        }

        uf_parent[a] = new_cluster;
        uf_parent[b] = new_cluster;
        uf_size[new_cluster] = size_a + size_b;
    }

    // Compute cluster stability
    let total_clusters = next_cluster;
    let mut stability = vec![0.0f64; total_clusters];
    let mut birth_lambda = vec![0.0f64; total_clusters];
    let mut children: Vec<Vec<usize>> = vec![vec![]; total_clusters];

    for &(parent, child, lambda, size) in &condensed {
        if parent < total_clusters {
            children[parent].push(child);
            if birth_lambda[parent] == 0.0 {
                birth_lambda[parent] = lambda;
            }
            // Stability contribution: sum of (lambda - birth_lambda) for each point
            if child < n {
                // Single point falling out
                stability[parent] += lambda - birth_lambda[parent];
            } else if size < min_cluster_size {
                // Small cluster falling out as noise
                stability[parent] += (lambda - birth_lambda[parent]) * size as f64;
            }
        }
    }

    // Select clusters based on method
    let mut selected = vec![false; total_clusters];
    let root = if total_clusters > n {
        total_clusters - 1
    } else {
        0
    };

    match method {
        ClusterSelectionMethod::EOM => {
            // Bottom-up: if stability[cluster] > sum(stability[children that are clusters]), keep it
            let mut is_cluster = vec![true; total_clusters];
            // Process bottom-up (lower indices first since we built them in order)
            for c in (n..total_clusters).rev() {
                let child_stability_sum: f64 = children[c]
                    .iter()
                    .filter(|&&ch| ch >= n)
                    .map(|&ch| stability[ch])
                    .sum();

                if child_stability_sum > stability[c] {
                    // Children are better
                    stability[c] = child_stability_sum;
                    is_cluster[c] = false;
                }
            }

            // Collect selected clusters
            for c in n..total_clusters {
                if is_cluster[c] && stability[c] > 0.0 {
                    selected[c] = true;
                }
            }

            if !allow_single_cluster && selected.iter().filter(|&&s| s).count() <= 1 {
                // Fall back to children of root
                selected[root] = false;
                for &ch in &children[root] {
                    if ch >= n {
                        selected[ch] = true;
                    }
                }
            }
        }
        ClusterSelectionMethod::Leaf => {
            // Select leaf clusters (no children that are clusters)
            for c in n..total_clusters {
                let has_cluster_child = children[c]
                    .iter()
                    .any(|&ch| ch >= n && uf_size[ch] >= min_cluster_size);
                if !has_cluster_child && uf_size[c] >= min_cluster_size {
                    selected[c] = true;
                }
            }
        }
    }

    // Assign labels: DFS from selected clusters
    let mut labels = vec![-1i64; n];
    let mut probabilities = vec![0.0f32; n];
    let mut persistence = Vec::new();

    let mut cluster_id = 0i64;
    for c in n..total_clusters {
        if !selected[c] {
            continue;
        }
        persistence.push(stability[c] as f32);

        // DFS to find all points in this cluster
        let mut stack = vec![c];
        while let Some(node) = stack.pop() {
            if node < n {
                labels[node] = cluster_id;
                probabilities[node] = 1.0;
            } else {
                for &ch in &children[node] {
                    if !selected[ch] || ch < n {
                        stack.push(ch);
                    }
                }
            }
        }
        cluster_id += 1;
    }

    (labels, probabilities, persistence)
}
