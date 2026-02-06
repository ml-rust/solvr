//! Generic Bisecting K-Means clustering implementation.

use crate::cluster::traits::bisecting_kmeans::{BisectingKMeansOptions, BisectingStrategy};
use crate::cluster::traits::kmeans::{KMeansInit, KMeansOptions, KMeansResult};
use crate::cluster::validation::{validate_cluster_dtype, validate_data_2d, validate_n_clusters};
use numr::dtype::DType;
use numr::error::Result;
use numr::ops::{
    CompareOps, ConditionalOps, CumulativeOps, DistanceOps, IndexingOps, RandomOps, ReduceOps,
    ScalarOps, ShapeOps, SortingOps, TensorOps, TypeConversionOps, UnaryOps, UtilityOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

pub fn bisecting_kmeans_impl<R, C>(
    client: &C,
    data: &Tensor<R>,
    options: &BisectingKMeansOptions,
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
    validate_cluster_dtype(data.dtype(), "bisecting_kmeans")?;
    validate_data_2d(data.shape(), "bisecting_kmeans")?;
    validate_n_clusters(options.n_clusters, data.shape()[0], "bisecting_kmeans")?;

    let n = data.shape()[0];
    let d = data.shape()[1];
    let dtype = data.dtype();
    let device = data.device();
    let k = options.n_clusters;

    // Start: all points in cluster 0
    let mut labels = Tensor::<R>::zeros(&[n], DType::I64, device);
    let mut current_k = 1usize;

    while current_k < k {
        let labels_f = client.cast(&labels, dtype)?;
        let counts = client.bincount(&labels, None, current_k)?;
        let counts_f = client.cast(&counts, dtype)?;

        let split_cluster: i64 = match options.bisecting_strategy {
            BisectingStrategy::BiggestCluster => {
                client.argmax(&counts_f, 0, false)?.reshape(&[1])?.item()?
            }
            BisectingStrategy::HighestSSE => {
                let labels_exp = labels.unsqueeze(1)?.broadcast_to(&[n, d])?;
                let dst = Tensor::<R>::zeros(&[current_k, d], dtype, device);
                let sums = client.scatter_reduce(
                    &dst,
                    0,
                    &labels_exp,
                    data,
                    numr::ops::ScatterReduceOp::Sum,
                    false,
                )?;
                let c_safe =
                    client.maximum(&counts_f, &Tensor::<R>::ones(&[current_k], dtype, device))?;
                let c_exp = c_safe.unsqueeze(1)?.broadcast_to(&[current_k, d])?;
                let centroids = client.div(&sums, &c_exp)?;
                let pt_c = client.index_select(&centroids, 0, &labels)?;
                let diff = client.sub(data, &pt_c)?;
                let sq = client.mul(&diff, &diff)?;
                let pt_sq = client.sum(&sq, &[1], false)?;
                let sse_dst = Tensor::<R>::zeros(&[current_k], dtype, device);
                let sse = client.scatter_reduce(
                    &sse_dst,
                    0,
                    &labels,
                    &pt_sq,
                    numr::ops::ScatterReduceOp::Sum,
                    false,
                )?;
                client.argmax(&sse, 0, false)?.reshape(&[1])?.item()?
            }
        };

        // Extract indices of points in the split cluster
        let split_val = Tensor::<R>::full_scalar(&[1], dtype, split_cluster as f64, device);
        let mask = client.eq(&labels_f, &split_val)?;
        let all_indices = client.arange(0.0, n as f64, 1.0, DType::I64)?;
        let mask_u8 = client.cast(&mask, DType::U8)?;
        let selected = client.masked_select(&all_indices, &mask_u8)?;
        let n_in = selected.shape()[0];

        if n_in < 2 {
            break;
        }

        let cluster_data = client.index_select(data, 0, &selected)?;

        // Run 2-means
        let sub_opts: KMeansOptions<R> = KMeansOptions {
            n_clusters: 2,
            max_iter: options.max_iter,
            tol: options.tol,
            n_init: options.n_init,
            init: KMeansInit::KMeansPlusPlus,
            ..Default::default()
        };
        let sub_result = super::kmeans::kmeans_impl(client, &cluster_data, &sub_opts)?;

        // Update labels: sub_label==1 → new cluster id (current_k)
        let is_one = client.eq(
            &sub_result.labels,
            &Tensor::<R>::ones(&[1], DType::I64, device),
        )?;
        let new_labels = client.where_cond(
            &is_one,
            &Tensor::<R>::full_scalar(&[n_in], DType::I64, current_k as f64, device),
            &Tensor::<R>::full_scalar(&[n_in], DType::I64, split_cluster as f64, device),
        )?;

        // Scatter back into global labels
        labels = client
            .scatter(
                &labels.unsqueeze(0)?,
                1,
                &selected.unsqueeze(0)?,
                &new_labels.unsqueeze(0)?,
            )?
            .squeeze(Some(0));

        current_k += 1;
    }

    // Compute final centroids and inertia
    let labels_exp = labels.unsqueeze(1)?.broadcast_to(&[n, d])?;
    let dst = Tensor::<R>::zeros(&[k, d], dtype, device);
    let sums = client.scatter_reduce(
        &dst,
        0,
        &labels_exp,
        data,
        numr::ops::ScatterReduceOp::Sum,
        false,
    )?;
    let counts = client.bincount(&labels, None, k)?;
    let counts_f = client.cast(&counts, dtype)?;
    let counts_safe = client.maximum(&counts_f, &Tensor::<R>::ones(&[k], dtype, device))?;
    let counts_exp = counts_safe.unsqueeze(1)?.broadcast_to(&[k, d])?;
    let centroids = client.div(&sums, &counts_exp)?;

    let pt_c = client.index_select(&centroids, 0, &labels)?;
    let diff = client.sub(data, &pt_c)?;
    let sq = client.mul(&diff, &diff)?;
    let inertia = client.sum(&sq, &[0, 1], false)?;

    Ok(KMeansResult {
        centroids,
        labels,
        inertia,
        n_iter: current_k,
    })
}
