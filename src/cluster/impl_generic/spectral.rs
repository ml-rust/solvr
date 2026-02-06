//! Generic spectral clustering implementation.
//!
//! 1. Build affinity matrix (RBF, kNN, or precomputed)
//! 2. Compute graph Laplacian
//! 3. Eigendecomposition → take k smallest eigenvectors
//! 4. Run k-means on eigenvector embedding

use crate::cluster::traits::kmeans::{KMeansInit, KMeansOptions, KMeansResult};
use crate::cluster::traits::spectral::{AffinityType, LaplacianType, SpectralOptions};
use crate::cluster::validation::{validate_cluster_dtype, validate_data_2d, validate_n_clusters};
use numr::algorithm::linalg::LinearAlgebraAlgorithms;
use numr::error::Result;
use numr::ops::{
    CompareOps, ConditionalOps, CumulativeOps, DistanceMetric, DistanceOps, IndexingOps, LinalgOps,
    RandomOps, ReduceOps, ScalarOps, ShapeOps, SortingOps, TensorOps, TypeConversionOps, UnaryOps,
    UtilityOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Generic spectral clustering implementation.
pub fn spectral_clustering_impl<R, C>(
    client: &C,
    data: &Tensor<R>,
    options: &SpectralOptions,
) -> Result<KMeansResult<R>>
where
    R: Runtime,
    C: LinearAlgebraAlgorithms<R>
        + DistanceOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TensorOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ShapeOps<R>
        + UnaryOps<R>
        + UtilityOps<R>
        + IndexingOps<R>
        + RandomOps<R>
        + SortingOps<R>
        + CumulativeOps<R>
        + TypeConversionOps<R>
        + LinalgOps<R>
        + RuntimeClient<R>,
{
    validate_cluster_dtype(data.dtype(), "spectral_clustering")?;
    validate_data_2d(data.shape(), "spectral_clustering")?;
    validate_n_clusters(options.n_clusters, data.shape()[0], "spectral_clustering")?;

    let n = data.shape()[0];
    let k = options.n_clusters;
    let dtype = data.dtype();
    let device = data.device();

    // 1. Build affinity matrix [n, n]
    let affinity = match &options.affinity {
        AffinityType::Precomputed => data.clone(),
        AffinityType::RBF { gamma } => {
            let sq_dists = client.cdist(data, data, DistanceMetric::SquaredEuclidean)?; // [n, n]
            let gamma_val = match gamma {
                Some(g) => *g,
                None => {
                    // Auto-estimate: 1 / (n_features * variance)
                    let d = data.shape()[1];
                    1.0 / d as f64
                }
            };
            let neg_gamma = Tensor::<R>::full_scalar(&[n, n], dtype, -gamma_val, device);
            let scaled = client.mul(&neg_gamma, &sq_dists)?;
            client.exp(&scaled)?
        }
        AffinityType::NearestNeighbors { n_neighbors } => {
            let dists = client.cdist(data, data, DistanceMetric::SquaredEuclidean)?;
            let sorted = client.sort(&dists, 1, false)?;
            // k-th neighbor distance threshold
            let k_dist = sorted.narrow(1, *n_neighbors, 1)?; // [n, 1]
            let k_dist_broadcast = k_dist.broadcast_to(&[n, n])?;
            // Binary adjacency: dist <= k-th neighbor distance
            let mask = client.le(&dists, &k_dist_broadcast)?;
            // Symmetrize
            let mask_t = mask.transpose(0, 1)?;
            // OR via maximum since mask values are 0/1
            client.maximum(&mask, &mask_t)?
        }
    };

    // 2. Compute graph Laplacian
    // Degree: D = diag(sum(W, dim=1))
    let degree = client.sum(&affinity, &[1], false)?; // [n]

    let embedding = match options.laplacian {
        LaplacianType::Unnormalized => {
            // L = D - W
            let d_diag = LinalgOps::diagflat(client, &degree)?; // [n, n]
            let laplacian = client.sub(&d_diag, &affinity)?;

            // Eigendecomposition of symmetric Laplacian
            let eig = client.eig_decompose_symmetric(&laplacian)?;
            // eigenvalues sorted descending by magnitude; we want k smallest
            // Laplacian eigenvalues are non-negative, smallest = last k in descending order
            let eigvecs = eig.eigenvectors; // [n, n]
            // Take last k columns (smallest eigenvalues)
            eigvecs.narrow(1, n - k, k)?
        }
        LaplacianType::SymmetricNormalized => {
            // L_sym = I - D^{-1/2} W D^{-1/2}
            let d_inv_sqrt = client.pow_scalar(&degree, -0.5)?; // [n]
            // Handle zero degree
            let zero = Tensor::<R>::zeros(&[n], dtype, device);
            let is_zero = client.eq(&degree, &zero)?;
            let d_inv_sqrt = client.where_cond(&is_zero, &zero, &d_inv_sqrt)?;

            let d_inv_sqrt_row = d_inv_sqrt.unsqueeze(1)?.broadcast_to(&[n, n])?;
            let d_inv_sqrt_col = d_inv_sqrt.unsqueeze(0)?.broadcast_to(&[n, n])?;
            let normalized =
                client.mul(&client.mul(&d_inv_sqrt_row, &affinity)?, &d_inv_sqrt_col)?;

            // L_sym = I - normalized
            let eye = LinalgOps::diagflat(client, &Tensor::<R>::ones(&[n], dtype, device))?;
            let laplacian = client.sub(&eye, &normalized)?;

            let eig = client.eig_decompose_symmetric(&laplacian)?;
            let eigvecs = eig.eigenvectors;
            eigvecs.narrow(1, n - k, k)?
        }
        LaplacianType::RandomWalk => {
            // L_rw = I - D^{-1} W
            let d_inv = client.pow_scalar(&degree, -1.0)?;
            let zero = Tensor::<R>::zeros(&[n], dtype, device);
            let is_zero = client.eq(&degree, &zero)?;
            let d_inv = client.where_cond(&is_zero, &zero, &d_inv)?;

            let _d_inv_diag = LinalgOps::diagflat(client, &d_inv)?;
            // Use symmetric normalized for eigenvectors (same eigenvectors for random walk)
            let d_inv_sqrt = client.pow_scalar(&degree, -0.5)?;
            let d_inv_sqrt = client.where_cond(&is_zero, &zero, &d_inv_sqrt)?;
            let d_inv_sqrt_row = d_inv_sqrt.unsqueeze(1)?.broadcast_to(&[n, n])?;
            let d_inv_sqrt_col = d_inv_sqrt.unsqueeze(0)?.broadcast_to(&[n, n])?;
            let normalized =
                client.mul(&client.mul(&d_inv_sqrt_row, &affinity)?, &d_inv_sqrt_col)?;
            let eye = LinalgOps::diagflat(client, &Tensor::<R>::ones(&[n], dtype, device))?;
            let laplacian = client.sub(&eye, &normalized)?;

            let eig = client.eig_decompose_symmetric(&laplacian)?;
            let eigvecs = eig.eigenvectors;
            eigvecs.narrow(1, n - k, k)?
        }
    };

    // 3. Normalize rows of embedding (for symmetric normalized Laplacian)
    let row_norms = client.sum(&client.mul(&embedding, &embedding)?, &[1], true)?; // [n, 1]
    let row_norms = client.sqrt(&row_norms)?;
    let safe_norms = client.maximum(
        &row_norms,
        &Tensor::<R>::full_scalar(&[1, 1], dtype, 1e-10, device),
    )?;
    let embedding = client.div(&embedding, &safe_norms.broadcast_to(&[n, k])?)?;

    // 4. Run k-means on the embedding
    let km_opts = KMeansOptions {
        n_clusters: k,
        max_iter: 300,
        tol: 1e-4,
        n_init: options.n_init,
        init: KMeansInit::KMeansPlusPlus,
        ..Default::default()
    };

    super::kmeans::kmeans_impl(client, &embedding, &km_opts)
}
