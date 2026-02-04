//! Generic distance computation implementations.
//!
//! These are thin wrappers around numr's DistanceOps trait, providing
//! a consistent interface within solvr's spatial module.

use crate::spatial::{validate_matching_dims, validate_points_2d, validate_points_dtype};
use numr::error::Result;
use numr::ops::DistanceOps;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use crate::spatial::traits::distance::DistanceMetric;

/// Generic implementation of cdist (pairwise distances between two point sets).
pub fn cdist_impl<R, C>(
    client: &C,
    x: &Tensor<R>,
    y: &Tensor<R>,
    metric: DistanceMetric,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: DistanceOps<R> + RuntimeClient<R>,
{
    validate_points_dtype(x.dtype(), "cdist")?;
    validate_points_dtype(y.dtype(), "cdist")?;
    validate_points_2d(x.shape(), "cdist")?;
    validate_points_2d(y.shape(), "cdist")?;
    validate_matching_dims(x.shape(), y.shape(), "cdist")?;

    // Delegate to numr's DistanceOps
    client.cdist(x, y, metric)
}

/// Generic implementation of pdist (pairwise distances within a point set).
pub fn pdist_impl<R, C>(client: &C, x: &Tensor<R>, metric: DistanceMetric) -> Result<Tensor<R>>
where
    R: Runtime,
    C: DistanceOps<R> + RuntimeClient<R>,
{
    validate_points_dtype(x.dtype(), "pdist")?;
    validate_points_2d(x.shape(), "pdist")?;

    // Delegate to numr's DistanceOps
    client.pdist(x, metric)
}

/// Generic implementation of squareform (condensed to square).
pub fn squareform_impl<R, C>(client: &C, condensed: &Tensor<R>, n: usize) -> Result<Tensor<R>>
where
    R: Runtime,
    C: DistanceOps<R> + RuntimeClient<R>,
{
    // Delegate to numr's DistanceOps
    client.squareform(condensed, n)
}

/// Generic implementation of squareform_inverse (square to condensed).
pub fn squareform_inverse_impl<R, C>(client: &C, square: &Tensor<R>) -> Result<Tensor<R>>
where
    R: Runtime,
    C: DistanceOps<R> + RuntimeClient<R>,
{
    // Delegate to numr's DistanceOps
    client.squareform_inverse(square)
}
