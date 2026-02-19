//! Generic grey-scale morphology implementations.
use crate::DType;

use crate::signal::traits::nd_filters::{BoundaryMode, NdFilterAlgorithms};
use numr::error::Result;
use numr::ops::ScalarOps;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Grey-scale erosion (local minimum).
pub fn grey_erosion_impl<R, C>(client: &C, input: &Tensor<R>, size: &[usize]) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: NdFilterAlgorithms<R> + RuntimeClient<R>,
{
    client.minimum_filter(input, size, BoundaryMode::Nearest)
}

/// Grey-scale dilation (local maximum).
pub fn grey_dilation_impl<R, C>(client: &C, input: &Tensor<R>, size: &[usize]) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: NdFilterAlgorithms<R> + RuntimeClient<R>,
{
    client.maximum_filter(input, size, BoundaryMode::Nearest)
}

/// Grey-scale opening (erosion then dilation).
pub fn grey_opening_impl<R, C>(client: &C, input: &Tensor<R>, size: &[usize]) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: NdFilterAlgorithms<R> + RuntimeClient<R>,
{
    let eroded = grey_erosion_impl(client, input, size)?;
    grey_dilation_impl(client, &eroded, size)
}

/// Grey-scale closing (dilation then erosion).
pub fn grey_closing_impl<R, C>(client: &C, input: &Tensor<R>, size: &[usize]) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: NdFilterAlgorithms<R> + RuntimeClient<R>,
{
    let dilated = grey_dilation_impl(client, input, size)?;
    grey_erosion_impl(client, &dilated, size)
}

/// Morphological gradient (dilation - erosion).
pub fn morphological_gradient_impl<R, C>(
    client: &C,
    input: &Tensor<R>,
    size: &[usize],
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: NdFilterAlgorithms<R> + ScalarOps<R> + RuntimeClient<R>,
{
    let dilated = grey_dilation_impl(client, input, size)?;
    let eroded = grey_erosion_impl(client, input, size)?;
    client.sub(&dilated, &eroded)
}

/// White tophat (input - opening).
pub fn white_tophat_impl<R, C>(client: &C, input: &Tensor<R>, size: &[usize]) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: NdFilterAlgorithms<R> + ScalarOps<R> + RuntimeClient<R>,
{
    let opened = grey_opening_impl(client, input, size)?;
    client.sub(input, &opened)
}

/// Black tophat (closing - input).
pub fn black_tophat_impl<R, C>(client: &C, input: &Tensor<R>, size: &[usize]) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: NdFilterAlgorithms<R> + ScalarOps<R> + RuntimeClient<R>,
{
    let closed = grey_closing_impl(client, input, size)?;
    client.sub(&closed, input)
}
