//! Generic binary morphology implementations.
//!
//! Binary erosion = minimum filter on binary input, then threshold.
//! Binary dilation = maximum filter on binary input, then threshold.
use crate::DType;

use crate::morphology::traits::binary::StructuringElement;
use crate::signal::traits::nd_filters::{BoundaryMode, NdFilterAlgorithms};
use numr::error::{Error, Result};
use numr::ops::{CompareOps, ConditionalOps, ReduceOps, ScalarOps, UtilityOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Generic binary erosion: minimum filter + threshold at 1.0.
pub fn binary_erosion_impl<R, C>(
    client: &C,
    input: &Tensor<R>,
    _structure: StructuringElement,
    iterations: usize,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: NdFilterAlgorithms<R>
        + ScalarOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + UtilityOps<R>
        + RuntimeClient<R>,
{
    if input.ndim() == 0 {
        return Err(Error::InvalidArgument {
            arg: "input",
            reason: "binary_erosion requires at least 1D input".to_string(),
        });
    }

    let ndim = input.ndim();
    let size = vec![3usize; ndim];

    // Binarize input: nonzero -> 1.0, zero -> 0.0
    let zero = client.fill(input.shape(), 0.0, input.dtype())?;
    let one = client.fill(input.shape(), 1.0, input.dtype())?;
    let mask = client.ne(input, &zero)?;
    let mut result = client.where_cond(&mask, &one, &zero)?;

    for _ in 0..iterations {
        // Minimum filter: if any neighbor is 0, result is 0
        let filtered = client.minimum_filter(&result, &size, BoundaryMode::Constant(0.0))?;
        // Threshold: >= 1.0 means all neighbors were 1
        let thresh = client.ge(&filtered, &one)?;
        result = client.where_cond(&thresh, &one, &zero)?;
    }

    Ok(result)
}

/// Generic binary dilation: maximum filter on binary input.
pub fn binary_dilation_impl<R, C>(
    client: &C,
    input: &Tensor<R>,
    _structure: StructuringElement,
    iterations: usize,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: NdFilterAlgorithms<R>
        + ScalarOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + UtilityOps<R>
        + RuntimeClient<R>,
{
    if input.ndim() == 0 {
        return Err(Error::InvalidArgument {
            arg: "input",
            reason: "binary_dilation requires at least 1D input".to_string(),
        });
    }

    let ndim = input.ndim();
    let size = vec![3usize; ndim];

    let zero = client.fill(input.shape(), 0.0, input.dtype())?;
    let one = client.fill(input.shape(), 1.0, input.dtype())?;
    let mask = client.ne(input, &zero)?;
    let mut result = client.where_cond(&mask, &one, &zero)?;

    for _ in 0..iterations {
        let filtered = client.maximum_filter(&result, &size, BoundaryMode::Constant(0.0))?;
        let thresh = client.gt(&filtered, &zero)?;
        result = client.where_cond(&thresh, &one, &zero)?;
    }

    Ok(result)
}

/// Binary opening: erosion then dilation.
pub fn binary_opening_impl<R, C>(
    client: &C,
    input: &Tensor<R>,
    structure: StructuringElement,
    iterations: usize,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: NdFilterAlgorithms<R>
        + ScalarOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + UtilityOps<R>
        + RuntimeClient<R>,
{
    let eroded = binary_erosion_impl(client, input, structure, iterations)?;
    binary_dilation_impl(client, &eroded, structure, iterations)
}

/// Binary closing: dilation then erosion.
pub fn binary_closing_impl<R, C>(
    client: &C,
    input: &Tensor<R>,
    structure: StructuringElement,
    iterations: usize,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: NdFilterAlgorithms<R>
        + ScalarOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + UtilityOps<R>
        + RuntimeClient<R>,
{
    let dilated = binary_dilation_impl(client, input, structure, iterations)?;
    binary_erosion_impl(client, &dilated, structure, iterations)
}

/// Fill holes in binary objects.
///
/// Algorithm: complement the input, flood-fill from edges, complement again.
/// Implemented iteratively: dilate the edge mask, intersect with complement,
/// repeat until convergence.
pub fn binary_fill_holes_impl<R, C>(client: &C, input: &Tensor<R>) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: NdFilterAlgorithms<R>
        + ScalarOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + UtilityOps<R>
        + RuntimeClient<R>,
{
    let ndim = input.ndim();
    let size = vec![3usize; ndim];

    let zero = client.fill(input.shape(), 0.0, input.dtype())?;
    let one = client.fill(input.shape(), 1.0, input.dtype())?;

    // Binarize
    let mask = client.ne(input, &zero)?;
    let binary = client.where_cond(&mask, &one, &zero)?;

    // Complement
    let complement = client.sub(&one, &binary)?;

    // Initialize marker as complement (all holes start as candidates)
    // then iteratively constrain by dilating, intersecting with complement
    let mut marker = complement.clone();

    // Iterative geodesic dilation: dilate marker, intersect with complement
    let max_iter = input.shape().iter().sum::<usize>(); // conservative bound
    for _ in 0..max_iter {
        let dilated = client.maximum_filter(&marker, &size, BoundaryMode::Constant(1.0))?;
        let new_marker = client.minimum(&dilated, &complement)?;

        // Check convergence
        let diff = client.sub(&new_marker, &marker)?;
        let diff_abs = client.abs(&diff)?;
        let diff_sum = client.sum(&diff_abs, &[], false)?;
        let diff_val: Vec<f64> = diff_sum.to_vec();
        marker = new_marker;

        if diff_val[0] < 1e-10 {
            break;
        }
    }

    // Holes = complement - marker (regions not reachable from edges)
    let holes = client.sub(&complement, &marker)?;

    // Result = input OR holes
    client.add(&binary, &holes)
}
