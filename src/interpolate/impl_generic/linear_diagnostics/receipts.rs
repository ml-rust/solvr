//! Independent signed-ordinate Linear error decomposition over retained ordinary intermediates.

use crate::DType;
use crate::common::arithmetic::bounds::Bounds;
use crate::interpolate::error::InterpolateResult;
use crate::interpolate::impl_generic::interp1d::linear::LinearEvaluation;
use crate::interpolate::traits::linear_diagnostics::LinearInterp1dReceipt;
use numr::ops::{CompareOps, TensorOps, TwoSumOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

pub(super) fn assemble<R, C>(
    client: &C,
    query: &Tensor<R>,
    evaluation: LinearEvaluation<R>,
) -> InterpolateResult<LinearInterp1dReceipt<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + CompareOps<R> + TwoSumOps<R> + RuntimeClient<R>,
{
    let zero = Tensor::zeros(query.shape(), query.dtype(), client.device())?;
    let one = Tensor::ones(query.shape(), query.dtype(), client.device())?;
    let mut bounds = Bounds::new(client, query)?;
    let left_coordinate = client.where_cond(&evaluation.active, &evaluation.x0, &zero)?;
    let right_coordinate = client.where_cond(&evaluation.active, &evaluation.x1, &one)?;
    let (width, width_low) = client.two_sum(&right_coordinate, &client.neg(&left_coordinate)?)?;
    let used_query = client.where_cond(&evaluation.active, query, &zero)?;
    let (offset, offset_low) = client.two_sum(&used_query, &client.neg(&left_coordinate)?)?;
    let (difference, difference_low) =
        client.two_sum(&evaluation.right, &client.neg(&evaluation.left)?)?;
    for value in [
        &width,
        &width_low,
        &offset,
        &offset_low,
        &difference,
        &difference_low,
    ] {
        bounds.finite(value)?;
    }
    for mismatch in [
        client.ne(&width, &evaluation.width)?,
        client.ne(&offset, &evaluation.offset)?,
        client.ne(&difference, &evaluation.difference)?,
        client.lt(&evaluation.coordinate, &zero)?,
        client.gt(&evaluation.coordinate, &one)?,
    ] {
        bounds.mark(&mismatch)?;
    }

    let coordinate_numerator = bounds.add(&client.abs(&width_low)?, &client.abs(&offset_low)?)?;
    let coordinate_input_error = bounds.divide(&coordinate_numerator, &evaluation.width)?;
    let coordinate_rounding = bounds.rounding(&evaluation.coordinate)?;
    let coordinate_error = bounds.add(&coordinate_input_error, &coordinate_rounding)?;
    let ordinate_coordinate_error =
        bounds.multiply(&client.abs(&evaluation.difference)?, &coordinate_error)?;
    let increment_rounding = bounds.rounding(&evaluation.increment)?;
    let used_output = client.where_cond(&evaluation.active, &evaluation.values, &zero)?;
    let output_rounding = bounds.rounding(&used_output)?;
    let rounding_error = bounds.add(&increment_rounding, &output_rounding)?;
    let represented_difference_error = bounds.add(&ordinate_coordinate_error, &rounding_error)?;
    let error = bounds.add(&client.abs(&difference_low)?, &represented_difference_error)?;
    let absolute_error = client.where_cond(&evaluation.active, &error, &zero)?;
    bounds.finite(&absolute_error)?;
    bounds.mark(&client.lt(&absolute_error, &zero)?)?;
    bounds.finish()?;
    Ok(LinearInterp1dReceipt {
        values: evaluation.values,
        absolute_error,
        lower_indices: evaluation.lower,
        upper_indices: evaluation.upper,
        at_lower: evaluation.at_lower,
        at_upper: evaluation.at_upper,
        constant: evaluation.constant,
    })
}
