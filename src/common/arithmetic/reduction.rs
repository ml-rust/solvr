//! Fixed adjacent-pair reductions with unchanged odd leaves.

use super::error::ArithmeticResult;
use super::interval::{Interval, IntervalArithmetic};
use crate::DType;
use numr::error::Error;
use numr::ops::{CompareOps, NextRepresentableOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

pub(crate) fn pairwise_sum_rows<R, C>(
    arithmetic: &mut IntervalArithmetic<'_, R, C>,
    input: &Tensor<R>,
) -> ArithmeticResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + CompareOps<R> + NextRepresentableOps<R> + RuntimeClient<R>,
{
    let (rows, mut width) = row_shape(input)?;
    arithmetic.finite(input)?;
    let client = arithmetic.client();
    if rows == 0 || width == 0 {
        return Ok(Tensor::zeros(&[rows], input.dtype(), client.device())?);
    }
    let mut values = input.clone();
    while width > 1 {
        let (even, odd) = pair_indices(client, width)?;
        let left = client.index_select(&values, 1, &even)?;
        let right = client.index_select(&values, 1, &odd)?;
        let sum = client.add(&left, &right)?;
        arithmetic.finite(&sum)?;
        values = if width % 2 == 0 {
            sum
        } else {
            let tail = values.narrow(1, width - 1, 1)?;
            client.cat(&[&sum, &tail], 1)?
        };
        width = width.div_ceil(2);
    }
    Ok(values.squeeze(Some(1)))
}

pub(crate) fn pairwise_interval_sum_rows<R, C>(
    arithmetic: &mut IntervalArithmetic<'_, R, C>,
    input: &Interval<R>,
) -> ArithmeticResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + CompareOps<R> + NextRepresentableOps<R> + RuntimeClient<R>,
{
    let (rows, mut width) = row_shape(&input.lower)?;
    let mut values = arithmetic.bounds(input.lower.clone(), input.upper.clone())?;
    let client = arithmetic.client();
    if rows == 0 || width == 0 {
        let zeros = Tensor::zeros(&[rows], input.lower.dtype(), client.device())?;
        return arithmetic.point(&zeros);
    }
    while width > 1 {
        let (even, odd) = pair_indices(client, width)?;
        let left = Interval {
            lower: client.index_select(&values.lower, 1, &even)?,
            upper: client.index_select(&values.upper, 1, &even)?,
        };
        let right = Interval {
            lower: client.index_select(&values.lower, 1, &odd)?,
            upper: client.index_select(&values.upper, 1, &odd)?,
        };
        let sum = arithmetic.add(&left, &right)?;
        values = if width % 2 == 0 {
            sum
        } else {
            let tail_lower = values.lower.narrow(1, width - 1, 1)?;
            let tail_upper = values.upper.narrow(1, width - 1, 1)?;
            Interval {
                lower: client.cat(&[&sum.lower, &tail_lower], 1)?,
                upper: client.cat(&[&sum.upper, &tail_upper], 1)?,
            }
        };
        width = width.div_ceil(2);
    }
    Ok(Interval {
        lower: values.lower.squeeze(Some(1)),
        upper: values.upper.squeeze(Some(1)),
    })
}

fn row_shape<R: Runtime<DType = DType>>(input: &Tensor<R>) -> ArithmeticResult<(usize, usize)> {
    let [rows, width] = input.shape() else {
        return Err(Error::InvalidArgument {
            arg: "pairwise_sum_rows",
            reason: format!(
                "Pairwise input shape {:?} is not rank two. Supply [rows, width].",
                input.shape()
            ),
        }
        .into());
    };
    if *rows > u32::MAX as usize || *width > u32::MAX as usize || input.numel() > u32::MAX as usize
    {
        return Err(Error::InvalidArgument {
            arg: "pairwise_sum_rows",
            reason: format!(
                "Pairwise shape {:?} exceeds u32 indexing. Reduce the query count.",
                input.shape()
            ),
        }
        .into());
    }
    if input
        .numel()
        .checked_mul(input.dtype().size_in_bytes())
        .is_none()
        || rows.checked_mul(input.dtype().size_in_bytes()).is_none()
    {
        return Err(Error::InvalidArgument {
            arg: "pairwise_sum_rows",
            reason: format!(
                "Pairwise shape {:?} exceeds allocation byte limits. Reduce the query count.",
                input.shape()
            ),
        }
        .into());
    }
    Ok((*rows, *width))
}

fn pair_indices<R, C>(client: &C, width: usize) -> ArithmeticResult<(Tensor<R>, Tensor<R>)>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R>,
{
    let stop = (width / 2 * 2) as f64;
    Ok((
        client.arange(0.0, stop, 2.0, DType::I64)?,
        client.arange(1.0, stop, 2.0, DType::I64)?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    #[test]
    fn adjacent_pair_tree_carries_odd_leaves_and_bounds_cancellation() -> ArithmeticResult<()> {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        let mut arithmetic = IntervalArithmetic::new(&client, DType::F64)?;
        let input = Tensor::from_slice(&[1.0e16f64, 1.0, -1.0e16, 1.0, 7.0], &[1, 5], &device)?;
        let nominal = pairwise_sum_rows(&mut arithmetic, &input)?;
        let interval = arithmetic.point(&input)?;
        let result = pairwise_interval_sum_rows(&mut arithmetic, &interval)?;
        arithmetic.check("pairwise cancellation")?;
        assert_eq!(nominal.shape(), &[1]);
        assert_eq!(nominal.item::<f64>()?, 7.0);
        assert!(result.lower.item::<f64>()? <= 9.0);
        assert!(result.upper.item::<f64>()? >= 9.0);
        Ok(())
    }

    #[test]
    fn empty_and_zero_width_rows_return_exact_zeros() -> ArithmeticResult<()> {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        let mut arithmetic = IntervalArithmetic::new(&client, DType::F64)?;
        for shape in [[0, 5], [2, 0]] {
            let input = Tensor::zeros(&shape, DType::F64, &device)?;
            let nominal = pairwise_sum_rows(&mut arithmetic, &input)?;
            let interval = arithmetic.point(&input)?;
            let result = pairwise_interval_sum_rows(&mut arithmetic, &interval)?;
            assert_eq!(nominal.shape(), &[shape[0]]);
            for output in [&nominal, &result.lower, &result.upper] {
                assert!(
                    output
                        .try_to_vec::<f64>()?
                        .iter()
                        .all(|value| value.to_bits() == 0)
                );
            }
        }
        arithmetic.check("empty pairwise rows")?;
        Ok(())
    }

    #[test]
    fn zero_width_rows_reject_unrepresentable_output_counts() -> ArithmeticResult<()> {
        use super::super::error::ArithmeticError;

        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        let mut arithmetic = IntervalArithmetic::new(&client, DType::F64)?;
        let rows = usize::try_from(u64::from(u32::MAX) + 1)
            .unwrap_or(usize::MAX / DType::F64.size_in_bytes() + 1);
        let input = Tensor::empty(&[rows, 0], DType::F64, &device)?;
        assert_eq!(input.numel(), 0);
        let interval = Interval {
            lower: input.clone(),
            upper: input.clone(),
        };
        for result in [
            pairwise_sum_rows(&mut arithmetic, &input).map(|_| ()),
            pairwise_interval_sum_rows(&mut arithmetic, &interval).map(|_| ()),
        ] {
            assert!(matches!(
                result,
                Err(ArithmeticError::Numr(Error::InvalidArgument {
                    arg: "pairwise_sum_rows",
                    ..
                }))
            ));
        }
        Ok(())
    }
}
