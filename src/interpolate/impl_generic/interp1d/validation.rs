//! Input and finite-arithmetic admission for a common 1D grid.

use crate::DType;
use crate::interpolate::error::{InterpolateError, InterpolateResult};
use numr::ops::{CompareOps, TensorOps};
use numr::runtime::{Device, Runtime, RuntimeClient};
use numr::tensor::Tensor;

pub(in crate::interpolate::impl_generic) fn check_inputs<R, C>(
    client: &C,
    x: &Tensor<R>,
    y: &Tensor<R>,
    query: &Tensor<R>,
) -> InterpolateResult<()>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + CompareOps<R> + RuntimeClient<R>,
{
    check_metadata(client, x, y, query)?;
    check_values(client, x, y, query)
}

pub(in crate::interpolate::impl_generic) fn check_metadata<R, C>(
    client: &C,
    x: &Tensor<R>,
    y: &Tensor<R>,
    query: &Tensor<R>,
) -> InterpolateResult<()>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + CompareOps<R> + RuntimeClient<R>,
{
    for (name, value) in [("x", x), ("y", y), ("x_new", query)] {
        if value.ndim() != 1 {
            return Err(InterpolateError::DimensionMismatch {
                expected: 1,
                actual: value.ndim(),
                context: format!("interp1d {name}: require rank1"),
            });
        }
        if value.dtype() != x.dtype() {
            return Err(InterpolateError::InvalidParameter {
                parameter: name.into(),
                message: format!(
                    "dtype={:?}: require matching {:?}",
                    value.dtype(),
                    x.dtype()
                ),
            });
        }
        if !client.device().is_same(value.device()) {
            return Err(InterpolateError::InvalidParameter {
                parameter: name.into(),
                message: "device differs from configured client".into(),
            });
        }
    }
    if !matches!(x.dtype(), DType::F32 | DType::F64) || !R::computes_float_dtype(x.dtype()) {
        return Err(InterpolateError::InvalidParameter {
            parameter: "dtype".into(),
            message: format!("{:?}: require backend-supported F32 or F64", x.dtype()),
        });
    }
    let n = x.numel();
    if y.numel() != n {
        return Err(InterpolateError::ShapeMismatch {
            expected: n,
            actual: y.numel(),
            context: "interp1d grid values".into(),
        });
    }
    if n < 2 {
        return Err(InterpolateError::InsufficientData {
            required: 2,
            actual: n,
            context: "interp1d grid".into(),
        });
    }
    Ok(())
}

pub(in crate::interpolate::impl_generic) fn check_values<R, C>(
    client: &C,
    x: &Tensor<R>,
    y: &Tensor<R>,
    query: &Tensor<R>,
) -> InterpolateResult<()>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + CompareOps<R> + RuntimeClient<R>,
{
    let n = x.numel();
    for (name, value) in [("x", x), ("y", y), ("x_new", query)] {
        if value.numel() > 0 {
            finite(client, value, name)?;
        }
    }
    let lo = x.narrow(0, 0, n - 1)?;
    let hi = x.narrow(0, 1, n - 1)?;
    let dx = client.sub(&hi, &lo)?;
    finite(client, &dx, "adjacent grid width")?;
    let zero = Tensor::zeros(dx.shape(), dx.dtype(), client.device())?;
    if any(client, &client.le(&dx, &zero)?)? {
        return Err(InterpolateError::NotMonotonic {context:"interp1d: require strictly increasing coordinates with positive represented adjacent widths".into()});
    }
    Ok(())
}

pub(in crate::interpolate::impl_generic) fn finite<R, C>(
    client: &C,
    value: &Tensor<R>,
    name: &str,
) -> InterpolateResult<()>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R>,
{
    let invalid = client.maximum(
        &client.cast(&client.isnan(value)?, value.dtype())?,
        &client.cast(&client.isinf(value)?, value.dtype())?,
    )?;
    if any(client, &invalid)? {
        return Err(InterpolateError::NumericalError {
            message: format!(
                "interp1d {name} contains nonfinite arithmetic. Rescale the grid, values, or queries"
            ),
        });
    }
    Ok(())
}

pub(in crate::interpolate::impl_generic) fn any<R, C>(
    client: &C,
    mask: &Tensor<R>,
) -> InterpolateResult<bool>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R>,
{
    let value = client.any(mask, &[], false)?;
    match mask.dtype() {
        DType::F32 => Ok(value.item::<f32>()? != 0.0),
        DType::F64 => Ok(value.item::<f64>()? != 0.0),
        _ => Err(InterpolateError::NumericalError {
            message: "interp1d control mask requires F32 or F64".into(),
        }),
    }
}

pub(in crate::interpolate::impl_generic) fn nonzero<R, C>(
    client: &C,
    value: &Tensor<R>,
    required: &Tensor<R>,
    name: &str,
) -> InterpolateResult<()>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + CompareOps<R> + RuntimeClient<R>,
{
    let zero = Tensor::zeros(value.shape(), value.dtype(), client.device())?;
    if any(client, &client.mul(required, &client.eq(value, &zero)?)?)? {
        return Err(InterpolateError::NumericalError {
            message: format!(
                "interp1d {name} underflows to zero. Rescale the grid, values, or queries"
            ),
        });
    }
    Ok(())
}
