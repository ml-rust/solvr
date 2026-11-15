use super::super::certified::native::{self, RootClient};
use crate::DType;
use crate::optimize::{OptimizeError, OptimizeResult};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

pub(super) fn expanded<R, C>(
    client: &C,
    lower: &Tensor<R>,
    upper: &Tensor<R>,
    limit: &Tensor<R>,
) -> OptimizeResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: RootClient<R>,
{
    let zero = Tensor::zeros(upper.shape(), upper.dtype(), client.device())?;
    let midpoint = client.midpoint(lower, limit)?;
    let capped = client.ge(upper, &midpoint)?;
    // Mask capped rows before either subtraction or addition can overflow.
    let safe_lower = client.where_cond(&capped, &zero, lower)?;
    let safe_upper = client.where_cond(&capped, &zero, upper)?;
    let doubled = client.add(&safe_upper, &client.sub(&safe_upper, &safe_lower)?)?;
    let point = client.minimum(&client.where_cond(&capped, limit, &doubled)?, limit)?;
    let stalled = client.mul(&client.le(&point, upper)?, &client.lt(upper, limit)?)?;
    // Inactive rows include finite MAX. Never pass those values to next_up.
    let next_input = client.where_cond(&stalled, upper, &zero)?;
    let next = client.minimum(&client.next_up(&next_input)?, limit)?;
    let point = client.where_cond(&stalled, &next, &point)?;
    let invalid = client.maximum(
        &native::nonfinite(client, &point)?,
        &client.maximum(&client.lt(&point, upper)?, &client.gt(&point, limit)?)?,
    )?;
    if native::any(client, &invalid)? {
        return Err(OptimizeError::NumericalError {
            message: "Certified upper bracket candidate is nonfinite or outside [upper, upper_limit]. Check native arithmetic.".into(),
        });
    }
    Ok(point)
}
