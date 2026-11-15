use super::fixture::{Case, Client, constant, decode};
use numr::prelude::*;
use solvr::optimize::roots::{CertifiedRootQuery, CertifiedRootValues};

pub fn affine<R, C>(
    client: &C,
    x: &Tensor<R>,
    root: f64,
) -> numr::error::Result<CertifiedRootValues<R>>
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    let value = client.fused_mul_add_scalar(x, 1.0, -root)?;
    let zero = constant(client, x, 0.0)?;
    let exact_zero = client.eq(x, &constant(client, x, root)?)?;
    // Distinct represented affine operands differ by at least one subnormal.
    Ok(CertifiedRootValues {
        lower: client.where_cond(&exact_zero, &zero, &client.next_down(&value)?)?,
        upper: client.where_cond(&exact_zero, &zero, &client.next_up(&value)?)?,
    })
}
pub fn evaluate<R, C>(
    client: &C,
    case: &Case,
    query: &CertifiedRootQuery<R>,
) -> numr::error::Result<CertifiedRootValues<R>>
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    let x = &query.point;
    if case.oracle == "domain_enclosure" {
        return Ok(CertifiedRootValues {
            lower: affine(client, x, decode(case.upper, case.dtype))?.lower,
            upper: affine(client, x, decode(case.lower, case.dtype))?.upper,
        });
    }
    let value = match case.function {
        "three_x_minus_one" => {
            let value = client.fused_mul_add_scalar(x, 3.0, -1.0)?;
            return Ok(CertifiedRootValues {
                lower: client.next_down(&value)?,
                upper: client.next_up(&value)?,
            });
        }
        "x_squared_minus_two" => {
            let value = client.fused_mul_add(x, x, &constant(client, x, -2.0)?)?;
            return Ok(CertifiedRootValues {
                lower: client.next_down(&value)?,
                upper: client.next_up(&value)?,
            });
        }
        "x" => affine(client, x, 0.0)?,
        "x_minus_half" => affine(client, x, 0.5)?,
        "x_minus_one" => affine(client, x, 1.0)?,
        "x_minus_quarter" => affine(client, x, 0.25)?,
        "x_minus_three_quarters" => affine(client, x, 0.75)?,
        "x_plus_one" => affine(client, x, -1.0)?,
        "x_plus_quarter" => affine(client, x, -0.25)?,
        _ => return Err(super::fixture::sentinel()),
    };
    match case.oracle {
        "exact_physical" => Ok(value),
        "physical_plus_minus_absolute_residual" => {
            let zero = constant(client, x, 0.0)?;
            // Scaling by two is exact throughout these bounded fixture domains.
            Ok(CertifiedRootValues {
                lower: client.mul_scalar(&client.minimum(&value.lower, &zero)?, 2.0)?,
                upper: client.mul_scalar(&client.maximum(&value.upper, &zero)?, 2.0)?,
            })
        }
        "physical_plus_minus_quarter" | "physical_plus_minus_half" => {
            let radius = if case.oracle.ends_with("quarter") {
                0.25
            } else {
                0.5
            };
            Ok(CertifiedRootValues {
                lower: client.next_down(&client.add_scalar(&value.lower, -radius)?)?,
                upper: client.next_up(&client.add_scalar(&value.upper, radius)?)?,
            })
        }
        "nominal_x_minus_half_plus_minus_half" => Ok(CertifiedRootValues {
            lower: affine(client, x, 1.0)?.lower,
            upper: x.clone(),
        }),
        "nominal_jump_sixteenth_plus_minus_eighth" => {
            let above = client.ge(x, &constant(client, x, 0.25)?)?;
            let bias = client.where_cond(
                &above,
                &constant(client, x, 0.0625)?,
                &constant(client, x, -0.0625)?,
            )?;
            let lower = client.next_down(&client.add(&value.lower, &bias)?)?;
            let upper = client.next_up(&client.add(&value.upper, &bias)?)?;
            Ok(CertifiedRootValues {
                lower: client.next_down(&client.add_scalar(&lower, -0.125)?)?,
                upper: client.next_up(&client.add_scalar(&upper, 0.125)?)?,
            })
        }
        _ => Err(super::fixture::sentinel()),
    }
}
