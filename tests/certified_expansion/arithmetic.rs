use super::fixture::{self, Client, Operation, Result, decode, enclosed, input, tensor, values};
use numr::prelude::*;
use solvr::common::arithmetic::with_interval_arithmetic;

pub fn references<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    let cases = if dtype == DType::F32 {
        super::reference_f32::CASES
    } else {
        super::reference_f64::CASES
    };
    assert_eq!(cases.len(), 14);
    for strided in [false, true] {
        for case in cases {
            let component =
                |bits| input::<R>(&[decode(bits, dtype); 2], dtype, client.device(), strided);
            let a = [
                component(case.a[0])?,
                component(case.a[1])?,
                component(case.a[2])?,
            ];
            let b = [
                component(case.b[0])?,
                component(case.b[1])?,
                component(case.b[2])?,
            ];
            let anchor = component(case.anchor)?;
            let before = a
                .iter()
                .chain(&b)
                .map(|value| fixture::bits(client, value))
                .collect::<numr::error::Result<Vec<_>>>()?;
            let (result, residual) = with_interval_arithmetic::<R, _, _, _>(
                client,
                dtype,
                "expansion reference",
                |context| {
                    let left = context.expansion_bounds(&a[0], &a[1], &a[2])?;
                    let right = context.expansion_bounds(&b[0], &b[1], &b[2])?;
                    let result = match case.op {
                        Operation::Add => context.expansion_add(&left, &right)?,
                        Operation::Sub => context.expansion_sub(&left, &right)?,
                        Operation::Mul => context.expansion_mul(&left, &right)?,
                        Operation::Div => context.expansion_div_positive(&left, &right)?,
                    };
                    let anchor = context.expansion_point(&anchor)?;
                    let residual = context.expansion_sub(&result, &anchor)?;
                    Ok((result, context.expansion_interval(&residual)?))
                },
            )?;
            enclosed(
                client,
                &residual,
                case.residual.map(|bits| decode(bits, dtype)),
                case.id,
            )?;
            for radius in values(client, result.abs_error())? {
                assert!(
                    radius.is_finite() && radius >= 0.0,
                    "{}: invalid radius",
                    case.id
                );
                assert!(
                    radius <= decode(case.max_abs_error, dtype),
                    "{}: radius {radius} exceeds budget",
                    case.id
                );
                if case.zero_error {
                    assert_eq!(radius, 0.0, "{}: exact expansion", case.id);
                }
                if case.nonzero_error {
                    assert!(radius > 0.0, "{}: missing uncertainty", case.id);
                }
            }
            for component in [result.high(), result.low(), result.abs_error()] {
                assert_eq!(component.shape(), &[2]);
                assert_eq!(component.dtype(), dtype);
                assert!(component.device().is_same(client.device()));
                assert!(component.is_contiguous());
            }
            let after = a
                .iter()
                .chain(&b)
                .map(|value| fixture::bits(client, value))
                .collect::<numr::error::Result<Vec<_>>>()?;
            assert_eq!(before, after, "{}: input mutation", case.id);
        }
    }
    Ok(())
}

pub fn cancellation<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    let delta_v = if dtype == DType::F32 {
        2.0_f64.powi(-24)
    } else {
        f64::from_bits(0x3cb105bc4806908f)
    };
    let delta_m = if dtype == DType::F32 {
        2.0_f64.powi(-23)
    } else {
        f64::from_bits(0x3cb9800000000000)
    };
    let scalar = |value| tensor::<R>(&[value], &[1], dtype, client.device());
    let seven = scalar(7.0)?;
    let negative_delta = scalar(-delta_v)?;
    let mass = scalar(delta_m)?;
    let hundred = scalar(100.0)?;
    let length = scalar(0.3)?;
    let fourteen = scalar(14.0)?;
    let delta = scalar(delta_v)?;
    let (volume, deficit, boosted, unequal) = with_interval_arithmetic::<R, _, _, _>(
        client,
        dtype,
        "expansion cancellation pipeline",
        |context| {
            let cap = context.expansion_point(&seven)?;
            let negative_delta = context.expansion_point(&negative_delta)?;
            let volume = context.expansion_add(&cap, &negative_delta)?;
            let deficit = context.expansion_sub(&cap, &volume)?;
            let mass = context.expansion_point(&mass)?;
            let hundred = context.expansion_point(&hundred)?;
            let boost = context.expansion_mul(&hundred, &volume)?;
            let momentum = context.expansion_add(&mass, &boost)?;
            let velocity = context.expansion_div_positive(&momentum, &volume)?;
            let boosted = context.expansion_sub(&velocity, &hundred)?;
            let length = context.expansion_point(&length)?;
            let fourteen = context.expansion_point(&fourteen)?;
            let cap_volume = context.expansion_mul(&fourteen, &length)?;
            let delta = context.expansion_point(&delta)?;
            let water = context.expansion_sub(&cap_volume, &delta)?;
            let missing_volume = context.expansion_sub(&cap_volume, &water)?;
            let unequal = context.expansion_div_positive(&missing_volume, &length)?;
            Ok((
                volume,
                context.expansion_interval(&deficit)?,
                context.expansion_interval(&boosted)?,
                context.expansion_interval(&unequal)?,
            ))
        },
    )?;
    assert_eq!(values(client, volume.high())?, [7.0]);
    assert_eq!(values(client, volume.low())?, [-delta_v]);
    assert_eq!(values(client, volume.abs_error())?, [0.0]);
    enclosed(client, &deficit, [delta_v; 2], "water deficit")?;
    assert!(values(client, deficit.lower())?[0] > 0.0);
    assert!(values(client, boosted.lower())?[0] > 0.0);
    assert!(values(client, unequal.lower())?[0] > 0.0);
    let cases = if dtype == DType::F32 {
        super::reference_f32::CASES
    } else {
        super::reference_f64::CASES
    };
    for (id, actual) in [
        ("captured_velocity", &boosted),
        ("unequal_length_deficit", &unequal),
    ] {
        let case = cases
            .iter()
            .find(|case| case.id == id)
            .ok_or("missing pipeline reference")?;
        assert_eq!(decode(case.anchor, dtype), 0.0);
        enclosed(
            client,
            actual,
            case.residual.map(|bits| decode(bits, dtype)),
            id,
        )?;
    }
    Ok(())
}

pub fn product_guards<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    let (p, low_exponent, high_exponent, normal_bits, one_bits) = if dtype == DType::F32 {
        (23, -40, 40, 0x00800000, 0x3f800000)
    } else {
        (52, -450, 450, 0x0010000000000000, 0x3ff0000000000000)
    };
    let encode = |value: f64| {
        if dtype == DType::F32 {
            u64::from((value as f32).to_bits())
        } else {
            value.to_bits()
        }
    };
    let low_power = 2.0_f64.powi(low_exponent);
    let high_power = 2.0_f64.powi(high_exponent);
    let next_one = decode(one_bits + 1, dtype);
    let low_x = decode(encode(low_power) - 1, dtype);
    let high_x = decode(encode(high_power) + 1, dtype);
    let high_anchor = decode(encode(high_power) + 2, dtype);
    let low_residual = decode(encode(2.0_f64.powi(low_exponent - p - 1)) - 2, dtype);
    let high_residual = 2.0_f64.powi(high_exponent - 2 * p);
    let cases = [
        (
            "normal product with subrepresentable residual",
            decode(normal_bits + 1, dtype),
            next_one,
            decode(normal_bits + 2, dtype),
            [0.0, decode(1, dtype)],
            true,
            false,
        ),
        (
            "lower guard positive residual",
            low_x,
            next_one,
            low_power,
            [low_residual; 2],
            false,
            false,
        ),
        (
            "lower guard negative residual",
            low_x,
            -next_one,
            -low_power,
            [-low_residual; 2],
            false,
            false,
        ),
        (
            "upper guard positive residual",
            high_x,
            next_one,
            high_anchor,
            [high_residual; 2],
            false,
            false,
        ),
        (
            "upper guard negative residual",
            high_x,
            -next_one,
            -high_anchor,
            [-high_residual; 2],
            false,
            false,
        ),
        (
            "lower guard exact product",
            low_power,
            next_one,
            decode(encode(low_power) + 1, dtype),
            [0.0; 2],
            false,
            true,
        ),
    ];
    for strided in [false, true] {
        for (id, x, y, anchor, expected, nonzero_error, zero_error) in cases {
            let x = input::<R>(&[x; 2], dtype, client.device(), strided)?;
            let y = input::<R>(&[y; 2], dtype, client.device(), strided)?;
            let anchor = input::<R>(&[anchor; 2], dtype, client.device(), strided)?;
            let (product, residual) = with_interval_arithmetic::<R, _, _, _>(
                client,
                dtype,
                "expansion product guard",
                |context| {
                    let x = context.expansion_point(&x)?;
                    let y = context.expansion_point(&y)?;
                    let product = context.expansion_mul(&x, &y)?;
                    let anchor = context.expansion_point(&anchor)?;
                    let residual = context.expansion_sub(&product, &anchor)?;
                    Ok((product, context.expansion_interval(&residual)?))
                },
            )?;
            enclosed(client, &residual, expected, id)?;
            for error in values(client, product.abs_error())? {
                assert!(error.is_finite() && error >= 0.0, "{id}: invalid radius");
                if nonzero_error {
                    assert!(
                        error > 0.0,
                        "{id}: missing subrepresentable residual radius"
                    );
                }
                if zero_error {
                    assert_eq!(error, 0.0, "{id}: exact product radius");
                }
            }
        }
    }
    Ok(())
}
