use super::fixture::{Client, Result, affine, constant, input, read, tensor};
use numr::prelude::*;
use solvr::optimize::roots::{
    CertifiedRootValues, CertifiedUpperBracketAlgorithms, CertifiedUpperBracketOptions,
    CertifiedUpperBracketStatus,
};
use std::cell::{Cell, RefCell};

pub fn run<R, C>(client: &C) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R> + CertifiedUpperBracketAlgorithms<R>,
{
    let defaults = CertifiedUpperBracketOptions::default();
    assert_eq!(defaults.max_expansions, 4096);
    assert_eq!(defaults.max_points, 1 << 20);
    for (status, code) in [
        (CertifiedUpperBracketStatus::Bracketed, 0),
        (CertifiedUpperBracketStatus::NoRoot, 1),
        (CertifiedUpperBracketStatus::Unresolved, 2),
        (CertifiedUpperBracketStatus::WorkLimit, 3),
    ] {
        assert_eq!(status as i64, code);
    }
    for dtype in [DType::F32, DType::F64] {
        for strided in [false, true] {
            mixed_rows(client, dtype, strided)?;
        }
        uncertain_bounds(client, dtype)?;
        endpoints(client, dtype)?;
        represented_stagnation(client, dtype)?;
        extreme_bounds(client, dtype)?;
        tiny_scale(client, dtype)?;
    }
    Ok(())
}

fn represented_stagnation<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R> + CertifiedUpperBracketAlgorithms<R>,
{
    let ulp = if dtype == DType::F32 {
        f64::from(f32::EPSILON)
    } else {
        f64::EPSILON
    };
    let lower = tensor::<R>(&[1.0], dtype, client.device())?;
    let cap = tensor::<R>(&[1.0 + ulp], dtype, client.device())?;
    let result = client.bracket_upper_certified(
        |query| {
            affine(
                client,
                &query.point,
                &constant(client, &query.point, 1.0 + ulp)?,
            )
        },
        &lower,
        &lower,
        &cap,
        &CertifiedUpperBracketOptions::default(),
    )?;
    assert_eq!(result.status().try_to_vec::<i64>()?, [0]);
    assert_eq!(result.expansions().try_to_vec::<i64>()?, [1]);
    assert_eq!(read(result.lower())?, [1.0]);
    assert_eq!(read(result.upper())?, [1.0 + ulp]);
    assert_eq!(result.callback_points(), 3);
    Ok(())
}

fn mixed_rows<R, C>(client: &C, dtype: DType, strided: bool) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R> + CertifiedUpperBracketAlgorithms<R>,
{
    let lower = input::<R>(&[0.0; 5], dtype, client.device(), strided)?;
    let seed = input::<R>(&[1.0; 5], dtype, client.device(), strided)?;
    let cap = input::<R>(&[16.0; 5], dtype, client.device(), strided)?;
    // Owner order deliberately differs from root order.
    let roots = tensor::<R>(&[5.0, 0.5, 2.0, -1.0, 20.0], dtype, client.device())?;
    let traces = RefCell::new(Vec::new());
    let count = Cell::new(0usize);
    let result = client.bracket_upper_certified(
        |query| {
            assert_eq!(query.owner.dtype(), DType::I64);
            assert_eq!(query.owner.shape(), query.point.shape());
            assert_eq!(query.point.dtype(), dtype);
            assert!(query.point.device().is_same(client.device()));
            assert!(query.owner.device().is_same(client.device()));
            count.set(count.get() + query.point.numel());
            traces
                .borrow_mut()
                .push((query.owner.try_to_vec::<i64>()?, read(&query.point)?));
            affine(
                client,
                &query.point,
                &client.index_select(&roots, 0, &query.owner)?,
            )
        },
        &lower,
        &seed,
        &cap,
        &CertifiedUpperBracketOptions::default(),
    )?;
    assert_eq!(result.status().try_to_vec::<i64>()?, [0, 0, 0, 1, 1]);
    assert_eq!(result.expansions().dtype(), DType::I64);
    assert_eq!(result.expansions().try_to_vec::<i64>()?, [3, 0, 1, 0, 4]);
    assert_eq!(read(result.lower())?, [0.0; 5]);
    assert_eq!(read(result.upper())?, [8.0, 1.0, 2.0, 1.0, 16.0]);
    assert_eq!(result.expansion_rounds(), 4);
    assert_eq!(result.callback_points(), 18);
    assert_eq!(count.get(), 18);
    for value in [
        result.lower(),
        result.upper(),
        &result.lower_residual().lower,
        &result.lower_residual().upper,
        &result.upper_residual().lower,
        &result.upper_residual().upper,
    ] {
        assert_eq!(value.shape(), &[5]);
        assert_eq!(value.dtype(), dtype);
        assert!(value.device().is_same(client.device()));
    }
    let mut per_owner = vec![Vec::new(); 5];
    for (owners, points) in traces.borrow().iter() {
        for (&owner, &point) in owners.iter().zip(points) {
            assert!((0..5).contains(&owner));
            assert!((0.0..=16.0).contains(&point));
            per_owner[owner as usize].push(point);
        }
    }
    for (actual, expected) in per_owner.iter_mut().zip([
        vec![0.0, 1.0, 2.0, 4.0, 8.0],
        vec![0.0, 1.0],
        vec![0.0, 1.0, 2.0],
        vec![0.0, 1.0],
        vec![0.0, 1.0, 2.0, 4.0, 8.0, 16.0],
    ]) {
        actual.sort_by(f64::total_cmp);
        assert_eq!(*actual, expected);
    }
    let lo = read(&result.lower_residual().lower)?;
    let hi = read(&result.lower_residual().upper)?;
    let ulo = read(&result.upper_residual().lower)?;
    let uhi = read(&result.upper_residual().upper)?;
    for (i, root) in [5.0, 0.5, 2.0, -1.0, 20.0].into_iter().enumerate() {
        assert!(lo[i] <= -root && -root <= hi[i]);
        let value = [8.0, 1.0, 2.0, 1.0, 16.0][i] - root;
        assert!(ulo[i] <= value && value <= uhi[i]);
    }
    Ok(())
}

fn uncertain_bounds<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R> + CertifiedUpperBracketAlgorithms<R>,
{
    let lower = tensor::<R>(&[0.0; 3], dtype, client.device())?;
    let seed = tensor::<R>(&[1.0; 3], dtype, client.device())?;
    let cap = tensor::<R>(&[4.0; 3], dtype, client.device())?;
    let result = client.bracket_upper_certified(
        |query| {
            let physical = client.sub_scalar(&query.point, 2.0)?;
            let owner = |value| {
                Tensor::full_scalar(query.owner.shape(), DType::I64, value, client.device())
            };
            let first = client.eq(&query.owner, &owner(0.0)?)?;
            // Row zero encloses x-2. Remaining rows enclose the constant physical function zero.
            Ok(CertifiedRootValues {
                lower: client.where_cond(
                    &first,
                    &client.sub_scalar(&physical, 2.0)?,
                    &constant(client, &query.point, -1.0)?,
                )?,
                upper: client.where_cond(
                    &first,
                    &client.add_scalar(&physical, 2.0)?,
                    &constant(client, &query.point, 0.0)?,
                )?,
            })
        },
        &lower,
        &seed,
        &cap,
        &CertifiedUpperBracketOptions::default(),
    )?;
    assert_eq!(result.status().try_to_vec::<i64>()?, [0, 2, 2]);
    assert_eq!(result.expansions().try_to_vec::<i64>()?, [2, 2, 2]);
    assert_eq!(read(result.lower())?, [0.0; 3]);
    assert_eq!(read(result.upper())?, [4.0; 3]);
    Ok(())
}

fn endpoints<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R> + CertifiedUpperBracketAlgorithms<R>,
{
    let lower = tensor::<R>(&[0.0, 0.0, 2.0, 2.0, 0.0], dtype, client.device())?;
    let seed = tensor::<R>(&[1.0, 1.0, 2.0, 2.0, 0.0], dtype, client.device())?;
    let cap = tensor::<R>(&[4.0, 4.0, 2.0, 2.0, 1.0], dtype, client.device())?;
    let tiny = if dtype == DType::F32 {
        f64::from(f32::from_bits(1))
    } else {
        f64::from_bits(1)
    };
    let roots = tensor::<R>(&[0.0, 1.0, 2.0, 3.0, tiny], dtype, client.device())?;
    let result = client.bracket_upper_certified(
        |query| {
            affine(
                client,
                &query.point,
                &client.index_select(&roots, 0, &query.owner)?,
            )
        },
        &lower,
        &seed,
        &cap,
        &CertifiedUpperBracketOptions::default(),
    )?;
    assert_eq!(result.status().try_to_vec::<i64>()?, [0, 0, 0, 1, 0]);
    assert_eq!(read(result.lower())?, [0.0, 0.0, 2.0, 2.0, 0.0]);
    assert_eq!(read(result.upper())?, [1.0, 1.0, 2.0, 2.0, tiny]);
    assert_eq!(result.expansions().try_to_vec::<i64>()?, [0, 0, 0, 0, 1]);
    assert_eq!(result.callback_points(), 11);
    // A represented zero-width seed requires one next-up step.
    Ok(())
}

fn extreme_bounds<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R> + CertifiedUpperBracketAlgorithms<R>,
{
    let max = if dtype == DType::F32 {
        f64::from(f32::MAX)
    } else {
        f64::MAX
    };
    let lower = tensor::<R>(&[-max], dtype, client.device())?;
    let seed = tensor::<R>(&[-max * 0.5], dtype, client.device())?;
    let cap = tensor::<R>(&[max], dtype, client.device())?;
    let result = client.bracket_upper_certified(
        |query| {
            for point in read(&query.point)? {
                assert!(point.is_finite() && -max <= point && point <= max);
            }
            // f(x)=x/2 has exact scaling and finite residuals at both finite extrema.
            let value = client.mul_scalar(&query.point, 0.5)?;
            Ok(CertifiedRootValues {
                lower: value.clone(),
                upper: value,
            })
        },
        &lower,
        &seed,
        &cap,
        &CertifiedUpperBracketOptions::default(),
    )?;
    assert_eq!(result.status().try_to_vec::<i64>()?, [0]);
    assert_eq!(read(result.lower())?, [-max]);
    let upper = read(result.upper())?[0];
    assert!(upper >= 0.0 && upper <= max);
    Ok(())
}

fn tiny_scale<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R> + CertifiedUpperBracketAlgorithms<R>,
{
    let (root, initial, rounds) = if dtype == DType::F64 {
        (2.0f64.powi(-600), 2.0f64.powi(-1000), 400)
    } else {
        (2.0f64.powi(-60), 2.0f64.powi(-120), 60)
    };
    let lower = tensor::<R>(&[0.0], dtype, client.device())?;
    let seed = tensor::<R>(&[initial], dtype, client.device())?;
    let cap = tensor::<R>(&[root * 2.0], dtype, client.device())?;
    let result = client.bracket_upper_certified(
        |query| affine(client, &query.point, &constant(client, &query.point, root)?),
        &lower,
        &seed,
        &cap,
        &CertifiedUpperBracketOptions::default(),
    )?;
    assert_eq!(result.status().try_to_vec::<i64>()?, [0]);
    assert_eq!(read(result.lower())?, [0.0]);
    assert_eq!(read(result.upper())?, [root]);
    assert_eq!(result.expansion_rounds(), rounds);
    assert_eq!(result.expansions().try_to_vec::<i64>()?, [rounds as i64]);
    assert_eq!(result.callback_points(), rounds + 2);
    assert_eq!(read(&result.upper_residual().lower)?, [0.0]);
    assert_eq!(read(&result.upper_residual().upper)?, [0.0]);
    Ok(())
}
