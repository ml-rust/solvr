use super::fixture::{self, Client, Result, constant, tensor};
use numr::prelude::*;
use solvr::optimize::roots::{
    CertifiedBisectionOptions, CertifiedMappedBisectionAlgorithms, CertifiedRootAdvanceAllowance,
    CertifiedRootQuery, CertifiedRootTolerance, CertifiedRootValues,
};
use std::cell::{Cell, RefCell};

pub fn run<R, C>(client: &C) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R> + CertifiedMappedBisectionAlgorithms<R>,
{
    for dtype in [DType::F32, DType::F64] {
        private_state_survives_alias_mutation(client, dtype)?;
        for strided in [false, true] {
            mixed_owners_keep_permanent_floors(client, dtype, strided)?;
        }
    }
    Ok(())
}

fn private_state_survives_alias_mutation<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R> + CertifiedMappedBisectionAlgorithms<R>,
{
    let lower = tensor::<R>(&[1.0], &[1], dtype, client.device())?;
    let upper = tensor::<R>(&[2.0], &[1], dtype, client.device())?;
    let derivative = tensor::<R>(&[2.0], &[1], dtype, client.device())?;
    let residual_queries = RefCell::new(Vec::new());
    let residual_values = RefCell::new(Vec::new());
    let mapped_queries = RefCell::new(Vec::new());
    let progress = client.bisect_certified_mapped(
        |query| {
            residual_queries
                .borrow_mut()
                .push((query.owner.clone(), query.point.clone()));
            let value = fixture::quadratic(client, query)?;
            residual_values
                .borrow_mut()
                .push((value.lower.clone(), value.upper.clone()));
            Ok(value)
        },
        |query| {
            mapped_queries.borrow_mut().push((
                query.owner.clone(),
                query.lower.clone(),
                query.upper.clone(),
            ));
            fixture::identity(query)
        },
        &lower,
        &upper,
        &derivative,
        &CertifiedBisectionOptions {
            atol: 2.0,
            rtol: 0.0,
            max_iterations: 64,
            max_points: 128,
            probe: Default::default(),
        },
    )?;
    assert_eq!(progress.result().iterations(), 0);
    let (snapshot, continuation) = progress.into_parts();
    for value in [
        &lower,
        &upper,
        &derivative,
        snapshot.status(),
        snapshot.root_owner(),
        snapshot.root_lower(),
        snapshot.root_upper(),
    ] {
        overwrite(client, value, 99.0)?;
    }
    for (owner, point) in residual_queries.borrow().iter() {
        overwrite(client, owner, 77.0)?;
        overwrite(client, point, 77.0)?;
    }
    for (lower, upper) in residual_values.borrow().iter() {
        overwrite(client, lower, 66.0)?;
        overwrite(client, upper, 66.0)?;
    }
    for (owner, lower, upper) in mapped_queries.borrow().iter() {
        overwrite(client, owner, 88.0)?;
        overwrite(client, lower, 88.0)?;
        overwrite(client, upper, 88.0)?;
    }
    let resumed = client.resume_certified_mapped(
        |query| fixture::quadratic(client, query),
        fixture::identity,
        continuation,
        &CertifiedRootTolerance {
            atol: 1e-4,
            rtol: 0.0,
        },
        &CertifiedRootAdvanceAllowance {
            max_iterations: 64,
            max_points: 128,
        },
    )?;
    assert_eq!(resumed.result().status().try_to_vec::<i64>()?, [0]);
    assert_eq!(resumed.result().root_owner().try_to_vec::<i64>()?, [0]);
    fixture::contains(resumed.result(), &[2.0_f64.sqrt()])?;
    assert!(resumed.result().iterations() <= 64);
    assert!(resumed.result().evaluated_points() <= 128);
    Ok(())
}

fn overwrite<R, C>(client: &C, value: &Tensor<R>, replacement: f64) -> numr::error::Result<()>
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    let replacement =
        Tensor::full_scalar(value.shape(), value.dtype(), replacement, client.device())?;
    client.copy_into(value, &replacement)
}

fn input<R, C>(
    client: &C,
    value: f64,
    dtype: DType,
    strided: bool,
) -> numr::error::Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    if strided {
        return Ok(tensor::<R>(
            &[
                19.0, value, 19.0, value, 19.0, value, 19.0, value, 19.0, value,
            ],
            &[5, 2],
            dtype,
            client.device(),
        )?
        .narrow(1, 1, 1)?
        .squeeze(Some(1)));
    }
    tensor(&[value; 5], &[5], dtype, client.device())
}

fn mixed_owners_keep_permanent_floors<R, C>(client: &C, dtype: DType, strided: bool) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R> + CertifiedMappedBisectionAlgorithms<R>,
{
    let lower = input(client, 0.0, dtype, strided)?;
    let upper = input(client, 1.0, dtype, strided)?;
    let derivative = input(client, 1.0, dtype, strided)?;
    let roots = tensor::<R>(
        &[0.5, -1.0, -0.25, 0.75, 0.25],
        &[5],
        dtype,
        client.device(),
    )?;
    let points = Cell::new(0usize);
    let mapped_owners = RefCell::new(Vec::new());
    let progress = client.bisect_certified_mapped(
        |query| {
            points.set(points.get() + query.point.numel());
            mixed_callback(client, &roots, query)
        },
        |query| {
            mapped_owners.borrow_mut().push(query.owner.clone());
            fixture::identity(query)
        },
        &lower,
        &upper,
        &derivative,
        &CertifiedBisectionOptions {
            atol: 0.0,
            rtol: -0.0,
            max_iterations: 1,
            max_points: 13,
            probe: Default::default(),
        },
    )?;
    assert_eq!(
        progress.result().status().try_to_vec::<i64>()?,
        [0, 1, 2, 3, 4]
    );
    assert_eq!(
        progress.result().root_owner().try_to_vec::<i64>()?,
        [0, 3, 4]
    );
    fixture::contains(progress.result(), &[0.5, 0.75, 0.25])?;
    let saved = fixture::snapshot(progress.result())?;
    let (snapshot, continuation) = progress.into_parts();
    let loose = client.resume_certified_mapped(
        |_| Err(fixture::sentinel()),
        fixture::identity,
        continuation,
        &CertifiedRootTolerance {
            atol: 2.0,
            rtol: 0.0,
        },
        &CertifiedRootAdvanceAllowance {
            max_iterations: usize::MAX,
            max_points: usize::MAX,
        },
    )?;
    assert_eq!(
        loose.result().status().try_to_vec::<i64>()?,
        [0, 1, 2, 0, 0]
    );
    assert_eq!(fixture::snapshot(&snapshot)?, saved);
    let (_, continuation) = loose.into_parts();
    let strict = client.resume_certified_mapped(
        |_| Err(fixture::sentinel()),
        fixture::identity,
        continuation,
        &CertifiedRootTolerance {
            atol: 0.0,
            rtol: 0.0,
        },
        &CertifiedRootAdvanceAllowance {
            max_iterations: usize::MAX,
            max_points: usize::MAX,
        },
    )?;
    assert_eq!(
        strict.result().status().try_to_vec::<i64>()?,
        [0, 1, 2, 3, 4]
    );
    assert_eq!(strict.result().root_owner().try_to_vec::<i64>()?, [0, 3, 4]);
    assert_eq!(strict.result().evaluated_points(), points.get());
    assert_eq!(strict.result().iterations(), saved.iterations);
    for owner in mapped_owners.borrow().iter() {
        assert!(
            owner
                .try_to_vec::<i64>()?
                .iter()
                .all(|row| [0, 3, 4].contains(row))
        );
    }
    Ok(())
}

fn mixed_callback<R, C>(
    client: &C,
    roots: &Tensor<R>,
    query: &CertifiedRootQuery<R>,
) -> numr::error::Result<CertifiedRootValues<R>>
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    let x = &query.point;
    let root = client.index_select(roots, 0, &query.owner)?;
    let value = client.sub(x, &root)?;
    let zero = constant(client, x, 0.0)?;
    let exact = client.eq(x, &root)?;
    let lo = client.where_cond(&exact, &zero, &client.next_down(&value)?)?;
    let hi = client.where_cond(&exact, &zero, &client.next_up(&value)?)?;
    let owner =
        |value| Tensor::full_scalar(query.owner.shape(), DType::I64, value, client.device());
    let uncertain = client.eq(&query.owner, &owner(2.0)?)?;
    let stalled = client.eq(&query.owner, &owner(3.0)?)?;
    let bounded = client.eq(&query.owner, &owner(4.0)?)?;
    let lo = client.where_cond(
        &uncertain,
        &client.next_down(&client.add_scalar(&lo, -0.5)?)?,
        &lo,
    )?;
    let hi = client.where_cond(
        &uncertain,
        &client.next_up(&client.add_scalar(&hi, 0.5)?)?,
        &hi,
    )?;
    let lo = client.where_cond(&stalled, &fixture::affine(client, x, 1.0)?.lower, &lo)?;
    let hi = client.where_cond(&stalled, x, &hi)?;
    Ok(CertifiedRootValues {
        lower: client.where_cond(
            &bounded,
            &client.mul_scalar(&client.minimum(&lo, &zero)?, 2.0)?,
            &lo,
        )?,
        upper: client.where_cond(
            &bounded,
            &client.mul_scalar(&client.maximum(&hi, &zero)?, 2.0)?,
            &hi,
        )?,
    })
}
