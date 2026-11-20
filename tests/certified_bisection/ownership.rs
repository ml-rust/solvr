use super::fixture::{self, Client, Result, constant, tensor};
use numr::prelude::*;
use solvr::optimize::roots::{
    CertifiedBisectionAlgorithms, CertifiedBisectionOptions, CertifiedRootValues,
};
use std::cell::{Cell, RefCell};

fn inputs<R, C>(
    client: &C,
    values: &[f64],
    dtype: DType,
    strided: bool,
) -> numr::error::Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    if strided {
        let paired: Vec<_> = values.iter().flat_map(|&value| [19.0, value]).collect();
        return Ok(
            tensor::<R>(&paired, &[values.len(), 2], dtype, client.device())?
                .narrow(1, 1, 1)?
                .squeeze(Some(1)),
        );
    }
    tensor(values, &[values.len()], dtype, client.device())
}

pub fn run<R, C>(client: &C) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R> + CertifiedBisectionAlgorithms<R>,
{
    for dtype in [DType::F32, DType::F64] {
        input_snapshot(client, dtype)?;
        for strided in [false, true] {
            let lower = inputs(client, &[-0.0, 0.0, 0.0, 0.0, 0.0], dtype, strided)?;
            let upper = inputs(client, &[1.0; 5], dtype, strided)?;
            let derivative = inputs(client, &[1.0; 5], dtype, strided)?;
            let roots = tensor::<R>(
                &[0.5, -1.0, -0.25, 0.75, 0.25],
                &[5],
                dtype,
                client.device(),
            )?;
            let shared_lower = Tensor::<R>::empty(&[10], dtype, client.device())?;
            let shared_upper = Tensor::<R>::empty(&[10], dtype, client.device())?;
            let queried = RefCell::new(Vec::new());
            let point_count = Cell::new(0usize);
            let before = [
                fixture::bits(client, &lower)?,
                fixture::bits(client, &upper)?,
                fixture::bits(client, &derivative)?,
            ];
            let result = client.bisect_certified(
                |query| {
                    point_count.set(point_count.get() + query.point.numel());
                    queried
                        .borrow_mut()
                        .push((query.owner.clone(), query.point.clone()));
                    let x = &query.point;
                    let root = client.index_select(&roots, 0, &query.owner)?;
                    let value = client.sub(x, &root)?;
                    let zero = constant(client, x, 0.0)?;
                    let exact = client.eq(x, &root)?;
                    let lo = client.where_cond(&exact, &zero, &client.next_down(&value)?)?;
                    let hi = client.where_cond(&exact, &zero, &client.next_up(&value)?)?;
                    let owner = |value| {
                        Tensor::full_scalar(query.owner.shape(), DType::I64, value, client.device())
                    };
                    let uncertain = client.eq(&query.owner, &owner(2.0)?)?;
                    let false_zero = client.eq(&query.owner, &owner(3.0)?)?;
                    let bounded = client.eq(&query.owner, &owner(4.0)?)?;
                    let uncertain_lo = client.next_down(&client.add_scalar(&lo, -0.5)?)?;
                    let uncertain_hi = client.next_up(&client.add_scalar(&hi, 0.5)?)?;
                    let domain_lo = super::callback::affine(client, x, 1.0)?.lower;
                    let bounded_lo = client.mul_scalar(&client.minimum(&lo, &zero)?, 2.0)?;
                    let bounded_hi = client.mul_scalar(&client.maximum(&hi, &zero)?, 2.0)?;
                    let lo = client.where_cond(&uncertain, &uncertain_lo, &lo)?;
                    let hi = client.where_cond(&uncertain, &uncertain_hi, &hi)?;
                    let lo = client.where_cond(&false_zero, &domain_lo, &lo)?;
                    let hi = client.where_cond(&false_zero, x, &hi)?;
                    let lo = client.where_cond(&bounded, &bounded_lo, &lo)?;
                    let hi = client.where_cond(&bounded, &bounded_hi, &hi)?;
                    let returned_lower = shared_lower.narrow(0, 0, x.numel())?;
                    let returned_upper = shared_upper.narrow(0, 0, x.numel())?;
                    client.copy_into(&returned_lower, &lo)?;
                    client.copy_into(&returned_upper, &hi)?;
                    Ok(CertifiedRootValues {
                        lower: returned_lower,
                        upper: returned_upper,
                    })
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
            assert_eq!(result.status().try_to_vec::<i64>()?, [0, 1, 2, 3, 4]);
            assert_eq!(result.root_owner().try_to_vec::<i64>()?, [0, 3, 4]);
            assert_eq!(result.root_owner().shape(), &[3]);
            assert_eq!(result.root_lower().shape(), &[3]);
            assert_eq!(result.root_upper().shape(), &[3]);
            let lo = fixture::read(result.root_lower())?;
            let hi = fixture::read(result.root_upper())?;
            for (index, root) in [0.5, 0.75, 0.25].into_iter().enumerate() {
                assert!(lo[index] <= root && root <= hi[index]);
            }
            assert_eq!((lo[0], hi[0]), (0.5, 0.5));
            assert_eq!(result.iterations(), 1);
            assert_eq!(result.evaluated_points(), point_count.get());
            assert!(result.evaluated_points() <= 13);
            let mut midpoint_owners = Vec::new();
            for (owner, point) in queried.borrow().iter() {
                let owners = owner.try_to_vec::<i64>()?;
                let points = fixture::read(point)?;
                assert_eq!(owners.len(), points.len());
                for (&owner, &x) in owners.iter().zip(&points) {
                    assert!((0..5).contains(&owner));
                    assert!((0.0..=1.0).contains(&x));
                    if x > 0.0 && x < 1.0 {
                        midpoint_owners.push(owner);
                    }
                }
            }
            midpoint_owners.sort_unstable();
            assert_eq!(midpoint_owners, [0, 3, 4]);
            assert_eq!(fixture::bits(client, &lower)?, before[0]);
            assert_eq!(fixture::bits(client, &upper)?, before[1]);
            assert_eq!(fixture::bits(client, &derivative)?, before[2]);
            let saved = [
                fixture::bits(client, result.root_lower())?,
                fixture::bits(client, result.root_upper())?,
            ];
            for alias in [&shared_lower, &shared_upper, &lower, &upper, &derivative] {
                let zeros = Tensor::zeros(alias.shape(), dtype, client.device())?;
                client.copy_into(alias, &zeros)?;
            }
            assert_eq!(fixture::bits(client, result.root_lower())?, saved[0]);
            assert_eq!(fixture::bits(client, result.root_upper())?, saved[1]);
            assert_eq!(result.status().try_to_vec::<i64>()?, [0, 1, 2, 3, 4]);
            assert_eq!(result.root_owner().try_to_vec::<i64>()?, [0, 3, 4]);
        }
    }
    Ok(())
}

fn input_snapshot<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R> + CertifiedBisectionAlgorithms<R>,
{
    let lower = tensor::<R>(&[0.0], &[1], dtype, client.device())?;
    let upper = tensor::<R>(&[1.0], &[1], dtype, client.device())?;
    let derivative = tensor::<R>(&[1.0], &[1], dtype, client.device())?;
    let calls = Cell::new(0usize);
    let queries = RefCell::new(Vec::new());
    let result = client.bisect_certified(
        |query| {
            queries.borrow_mut().push(query.point.clone());
            let output = super::callback::affine(client, &query.point, 0.5)?;
            if calls.get() == 0 {
                // Caller storage changes after entry. The physical function stays fixed.
                let replacement = constant(client, &lower, 9.0)?;
                for value in [&lower, &upper, &derivative] {
                    client.copy_into(value, &replacement)?;
                }
            }
            calls.set(calls.get() + 1);
            Ok(output)
        },
        &lower,
        &upper,
        &derivative,
        &CertifiedBisectionOptions {
            atol: 0.0,
            rtol: 0.0,
            max_iterations: 1,
            max_points: 3,
            probe: Default::default(),
        },
    )?;
    assert_eq!(result.status().try_to_vec::<i64>()?, [0]);
    assert_eq!(fixture::read(result.root_lower())?, [0.5]);
    assert_eq!(fixture::read(result.root_upper())?, [0.5]);
    assert_eq!(fixture::read(&lower)?, [9.0]);
    assert_eq!(fixture::read(&upper)?, [9.0]);
    assert_eq!(fixture::read(&derivative)?, [9.0]);
    let mut points = Vec::new();
    for query in queries.borrow().iter() {
        points.extend(fixture::read(query)?);
    }
    points.sort_by(f64::total_cmp);
    assert_eq!(points, [0.0, 0.5, 1.0]);
    Ok(())
}
