use super::fixture::{self, Client, Result, constant, tensor};
use numr::prelude::*;
use solvr::optimize::roots::{
    CertifiedBisectionAlgorithms, CertifiedBisectionOptions, CertifiedRootValues,
};
use std::cell::{Cell, RefCell};

pub fn run<R, C>(client: &C) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R> + CertifiedBisectionAlgorithms<R>,
{
    for dtype in [DType::F32, DType::F64] {
        pending_batch_budget(client, dtype)?;
        repeated_candidate(client, dtype)?;
        let zero = tensor::<R>(&[0.0], &[1], dtype, client.device())?;
        let one = tensor::<R>(&[1.0], &[1], dtype, client.device())?;
        for max_points in [0, 1] {
            let called = Cell::new(false);
            let result = client.bisect_certified(
                |_| {
                    called.set(true);
                    Err(fixture::sentinel())
                },
                &zero,
                &zero,
                &one,
                &CertifiedBisectionOptions {
                    max_points,
                    ..Default::default()
                },
            );
            assert!(
                result.is_err(),
                "initial 2*N reservation accepted {max_points} points"
            );
            assert!(!called.get());
        }
        for count in [
            usize::MAX / 2 + 1,
            usize::MAX / dtype.size_in_bytes() + 1,
            (u32::MAX as usize) / 2 + 1,
        ] {
            let lower = zero.broadcast_to(&[count])?;
            let upper = one.broadcast_to(&[count])?;
            let called = Cell::new(false);
            let result = client.bisect_certified(
                |_| {
                    called.set(true);
                    Err(fixture::sentinel())
                },
                &lower,
                &upper,
                &upper,
                &CertifiedBisectionOptions {
                    max_points: usize::MAX,
                    ..Default::default()
                },
            );
            assert!(
                result.is_err(),
                "virtual count {count} reached allocation or work"
            );
            assert!(!called.get());
        }
    }
    Ok(())
}

fn pending_batch_budget<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R> + CertifiedBisectionAlgorithms<R>,
{
    let lower = tensor::<R>(&[0.0, 0.0], &[2], dtype, client.device())?;
    let upper = tensor::<R>(&[1.0, 1.0], &[2], dtype, client.device())?;
    let points = Cell::new(0usize);
    let queries = RefCell::new(Vec::new());
    let result = client.bisect_certified(
        |query| {
            points.set(points.get() + query.point.numel());
            queries.borrow_mut().push(query.point.clone());
            // f(x)=x-1/2 lies inside [x-1,x] and has derivative one.
            Ok(CertifiedRootValues {
                lower: super::callback::affine(client, &query.point, 1.0)?.lower,
                upper: query.point.clone(),
            })
        },
        &lower,
        &upper,
        &upper,
        &CertifiedBisectionOptions {
            atol: 0.0,
            rtol: 0.0,
            max_iterations: 4,
            max_points: 5,
        },
    )?;
    assert_eq!(result.status().try_to_vec::<i64>()?, [4, 4]);
    assert_eq!(result.root_owner().try_to_vec::<i64>()?, [0, 1]);
    assert_eq!(fixture::read(result.root_lower())?, [0.0, 0.0]);
    assert_eq!(fixture::read(result.root_upper())?, [1.0, 1.0]);
    assert_eq!(result.evaluated_points(), 4);
    assert_eq!(points.get(), 4);
    assert_eq!(result.iterations(), 0);
    for query in queries.borrow().iter() {
        assert!(
            fixture::read(query)?
                .iter()
                .all(|&point| point == 0.0 || point == 1.0)
        );
    }
    Ok(())
}

fn repeated_candidate<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R> + CertifiedBisectionAlgorithms<R>,
{
    let (ulp, eta) = if dtype == DType::F32 {
        (f64::from(f32::EPSILON), f64::from(f32::from_bits(1)))
    } else {
        (f64::EPSILON, f64::from_bits(1))
    };
    let root = 1.0 + 2.0 * ulp;
    let lower = tensor::<R>(&[1.0], &[1], dtype, client.device())?;
    let upper = tensor::<R>(&[1.0 + 4.0 * ulp], &[1], dtype, client.device())?;
    let derivative = tensor::<R>(&[1.0], &[1], dtype, client.device())?;
    let points = Cell::new(0usize);
    let queries = RefCell::new(Vec::new());
    let result = client.bisect_certified(
        |query| {
            points.set(points.get() + query.point.numel());
            queries.borrow_mut().push(query.point.clone());
            let root_tensor = constant(client, &query.point, root)?;
            // Sterbenz subtraction is exact throughout [1,1+4 ULP].
            let residual = client.sub(&query.point, &root_tensor)?;
            let at_root = client.eq(&query.point, &root_tensor)?;
            // The fixed oracle encloses f(x)=x-root with derivative one.
            Ok(CertifiedRootValues {
                lower: client.where_cond(
                    &at_root,
                    &constant(client, &query.point, -2.0 * ulp)?,
                    &residual,
                )?,
                upper: client.where_cond(
                    &at_root,
                    &constant(client, &query.point, eta)?,
                    &residual,
                )?,
            })
        },
        &lower,
        &upper,
        &derivative,
        &CertifiedBisectionOptions {
            atol: 0.0,
            rtol: 0.0,
            max_iterations: 8,
            max_points: 10,
        },
    )?;
    assert_eq!(result.status().try_to_vec::<i64>()?, [3]);
    assert_eq!(result.root_owner().try_to_vec::<i64>()?, [0]);
    assert_eq!(fixture::read(result.root_lower())?, [1.0 + ulp]);
    assert_eq!(fixture::read(result.root_upper())?, [1.0 + 4.0 * ulp]);
    assert_eq!(result.evaluated_points(), 3);
    assert_eq!(points.get(), 3);
    assert_eq!(result.iterations(), 2);
    let mut sampled = Vec::new();
    for query in queries.borrow().iter() {
        sampled.extend(fixture::read(query)?);
    }
    sampled.sort_by(f64::total_cmp);
    assert_eq!(sampled, [1.0, root, 1.0 + 4.0 * ulp]);
    Ok(())
}
