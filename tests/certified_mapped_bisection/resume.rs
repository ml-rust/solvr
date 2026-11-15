use super::fixture::{self, Client, Result, tensor};
use numr::prelude::*;
use solvr::optimize::roots::{
    CertifiedBisectionAlgorithms, CertifiedBisectionOptions, CertifiedMappedBisectionAlgorithms,
    CertifiedRootAdvanceAllowance, CertifiedRootTolerance,
};
use std::cell::{Cell, RefCell};

pub fn run<R, C>(client: &C) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R> + CertifiedBisectionAlgorithms<R> + CertifiedMappedBisectionAlgorithms<R>,
{
    for dtype in [DType::F32, DType::F64] {
        contracted_existence_survives_resume(client, dtype)?;
        original_ceilings_survive_allowances(client, dtype)?;
    }
    Ok(())
}

fn contracted_existence_survives_resume<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R> + CertifiedBisectionAlgorithms<R> + CertifiedMappedBisectionAlgorithms<R>,
{
    let lower = tensor::<R>(&[0.0], &[1], dtype, client.device())?;
    let upper = tensor::<R>(&[1.0], &[1], dtype, client.device())?;
    let derivative = tensor::<R>(&[1.0], &[1], dtype, client.device())?;
    let progress = client.bisect_certified_mapped(
        |query| fixture::uncertain_affine(client, query),
        fixture::identity,
        &lower,
        &upper,
        &derivative,
        &CertifiedBisectionOptions {
            atol: 0.5,
            rtol: 0.0,
            max_iterations: 8,
            max_points: 32,
        },
    )?;
    assert_eq!(progress.result().status().try_to_vec::<i64>()?, [0]);
    assert_eq!(progress.result().iterations(), 1);
    let retained_lower = fixture::read(progress.result().root_lower())?[0];
    let retained_upper = fixture::read(progress.result().root_upper())?[0];
    assert!(retained_lower > 0.0 && retained_upper < 1.0);
    fixture::contains(progress.result(), &[0.375])?;

    // These endpoints enclose the unique root, but their oracle signs cannot re-establish existence.
    let restarted = client.bisect_certified(
        |query| fixture::uncertain_affine(client, query),
        progress.result().root_lower(),
        progress.result().root_upper(),
        &derivative,
        &CertifiedBisectionOptions {
            atol: 0.0,
            rtol: 0.0,
            max_iterations: 8,
            max_points: 32,
        },
    )?;
    assert_eq!(restarted.status().try_to_vec::<i64>()?, [2]);
    assert_eq!(restarted.root_owner().numel(), 0);

    let before_points = progress.result().evaluated_points();
    let before_rounds = progress.result().iterations();
    let (_, continuation) = progress.into_parts();
    let queried = RefCell::new(Vec::new());
    let resumed = client.resume_certified_mapped(
        |query| {
            queried.borrow_mut().push(query.point.clone());
            fixture::uncertain_affine(client, query)
        },
        fixture::identity,
        continuation,
        &CertifiedRootTolerance {
            atol: 0.0,
            rtol: 0.0,
        },
        &CertifiedRootAdvanceAllowance {
            max_iterations: 4,
            max_points: 8,
        },
    )?;
    assert_eq!(resumed.result().root_owner().try_to_vec::<i64>()?, [0]);
    assert_eq!(resumed.result().status().try_to_vec::<i64>()?, [3]);
    fixture::contains(resumed.result(), &[0.375])?;
    assert!(resumed.result().iterations() > before_rounds);
    let mut points = 0;
    for query in queried.borrow().iter() {
        let values = fixture::read(query)?;
        points += values.len();
        assert!(
            values
                .iter()
                .all(|&x| retained_lower < x && x < retained_upper)
        );
    }
    assert!(points > 0);
    assert_eq!(resumed.result().evaluated_points(), before_points + points);
    Ok(())
}

fn original_ceilings_survive_allowances<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R> + CertifiedMappedBisectionAlgorithms<R>,
{
    let lower = tensor::<R>(&[1.0], &[1], dtype, client.device())?;
    let upper = tensor::<R>(&[2.0], &[1], dtype, client.device())?;
    let derivative = tensor::<R>(&[2.0], &[1], dtype, client.device())?;
    let points = Cell::new(0usize);
    let progress = client.bisect_certified_mapped(
        |query| {
            points.set(points.get() + query.point.numel());
            fixture::quadratic(client, query)
        },
        fixture::identity,
        &lower,
        &upper,
        &derivative,
        &CertifiedBisectionOptions {
            atol: 1.0,
            rtol: 0.0,
            max_iterations: 2,
            max_points: 4,
        },
    )?;
    assert_eq!(progress.result().iterations(), 0);
    assert_eq!(progress.result().evaluated_points(), 2);
    let (_, continuation) = progress.into_parts();
    let stopped = client.resume_certified_mapped(
        |query| {
            points.set(points.get() + query.point.numel());
            fixture::quadratic(client, query)
        },
        fixture::identity,
        continuation,
        &CertifiedRootTolerance {
            atol: 0.0,
            rtol: 0.0,
        },
        &CertifiedRootAdvanceAllowance {
            max_iterations: 0,
            max_points: 0,
        },
    )?;
    assert_eq!(stopped.result().status().try_to_vec::<i64>()?, [4]);
    assert_eq!(stopped.result().iterations(), 0);
    assert_eq!(points.get(), 2);
    let (_, continuation) = stopped.into_parts();
    let advanced = client.resume_certified_mapped(
        |query| {
            points.set(points.get() + query.point.numel());
            fixture::quadratic(client, query)
        },
        fixture::identity,
        continuation,
        &CertifiedRootTolerance {
            atol: 0.0,
            rtol: 0.0,
        },
        &CertifiedRootAdvanceAllowance {
            max_iterations: 1,
            max_points: 1,
        },
    )?;
    assert_eq!(advanced.result().status().try_to_vec::<i64>()?, [4]);
    assert_eq!(advanced.result().iterations(), 1);
    assert_eq!(advanced.result().evaluated_points(), 3);
    let (_, continuation) = advanced.into_parts();
    let exhausted = client.resume_certified_mapped(
        |query| {
            points.set(points.get() + query.point.numel());
            fixture::quadratic(client, query)
        },
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
    assert_eq!(exhausted.result().status().try_to_vec::<i64>()?, [4]);
    assert_eq!(exhausted.result().iterations(), 2);
    assert_eq!(exhausted.result().evaluated_points(), 4);
    fixture::contains(exhausted.result(), &[2.0_f64.sqrt()])?;
    let before = fixture::snapshot(exhausted.result())?;
    let (_, continuation) = exhausted.into_parts();
    let repeated = client.resume_certified_mapped(
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
    assert_eq!(fixture::snapshot(repeated.result())?, before);
    assert_eq!(points.get(), 4);
    Ok(())
}
