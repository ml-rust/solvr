use super::fixture::{self, Client, Result, constant, tensor};
use numr::prelude::*;
use solvr::optimize::roots::{
    CertifiedBisectionAlgorithms, CertifiedBisectionOptions, CertifiedMappedBisectionAlgorithms,
    CertifiedRootAdvanceAllowance, CertifiedRootMappedValues, CertifiedRootTolerance,
};
use std::cell::{Cell, RefCell};

pub fn run<R, C>(client: &C) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R> + CertifiedBisectionAlgorithms<R> + CertifiedMappedBisectionAlgorithms<R>,
{
    for dtype in [DType::F32, DType::F64] {
        identity_matches_legacy(client, dtype)?;
        early_stopping_resumes(client, dtype)?;
        mapped_width_controls_convergence(client, dtype)?;
        exact_parameter_keeps_mapping_uncertainty(client, dtype)?;
        inward_tolerances_do_not_round_up(client, dtype)?;
    }
    Ok(())
}

fn inward_tolerances_do_not_round_up<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R> + CertifiedMappedBisectionAlgorithms<R>,
{
    let previous = f64::from(f32::from_bits(0x3f7fffff));
    let tolerance = previous + (1.0 - previous) * 0.75;
    assert_eq!(tolerance as f32, 1.0);
    let point = tensor::<R>(&[0.5], &[1], dtype, client.device())?;
    let derivative = tensor::<R>(&[1.0], &[1], dtype, client.device())?;
    for relative in [false, true] {
        let progress = client.bisect_certified_mapped(
            |query| fixture::affine(client, &query.point, 0.5),
            |query| {
                Ok(CertifiedRootMappedValues {
                    lower: constant(client, &query.lower, 0.0)?,
                    upper: constant(client, &query.upper, 1.0)?,
                })
            },
            &point,
            &point,
            &derivative,
            &CertifiedBisectionOptions {
                atol: if relative { 0.0 } else { tolerance },
                rtol: if relative { tolerance } else { 0.0 },
                max_iterations: 4,
                max_points: 8,
                probe: Default::default(),
            },
        )?;
        assert_eq!(progress.result().status().try_to_vec::<i64>()?, [3]);
    }
    Ok(())
}

fn identity_matches_legacy<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R> + CertifiedBisectionAlgorithms<R> + CertifiedMappedBisectionAlgorithms<R>,
{
    let lower = tensor::<R>(&[1.0, 1.0], &[2], dtype, client.device())?;
    let upper = tensor::<R>(&[2.0, 2.0], &[2], dtype, client.device())?;
    let derivative = tensor::<R>(&[2.0, 2.0], &[2], dtype, client.device())?;
    let options = CertifiedBisectionOptions {
        atol: 1e-5,
        rtol: 0.0,
        max_iterations: 64,
        max_points: 256,
        probe: Default::default(),
    };
    let legacy_queries = RefCell::new(Vec::new());
    let legacy = client.bisect_certified(
        |query| {
            legacy_queries
                .borrow_mut()
                .push((query.owner.clone(), query.point.clone()));
            fixture::quadratic(client, query)
        },
        &lower,
        &upper,
        &derivative,
        &options,
    )?;
    let mapped_queries = RefCell::new(Vec::new());
    let mapped = client.bisect_certified_mapped(
        |query| {
            mapped_queries
                .borrow_mut()
                .push((query.owner.clone(), query.point.clone()));
            fixture::quadratic(client, query)
        },
        fixture::identity,
        &lower,
        &upper,
        &derivative,
        &options,
    )?;
    assert_eq!(
        fixture::snapshot(&legacy)?,
        fixture::snapshot(mapped.result())?
    );
    assert!(mapped.mapped_points() >= mapped.result().root_owner().numel());
    assert_eq!(legacy_queries.borrow().len(), mapped_queries.borrow().len());
    for (legacy, mapped) in legacy_queries
        .borrow()
        .iter()
        .zip(mapped_queries.borrow().iter())
    {
        assert_eq!(legacy.0.try_to_vec::<i64>()?, mapped.0.try_to_vec::<i64>()?);
        assert_eq!(fixture::bits(&legacy.1)?, fixture::bits(&mapped.1)?);
    }
    Ok(())
}

fn early_stopping_resumes<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R> + CertifiedMappedBisectionAlgorithms<R>,
{
    let lower = tensor::<R>(&[1.0], &[1], dtype, client.device())?;
    let upper = tensor::<R>(&[2.0], &[1], dtype, client.device())?;
    let derivative = tensor::<R>(&[2.0], &[1], dtype, client.device())?;
    let calls = Cell::new(0usize);
    let progress = client.bisect_certified_mapped(
        |query| {
            calls.set(calls.get() + query.point.numel());
            fixture::quadratic(client, query)
        },
        |query| {
            Ok(CertifiedRootMappedValues {
                lower: client.mul_scalar(&query.lower, 1.0 / 1024.0)?,
                upper: client.mul_scalar(&query.upper, 1.0 / 1024.0)?,
            })
        },
        &lower,
        &upper,
        &derivative,
        &CertifiedBisectionOptions {
            atol: 1.0 / 512.0,
            rtol: 0.0,
            max_iterations: 64,
            max_points: 128,
            probe: Default::default(),
        },
    )?;
    assert_eq!(progress.result().status().try_to_vec::<i64>()?, [0]);
    assert_eq!(progress.result().iterations(), 0);
    assert_eq!(progress.result().evaluated_points(), 2);
    assert_eq!(calls.get(), 2);
    let mapped_before = progress.mapped_points();
    let (snapshot, continuation) = progress.into_parts();
    let before = fixture::snapshot(&snapshot)?;
    let queries = RefCell::new(Vec::new());
    let resumed = client.resume_certified_mapped(
        |query| {
            calls.set(calls.get() + query.point.numel());
            queries.borrow_mut().push(query.point.clone());
            fixture::quadratic(client, query)
        },
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
    fixture::contains(resumed.result(), &[2.0_f64.sqrt()])?;
    assert!(resumed.result().iterations() > 0);
    assert_eq!(resumed.result().evaluated_points(), calls.get());
    assert!(resumed.mapped_points() > mapped_before);
    for query in queries.borrow().iter() {
        assert!(fixture::read(query)?.iter().all(|&x| x > 1.0 && x < 2.0));
    }
    assert_eq!(fixture::snapshot(&snapshot)?, before);
    Ok(())
}

fn mapped_width_controls_convergence<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R> + CertifiedMappedBisectionAlgorithms<R>,
{
    let lower = tensor::<R>(&[0.0], &[1], dtype, client.device())?;
    let upper = tensor::<R>(&[1.0], &[1], dtype, client.device())?;
    let derivative = tensor::<R>(&[3.0], &[1], dtype, client.device())?;
    let scale = if dtype == DType::F32 {
        2.0_f64.powi(30)
    } else {
        2.0_f64.powi(60)
    };
    let progress = client.bisect_certified_mapped(
        |query| {
            let value = client.fused_mul_add_scalar(&query.point, 3.0, -1.0)?;
            Ok(solvr::optimize::roots::CertifiedRootValues {
                lower: client.next_down(&value)?,
                upper: client.next_up(&value)?,
            })
        },
        |query| {
            Ok(CertifiedRootMappedValues {
                lower: client.mul_scalar(&query.lower, scale)?,
                upper: client.mul_scalar(&query.upper, scale)?,
            })
        },
        &lower,
        &upper,
        &derivative,
        &CertifiedBisectionOptions {
            atol: 0.5,
            rtol: 0.0,
            max_iterations: 64,
            max_points: 128,
            probe: Default::default(),
        },
    )?;
    assert_eq!(progress.result().status().try_to_vec::<i64>()?, [3]);
    fixture::contains(progress.result(), &[1.0 / 3.0])?;
    Ok(())
}

fn exact_parameter_keeps_mapping_uncertainty<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R> + CertifiedMappedBisectionAlgorithms<R>,
{
    let point = tensor::<R>(&[0.5], &[1], dtype, client.device())?;
    let derivative = tensor::<R>(&[1.0], &[1], dtype, client.device())?;
    let calls = Cell::new(0usize);
    let mapping = |query: &solvr::optimize::roots::CertifiedRootDomainQuery<R>| {
        Ok(CertifiedRootMappedValues {
            lower: constant(client, &query.lower, 0.0)?,
            upper: constant(client, &query.upper, 1.0)?,
        })
    };
    let progress = client.bisect_certified_mapped(
        |query| {
            calls.set(calls.get() + query.point.numel());
            fixture::affine(client, &query.point, 0.5)
        },
        mapping,
        &point,
        &point,
        &derivative,
        &CertifiedBisectionOptions {
            atol: 0.0,
            rtol: 0.0,
            max_iterations: 8,
            max_points: 16,
            probe: Default::default(),
        },
    )?;
    assert_eq!(progress.result().status().try_to_vec::<i64>()?, [3]);
    fixture::contains(progress.result(), &[0.5])?;
    let initial = calls.get();
    let (_, continuation) = progress.into_parts();
    let loose = client.resume_certified_mapped(
        |_| {
            calls.set(calls.get() + 1);
            Err(fixture::sentinel())
        },
        mapping,
        continuation,
        &CertifiedRootTolerance {
            atol: 1.0,
            rtol: 0.0,
        },
        &CertifiedRootAdvanceAllowance {
            max_iterations: 8,
            max_points: 16,
        },
    )?;
    assert_eq!(loose.result().status().try_to_vec::<i64>()?, [0]);
    let (_, continuation) = loose.into_parts();
    let strict = client.resume_certified_mapped(
        |_| {
            calls.set(calls.get() + 1);
            Err(fixture::sentinel())
        },
        mapping,
        continuation,
        &CertifiedRootTolerance {
            atol: 0.0,
            rtol: 0.0,
        },
        &CertifiedRootAdvanceAllowance {
            max_iterations: 8,
            max_points: 16,
        },
    )?;
    assert_eq!(strict.result().status().try_to_vec::<i64>()?, [3]);
    assert_eq!(calls.get(), initial);
    assert_eq!(strict.result().evaluated_points(), initial);
    let lower = tensor::<R>(&[0.0], &[1], dtype, client.device())?;
    let upper = tensor::<R>(&[1.0], &[1], dtype, client.device())?;
    let last_round = client.bisect_certified_mapped(
        |query| fixture::affine(client, &query.point, 0.5),
        mapping,
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
    assert_eq!(last_round.result().status().try_to_vec::<i64>()?, [3]);
    assert_eq!(last_round.result().iterations(), 1);
    fixture::contains(last_round.result(), &[0.5])?;
    Ok(())
}
