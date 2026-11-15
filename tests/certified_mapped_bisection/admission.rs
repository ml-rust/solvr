use super::fixture::{self, Client, Result, constant, tensor};
use numr::prelude::*;
use solvr::optimize::roots::{
    CertifiedBisectionOptions, CertifiedMappedBisectionAlgorithms, CertifiedRootAdvanceAllowance,
    CertifiedRootMappedValues, CertifiedRootTolerance,
};
use std::cell::Cell;

pub fn run<R, C>(client: &C) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R> + CertifiedMappedBisectionAlgorithms<R>,
{
    for dtype in [DType::F32, DType::F64] {
        malformed_mapping_rejects(client, dtype)?;
        invalid_tolerances_precede_callbacks(client, dtype)?;
        empty_and_absent_roots_do_not_map(client, dtype)?;
        residual_errors_preserve_the_cause(client, dtype)?;
    }
    Ok(())
}

fn malformed_mapping_rejects<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R> + CertifiedMappedBisectionAlgorithms<R>,
{
    let lower = tensor::<R>(&[1.0], &[1], dtype, client.device())?;
    let upper = tensor::<R>(&[2.0], &[1], dtype, client.device())?;
    let derivative = tensor::<R>(&[2.0], &[1], dtype, client.device())?;
    let other = if dtype == DType::F32 {
        DType::F64
    } else {
        DType::F32
    };
    for malformed in 0..8 {
        let called = Cell::new(false);
        let result = client.bisect_certified_mapped(
            |query| fixture::quadratic(client, query),
            |query| {
                called.set(true);
                let lower = constant(client, &query.lower, 0.0)?;
                let upper = match malformed {
                    0 => Tensor::empty(&[query.lower.numel(), 1], dtype, client.device())?,
                    1 => Tensor::empty(&[query.lower.numel() + 1], dtype, client.device())?,
                    2 => Tensor::empty(query.lower.shape(), other, client.device())?,
                    3 => constant(client, &query.lower, f64::NAN)?,
                    4 => constant(client, &query.lower, f64::INFINITY)?,
                    5 => constant(client, &query.lower, -1.0)?,
                    6 => return Err(fixture::sentinel()),
                    _ => {
                        return Ok(CertifiedRootMappedValues {
                            lower: constant(client, &query.lower, f64::NEG_INFINITY)?,
                            upper: lower,
                        });
                    }
                };
                Ok(CertifiedRootMappedValues { lower, upper })
            },
            &lower,
            &upper,
            &derivative,
            &CertifiedBisectionOptions::default(),
        );
        assert!(
            result.is_err(),
            "malformed mapped bounds {malformed} accepted"
        );
        assert!(called.get());
        if malformed == 6 {
            match result {
                Err(error) => assert!(error.to_string().contains("mapped callback sentinel")),
                Ok(_) => return Err("Mapped callback error accepted".into()),
            }
        }
    }
    let progress = client.bisect_certified_mapped(
        |query| fixture::quadratic(client, query),
        fixture::identity,
        &lower,
        &upper,
        &derivative,
        &CertifiedBisectionOptions {
            atol: 2.0,
            ..Default::default()
        },
    )?;
    let (_, continuation) = progress.into_parts();
    let result = client.resume_certified_mapped(
        |_| Err(fixture::sentinel()),
        |_| Err(fixture::sentinel()),
        continuation,
        &CertifiedRootTolerance {
            atol: 0.0,
            rtol: 0.0,
        },
        &CertifiedRootAdvanceAllowance {
            max_iterations: 1,
            max_points: 1,
        },
    );
    match result {
        Err(error) => assert!(error.to_string().contains("mapped callback sentinel")),
        Ok(_) => return Err("Resumed mapped callback error accepted".into()),
    }
    Ok(())
}

fn invalid_tolerances_precede_callbacks<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R> + CertifiedMappedBisectionAlgorithms<R>,
{
    let lower = tensor::<R>(&[1.0], &[1], dtype, client.device())?;
    let upper = tensor::<R>(&[2.0], &[1], dtype, client.device())?;
    let derivative = tensor::<R>(&[2.0], &[1], dtype, client.device())?;
    for invalid in [-1.0, f64::NAN, f64::INFINITY] {
        for relative in [false, true] {
            let tolerance = CertifiedRootTolerance {
                atol: if relative { 0.0 } else { invalid },
                rtol: if relative { invalid } else { 0.0 },
            };
            let called = Cell::new(false);
            assert!(
                client
                    .bisect_certified_mapped(
                        |_| {
                            called.set(true);
                            Err(fixture::sentinel())
                        },
                        |_| {
                            called.set(true);
                            Err(fixture::sentinel())
                        },
                        &lower,
                        &upper,
                        &derivative,
                        &CertifiedBisectionOptions {
                            atol: tolerance.atol,
                            rtol: tolerance.rtol,
                            ..Default::default()
                        }
                    )
                    .is_err()
            );
            assert!(!called.get());
            let progress = client.bisect_certified_mapped(
                |query| fixture::quadratic(client, query),
                fixture::identity,
                &lower,
                &upper,
                &derivative,
                &CertifiedBisectionOptions {
                    atol: 2.0,
                    ..Default::default()
                },
            )?;
            let (_, continuation) = progress.into_parts();
            assert!(
                client
                    .resume_certified_mapped(
                        |_| {
                            called.set(true);
                            Err(fixture::sentinel())
                        },
                        |_| {
                            called.set(true);
                            Err(fixture::sentinel())
                        },
                        continuation,
                        &tolerance,
                        &CertifiedRootAdvanceAllowance {
                            max_iterations: 1,
                            max_points: 1
                        }
                    )
                    .is_err()
            );
            assert!(!called.get());
        }
    }
    Ok(())
}

fn empty_and_absent_roots_do_not_map<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R> + CertifiedMappedBisectionAlgorithms<R>,
{
    let empty = Tensor::<R>::empty(&[0], dtype, client.device())?;
    let called = Cell::new(false);
    let progress = client.bisect_certified_mapped(
        |_| {
            called.set(true);
            Err(fixture::sentinel())
        },
        |_| {
            called.set(true);
            Err(fixture::sentinel())
        },
        &empty,
        &empty,
        &empty,
        &CertifiedBisectionOptions {
            max_points: 0,
            ..Default::default()
        },
    )?;
    assert!(!called.get());
    assert_eq!(progress.result().status().shape(), [0]);
    assert_eq!(progress.result().root_owner().shape(), [0]);
    assert_eq!(progress.mapped_points(), 0);
    let (_, continuation) = progress.into_parts();
    let resumed = client.resume_certified_mapped(
        |_| {
            called.set(true);
            Err(fixture::sentinel())
        },
        |_| {
            called.set(true);
            Err(fixture::sentinel())
        },
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
    assert!(!called.get());
    assert_eq!(resumed.result().status().shape(), [0]);
    assert_eq!(
        (
            resumed.result().iterations(),
            resumed.result().evaluated_points()
        ),
        (0, 0)
    );
    assert_eq!(resumed.mapped_points(), 0);

    let lower = tensor::<R>(&[0.0], &[1], dtype, client.device())?;
    let upper = tensor::<R>(&[1.0], &[1], dtype, client.device())?;
    let absent = client.bisect_certified_mapped(
        |query| fixture::affine(client, &query.point, -1.0),
        |_| {
            called.set(true);
            Err(fixture::sentinel())
        },
        &lower,
        &upper,
        &upper,
        &CertifiedBisectionOptions::default(),
    )?;
    assert!(!called.get());
    assert_eq!(absent.result().status().try_to_vec::<i64>()?, [1]);
    assert_eq!(absent.result().root_owner().numel(), 0);
    Ok(())
}

fn residual_errors_preserve_the_cause<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R> + CertifiedMappedBisectionAlgorithms<R>,
{
    let lower = tensor::<R>(&[0.0], &[1], dtype, client.device())?;
    let upper = tensor::<R>(&[1.0], &[1], dtype, client.device())?;
    let called = Cell::new(false);
    let result = client.bisect_certified_mapped(
        |_| Err(fixture::sentinel()),
        |_| {
            called.set(true);
            Err(fixture::sentinel())
        },
        &lower,
        &upper,
        &upper,
        &CertifiedBisectionOptions::default(),
    );
    assert!(!called.get());
    match result {
        Err(error) => assert!(error.to_string().contains("mapped callback sentinel")),
        Ok(_) => return Err("Residual callback error accepted".into()),
    }
    Ok(())
}
