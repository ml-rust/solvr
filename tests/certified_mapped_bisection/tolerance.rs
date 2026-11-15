use super::fixture::{self, Client, Result, tensor};
use numr::ops::{MidpointOps, TwoSumOps};
use numr::prelude::*;
use solvr::common::arithmetic::with_interval_arithmetic;
use solvr::optimize::roots::{
    CertifiedBisectionOptions, CertifiedMappedBisectionAlgorithms, CertifiedRootMappedValues,
    CertifiedRootTolerance, CertifiedRootValues,
};

pub fn run<R, C>(client: &C) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R> + MidpointOps<R> + TwoSumOps<R> + CertifiedMappedBisectionAlgorithms<R>,
{
    for dtype in [DType::F32, DType::F64] {
        for strided in [false, true] {
            predicate_matches_solver(client, dtype, strided)?;
        }
        inward_tolerances(client, dtype)?;
        admission(client, dtype)?;
    }
    Ok(())
}

fn input<R, C>(
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

fn predicate_matches_solver<R, C>(client: &C, dtype: DType, strided: bool) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R> + MidpointOps<R> + TwoSumOps<R> + CertifiedMappedBisectionAlgorithms<R>,
{
    let lower = input(client, &[1.0, 0.0, 2.0, -4.0, -0.0], dtype, strided)?;
    let upper = input(client, &[1.0, 0.25, 2.5, -3.5, 0.0], dtype, strided)?;
    let before = [
        fixture::bits(&lower.contiguous()?)?,
        fixture::bits(&upper.contiguous()?)?,
    ];
    for (tolerance, expected) in [
        (
            CertifiedRootTolerance {
                atol: 0.0,
                rtol: 0.0,
            },
            [1.0, 0.0, 0.0, 0.0, 1.0],
        ),
        (
            CertifiedRootTolerance {
                atol: 0.25,
                rtol: 0.0,
            },
            [1.0, 1.0, 0.0, 0.0, 1.0],
        ),
        (
            CertifiedRootTolerance {
                atol: 0.0,
                rtol: 0.125,
            },
            [1.0, 0.0, 0.0, 1.0, 1.0],
        ),
        (
            CertifiedRootTolerance {
                atol: 1.0,
                rtol: 0.0,
            },
            [1.0; 5],
        ),
    ] {
        let mask = with_interval_arithmetic::<R, _, _, _>(
            client,
            dtype,
            "public root tolerance",
            |ctx| Ok(tolerance.satisfied(ctx, &lower, &upper)),
        )??;
        assert_eq!(fixture::read(&mask)?, expected);
        let point = Tensor::<R>::zeros(&[5], dtype, client.device())?;
        let derivative = Tensor::<R>::ones(&[5], dtype, client.device())?;
        let mapped = client.bisect_certified_mapped(
            |query| {
                Ok(CertifiedRootValues {
                    lower: query.point.clone(),
                    upper: query.point.clone(),
                })
            },
            |query| {
                Ok(CertifiedRootMappedValues {
                    lower: client.index_select(&lower, 0, &query.owner)?,
                    upper: client.index_select(&upper, 0, &query.owner)?,
                })
            },
            &point,
            &point,
            &derivative,
            &CertifiedBisectionOptions {
                atol: tolerance.atol,
                rtol: tolerance.rtol,
                max_iterations: 0,
                max_points: 10,
            },
        )?;
        assert_eq!(
            mapped.result().root_owner().try_to_vec::<i64>()?,
            [0, 1, 2, 3, 4]
        );
        let statuses = mapped.result().status().try_to_vec::<i64>()?;
        for (&accepted, status) in expected.iter().zip(statuses) {
            assert_eq!(status, if accepted == 1.0 { 0 } else { 3 });
        }
    }
    assert_eq!(fixture::bits(&lower.contiguous()?)?, before[0]);
    assert_eq!(fixture::bits(&upper.contiguous()?)?, before[1]);
    let empty = Tensor::<R>::empty(&[0], dtype, client.device())?;
    let empty_mask =
        with_interval_arithmetic::<R, _, _, _>(client, dtype, "empty root tolerance", |ctx| {
            Ok(CertifiedRootTolerance {
                atol: 0.0,
                rtol: 0.0,
            }
            .satisfied(ctx, &empty, &empty))
        })??;
    assert_eq!(empty_mask.shape(), [0]);
    assert_eq!(empty_mask.dtype(), dtype);
    Ok(())
}

fn inward_tolerances<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R> + MidpointOps<R> + TwoSumOps<R>,
{
    let previous = f64::from(f32::from_bits(0x3f7fffff));
    let tolerance = previous + (1.0 - previous) * 0.75;
    assert_eq!(tolerance as f32, 1.0);
    let lower = tensor::<R>(&[0.0], &[1], dtype, client.device())?;
    let upper = tensor::<R>(&[1.0], &[1], dtype, client.device())?;
    for relative in [false, true] {
        let tolerance = CertifiedRootTolerance {
            atol: if relative { 0.0 } else { tolerance },
            rtol: if relative { tolerance } else { 0.0 },
        };
        let mask = with_interval_arithmetic::<R, _, _, _>(
            client,
            dtype,
            "inward public tolerance",
            |ctx| Ok(tolerance.satisfied(ctx, &lower, &upper)),
        )??;
        assert_eq!(fixture::read(&mask)?, [0.0]);
    }
    Ok(())
}

fn admission<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R> + MidpointOps<R> + TwoSumOps<R>,
{
    let zero = tensor::<R>(&[0.0], &[1], dtype, client.device())?;
    let one = tensor::<R>(&[1.0], &[1], dtype, client.device())?;
    let other = if dtype == DType::F32 {
        DType::F64
    } else {
        DType::F32
    };
    let tolerance = CertifiedRootTolerance {
        atol: 0.0,
        rtol: 0.0,
    };
    for (lower, upper) in [
        (
            tensor::<R>(&[0.0], &[1, 1], dtype, client.device())?,
            tensor::<R>(&[1.0], &[1, 1], dtype, client.device())?,
        ),
        (
            zero.clone(),
            tensor::<R>(&[1.0, 1.0], &[2], dtype, client.device())?,
        ),
        (
            zero.clone(),
            tensor::<R>(&[1.0], &[1], other, client.device())?,
        ),
        (
            tensor::<R>(&[0.0], &[1], other, client.device())?,
            tensor::<R>(&[1.0], &[1], other, client.device())?,
        ),
        (one.clone(), zero.clone()),
        (
            tensor::<R>(&[f64::NAN], &[1], dtype, client.device())?,
            one.clone(),
        ),
        (
            zero.clone(),
            tensor::<R>(&[f64::INFINITY], &[1], dtype, client.device())?,
        ),
        (
            tensor::<R>(&[f64::NEG_INFINITY], &[1], dtype, client.device())?,
            one.clone(),
        ),
    ] {
        with_interval_arithmetic::<R, _, _, _>(
            client,
            dtype,
            "public tolerance admission",
            |ctx| {
                assert!(tolerance.satisfied(ctx, &lower, &upper).is_err());
                let _retained = ctx.point(&zero)?;
                Ok(())
            },
        )?;
    }
    for invalid in [-1.0, f64::NAN, f64::INFINITY] {
        for relative in [false, true] {
            let tolerance = CertifiedRootTolerance {
                atol: if relative { 0.0 } else { invalid },
                rtol: if relative { invalid } else { 0.0 },
            };
            with_interval_arithmetic::<R, _, _, _>(
                client,
                dtype,
                "invalid public tolerance",
                |ctx| {
                    assert!(tolerance.satisfied(ctx, &zero, &one).is_err());
                    let _retained = ctx.point(&zero)?;
                    Ok(())
                },
            )?;
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
pub fn foreign_device<R, C>(client: &C, foreign: &Tensor<R>) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R> + MidpointOps<R> + TwoSumOps<R>,
{
    with_interval_arithmetic::<R, _, _, _>(
        client,
        foreign.dtype(),
        "foreign public tolerance",
        |ctx| {
            assert!(
                CertifiedRootTolerance {
                    atol: 0.0,
                    rtol: 0.0
                }
                .satisfied(ctx, foreign, foreign)
                .is_err()
            );
            Ok(())
        },
    )?;
    Ok(())
}
