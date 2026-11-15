use super::fixture::{self, Client, Result, affine, constant, read, tensor};
use numr::prelude::*;
use solvr::optimize::roots::{CertifiedUpperBracketAlgorithms, CertifiedUpperBracketOptions};
use std::cell::Cell;

pub fn run<R, C>(client: &C) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R> + CertifiedUpperBracketAlgorithms<R>,
{
    for dtype in [DType::F32, DType::F64] {
        let lower = tensor::<R>(&[0.0, 0.0], dtype, client.device())?;
        let seed = tensor::<R>(&[1.0, 1.0], dtype, client.device())?;
        let cap = tensor::<R>(&[8.0, 8.0], dtype, client.device())?;
        for max_points in [0, 1, 2, 3] {
            let called = Cell::new(false);
            let result = client.bracket_upper_certified(
                |_| {
                    called.set(true);
                    Err(fixture::sentinel())
                },
                &lower,
                &seed,
                &cap,
                &CertifiedUpperBracketOptions {
                    max_points,
                    ..Default::default()
                },
            );
            assert!(result.is_err(), "initial reservation accepted {max_points}");
            assert!(!called.get());
        }
        for (max_expansions, max_points, root, expected_upper, status, rounds, points) in [
            (0, 4, 0.5, 1.0, 0, 0, 4),
            (0, 4, 2.0, 1.0, 3, 0, 4),
            (8, 4, 2.0, 1.0, 3, 0, 4),
            (8, 5, 2.0, 1.0, 3, 0, 4),
            (1, 6, 2.0, 2.0, 0, 1, 6),
            (1, 6, 4.0, 2.0, 3, 1, 6),
            (8, 6, 4.0, 2.0, 3, 1, 6),
            (2, 8, 4.0, 4.0, 0, 2, 8),
        ] {
            let count = Cell::new(0usize);
            let result = client.bracket_upper_certified(
                |query| {
                    count.set(count.get() + query.point.numel());
                    assert!(read(&query.point)?.iter().all(|&x| x <= expected_upper));
                    affine(client, &query.point, &constant(client, &query.point, root)?)
                },
                &lower,
                &seed,
                &cap,
                &CertifiedUpperBracketOptions {
                    max_expansions,
                    max_points,
                },
            )?;
            assert_eq!(result.status().try_to_vec::<i64>()?, [status; 2]);
            assert_eq!(read(result.lower())?, [0.0; 2]);
            assert_eq!(read(result.upper())?, [expected_upper; 2]);
            assert_eq!(result.expansion_rounds(), rounds);
            assert_eq!(result.expansions().try_to_vec::<i64>()?, [rounds as i64; 2]);
            assert_eq!(result.callback_points(), points);
            assert_eq!(count.get(), points);
            let lo = read(&result.upper_residual().lower)?;
            let hi = read(&result.upper_residual().upper)?;
            for i in 0..2 {
                let physical = expected_upper - root;
                assert!(lo[i] <= physical && physical <= hi[i]);
            }
        }
        // Completed rows consume no later callback points.
        let roots = tensor::<R>(&[0.5, 2.0], dtype, client.device())?;
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
            &CertifiedUpperBracketOptions {
                max_expansions: 1,
                max_points: 5,
            },
        )?;
        assert_eq!(result.status().try_to_vec::<i64>()?, [0, 0]);
        assert_eq!(result.expansions().try_to_vec::<i64>()?, [0, 1]);
        assert_eq!(result.callback_points(), 5);
        for count in [
            usize::MAX / 2 + 1,
            usize::MAX / dtype.size_in_bytes() + 1,
            (u32::MAX as usize) / 2 + 1,
        ] {
            let lo = lower.narrow(0, 0, 1)?.broadcast_to(&[count])?;
            let hi = seed.narrow(0, 0, 1)?.broadcast_to(&[count])?;
            let called = Cell::new(false);
            let result = client.bracket_upper_certified(
                |_| {
                    called.set(true);
                    Err(fixture::sentinel())
                },
                &lo,
                &hi,
                &hi,
                &CertifiedUpperBracketOptions {
                    max_points: usize::MAX,
                    ..Default::default()
                },
            );
            assert!(
                result.is_err(),
                "virtual count {count} reached callback work"
            );
            assert!(!called.get());
        }
    }
    Ok(())
}
