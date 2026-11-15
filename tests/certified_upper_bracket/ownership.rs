use super::fixture::{Client, Result, affine, constant, input, read};
use numr::prelude::*;
use solvr::optimize::roots::{
    CertifiedRootValues, CertifiedUpperBracketAlgorithms, CertifiedUpperBracketOptions,
};
use std::cell::{Cell, RefCell};

pub fn run<R, C>(client: &C) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R> + CertifiedUpperBracketAlgorithms<R>,
{
    for dtype in [DType::F32, DType::F64] {
        for strided in [false, true] {
            snapshots(client, dtype, strided)?;
        }
    }
    Ok(())
}

fn snapshots<R, C>(client: &C, dtype: DType, strided: bool) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R> + CertifiedUpperBracketAlgorithms<R>,
{
    let lower = input::<R>(&[0.0, 0.0, 0.0], dtype, client.device(), strided)?;
    let seed = input::<R>(&[1.0, 1.0, 1.0], dtype, client.device(), strided)?;
    let cap = input::<R>(&[8.0, 8.0, 8.0], dtype, client.device(), strided)?;
    let roots = input::<R>(&[4.0, 0.5, 2.0], dtype, client.device(), strided)?;
    let shared_lower = Tensor::<R>::empty(&[6], dtype, client.device())?;
    let shared_upper = Tensor::<R>::empty(&[6], dtype, client.device())?;
    let calls = Cell::new(0usize);
    let traces = RefCell::new(Vec::new());
    let result = client.bracket_upper_certified(
        |query| {
            traces
                .borrow_mut()
                .push((query.owner.try_to_vec::<i64>()?, read(&query.point)?));
            let root = client.index_select(&roots, 0, &query.owner)?;
            let value = affine(client, &query.point, &root)?;
            let returned_lower = shared_lower.narrow(0, 0, query.point.numel())?;
            let returned_upper = shared_upper.narrow(0, 0, query.point.numel())?;
            client.copy_into(&returned_lower, &value.lower)?;
            client.copy_into(&returned_upper, &value.upper)?;
            if calls.get() == 0 {
                // Caller aliases change after entry. The physical functions stay fixed.
                for alias in [&lower, &seed, &cap] {
                    client.copy_into(alias, &constant(client, alias, 99.0)?)?;
                }
            }
            calls.set(calls.get() + 1);
            Ok(CertifiedRootValues {
                lower: returned_lower,
                upper: returned_upper,
            })
        },
        &lower,
        &seed,
        &cap,
        &CertifiedUpperBracketOptions::default(),
    )?;
    assert_eq!(result.status().try_to_vec::<i64>()?, [0, 0, 0]);
    assert_eq!(result.expansions().try_to_vec::<i64>()?, [2, 0, 1]);
    assert_eq!(result.expansion_rounds(), 2);
    assert_eq!(result.callback_points(), 9);
    assert_eq!(read(result.lower())?, [0.0; 3]);
    assert_eq!(read(result.upper())?, [4.0, 1.0, 2.0]);
    for external in [&lower, &seed, &cap] {
        let captured = Tensor::empty(external.shape(), dtype, client.device())?;
        client.copy_into(&captured, external)?;
        assert_eq!(read(&captured)?, [99.0; 3]);
    }
    for (owners, points) in traces.borrow().iter() {
        for (&owner, &point) in owners.iter().zip(points) {
            assert!((0..3).contains(&owner));
            assert!((0.0..=8.0).contains(&point));
            if point > 1.0 {
                assert!(owner == 0 || owner == 2);
            }
        }
    }
    let endpoints = [
        result.lower(),
        result.upper(),
        &result.lower_residual().lower,
        &result.lower_residual().upper,
        &result.upper_residual().lower,
        &result.upper_residual().upper,
    ];
    let saved = endpoints
        .iter()
        .map(|value| read(value))
        .collect::<numr::error::Result<Vec<_>>>()?;
    let roots_host = [4.0, 0.5, 2.0];
    for (i, root) in roots_host.into_iter().enumerate() {
        assert!(saved[2][i] <= -root && -root <= saved[3][i]);
        let physical = saved[1][i] - root;
        assert!(saved[4][i] <= physical && physical <= saved[5][i]);
    }
    for alias in [&lower, &seed, &cap, &roots, &shared_lower, &shared_upper] {
        client.copy_into(
            alias,
            &Tensor::zeros(alias.shape(), dtype, client.device())?,
        )?;
    }
    for (value, expected) in endpoints.iter().zip(&saved) {
        assert_eq!(&read(value)?, expected);
    }
    assert_eq!(result.status().try_to_vec::<i64>()?, [0, 0, 0]);
    assert_eq!(result.expansions().try_to_vec::<i64>()?, [2, 0, 1]);
    Ok(())
}
