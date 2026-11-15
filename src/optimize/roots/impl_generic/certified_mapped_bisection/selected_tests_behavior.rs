use super::support::*;
use std::cell::{Cell, RefCell};

pub(super) fn mixed<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    let (_, mut continuation) = begin(client, dtype, &options(256, 2048))?.into_parts();
    let before = snapshot(&continuation)?;
    assert_eq!(before.owners, [0, 2, 3]);
    assert_eq!(before.status[2], 0);
    let demand = owners(client, &[3, 0])?;
    let mut residual_count = 0;
    let mut mapping_count = 0;
    for original in [&[3_i64, 0][..], &[0_i64][..], &[3_i64][..]] {
        let demand = if original.len() == 2 {
            demand.clone()
        } else {
            owners(client, original)?
        };
        let old = snapshot(&continuation)?;
        let residual = Cell::new(0);
        let mapped = Cell::new(0);
        let progress = client.resume_certified_mapped_selected(
            |query| {
                guard(&query.owner, original)?;
                residual.set(residual.get() + query.owner.numel());
                roots(client, query)
            },
            |query| {
                guard(&query.owner, original)?;
                mapped.set(mapped.get() + query.owner.numel());
                mapping(query)
            },
            continuation,
            &demand,
            &tolerance(),
            &allowance(1, 8),
        )?;
        let (_, next) = progress.into_parts();
        let after = snapshot(&next)?;
        let undemanded: Vec<_> = (0..4)
            .filter(|owner| !original.contains(&(*owner as i64)))
            .collect();
        untouched(&old, &after, &undemanded);
        assert_eq!(after.counters[0], old.counters[0] + 1);
        assert_eq!(after.counters[1], old.counters[1] + residual.get());
        assert_eq!(after.counters[2], old.counters[2] + mapped.get());
        assert!(residual.get() > 0 && mapped.get() > 0);
        residual_count += residual.get();
        mapping_count += mapped.get();
        contains(&next)?;
        continuation = next;
    }
    let after = snapshot(&continuation)?;
    untouched(&before, &after, &[1, 2]);
    assert_eq!(after.counters[1], before.counters[1] + residual_count);
    assert_eq!(after.counters[2], before.counters[2] + mapping_count);
    Ok(())
}

pub(super) fn empty<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    let (_, continuation) = begin(client, dtype, &options(256, 2048))?.into_parts();
    let progress = client.resume_certified_mapped_selected(
        |query| roots(client, query),
        mapping,
        continuation,
        &owners(client, &[3])?,
        &tolerance(),
        &allowance(1, 1),
    )?;
    let (result, continuation) = progress.into_parts();
    let before = snapshot(&continuation)?;
    for value in [
        result.status(),
        result.root_owner(),
        result.root_lower(),
        result.root_upper(),
    ] {
        let zero = Tensor::zeros(value.shape(), value.dtype(), client.device())?;
        client.copy_into(value, &zero)?;
    }
    let progress = client.resume_certified_mapped_selected(
        |_| Err(sentinel()),
        |_| Err(sentinel()),
        continuation,
        &owners(client, &[])?,
        &CertifiedRootTolerance {
            atol: 100.0,
            rtol: 1.0,
        },
        &allowance(usize::MAX, usize::MAX),
    )?;
    let (_, continuation) = progress.into_parts();
    assert_eq!(snapshot(&continuation)?, before);
    Ok(())
}

pub(super) fn all_owners<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    let (_, global) = begin(client, dtype, &options(256, 2048))?.into_parts();
    let (_, selected) = begin(client, dtype, &options(256, 2048))?.into_parts();
    let residual_global = RefCell::new(Vec::new());
    let mapped_global = RefCell::new(Vec::new());
    let global = client.resume_certified_mapped(
        |query| {
            residual_global
                .borrow_mut()
                .push(query.owner.try_to_vec::<i64>()?);
            roots(client, query)
        },
        |query| {
            mapped_global
                .borrow_mut()
                .push(query.owner.try_to_vec::<i64>()?);
            mapping(query)
        },
        global,
        &tolerance(),
        &allowance(4, 64),
    )?;
    let residual_selected = RefCell::new(Vec::new());
    let mapped_selected = RefCell::new(Vec::new());
    let selected = client.resume_certified_mapped_selected(
        |query| {
            residual_selected
                .borrow_mut()
                .push(query.owner.try_to_vec::<i64>()?);
            roots(client, query)
        },
        |query| {
            mapped_selected
                .borrow_mut()
                .push(query.owner.try_to_vec::<i64>()?);
            mapping(query)
        },
        selected,
        &owners(client, &[3, 1, 2, 0])?,
        &tolerance(),
        &allowance(4, 64),
    )?;
    let (_, global) = global.into_parts();
    let (_, selected) = selected.into_parts();
    assert_eq!(snapshot(&global)?, snapshot(&selected)?);
    assert_eq!(*residual_global.borrow(), *residual_selected.borrow());
    assert_eq!(*mapped_global.borrow(), *mapped_selected.borrow());
    Ok(())
}

pub(super) fn ceilings<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    for (max_iterations, max_points) in [(2, 64), (64, 10)] {
        let (_, continuation) =
            begin(client, dtype, &options(max_iterations, max_points))?.into_parts();
        let initial = snapshot(&continuation)?;
        let demand = owners(client, &[3, 0])?;
        assert_eq!(
            initial.counters[1], 7,
            "dtype={dtype:?}, ceilings=({max_iterations}, {max_points}), admission={:?}",
            initial.counters,
        );
        let residual_points = Cell::new(0usize);
        let progress = client.resume_certified_mapped_selected(
            |query| {
                guard(&query.owner, &[0, 3])?;
                residual_points.set(residual_points.get() + query.owner.numel());
                roots(client, query)
            },
            |query| {
                guard(&query.owner, &[0, 3])?;
                mapping(query)
            },
            continuation,
            &demand,
            &tolerance(),
            &allowance(usize::MAX, usize::MAX),
        )?;
        let (_, mut continuation) = progress.into_parts();
        let exhausted = snapshot(&continuation)?;
        let context = format!(
            "dtype={dtype:?}, ceilings=({max_iterations}, {max_points}), initial={:?}, exhausted={:?}, residual_points={}",
            initial.counters,
            exhausted.counters,
            residual_points.get(),
        );
        assert!(
            exhausted.counters[0] <= max_iterations && exhausted.counters[1] <= max_points,
            "{context}",
        );
        assert_eq!(
            exhausted.counters[1],
            initial.counters[1] + residual_points.get(),
            "{context}",
        );
        if max_iterations == 2 {
            assert_eq!(exhausted.counters[0], max_iterations, "{context}");
        } else {
            let one = if dtype == DType::F32 {
                u64::from(1.0_f32.to_bits())
            } else {
                1.0_f64.to_bits()
            };
            let pending = exhausted
                .owners
                .iter()
                .zip(&exhausted.fields[5])
                .filter(|(owner, active)| matches!(**owner, 0 | 3) && **active == one)
                .count();
            let remaining = max_points - exhausted.counters[1];
            assert!(
                exhausted.counters[0] > initial.counters[0]
                    && exhausted.counters[1] > initial.counters[1],
                "Point-limited resume makes no progress: {context}",
            );
            assert_eq!(exhausted.counters[0], initial.counters[0] + 1, "{context}");
            assert_eq!(residual_points.get(), 2, "{context}");
            assert_eq!(pending, 2, "{context}");
            assert!(
                remaining < pending,
                "Remaining points={remaining} fit demanded active batch={pending}: {context}",
            );
        }
        untouched(&initial, &exhausted, &[1, 2]);
        for resume in 0..2 {
            let before = snapshot(&continuation)?;
            let progress = client.resume_certified_mapped_selected(
                |_| Err(sentinel()),
                |query| {
                    guard(&query.owner, &[0, 3])?;
                    mapping(query)
                },
                continuation,
                &demand,
                &tolerance(),
                &allowance(usize::MAX, usize::MAX),
            )?;
            let (_, next) = progress.into_parts();
            let after = snapshot(&next)?;
            let context = format!(
                "{context}, repeated_resume={resume}, before={:?}, after={:?}",
                before.counters, after.counters,
            );
            assert_eq!(before.status, after.status, "{context}");
            assert_eq!(before.fields, after.fields, "{context}");
            assert_eq!(before.counters[..2], after.counters[..2], "{context}");
            assert_eq!(after.counters[2], before.counters[2] + 2, "{context}");
            assert_eq!(
                after.counters[3..],
                [max_iterations, max_points],
                "{context}"
            );
            assert!(
                after.counters[0] <= max_iterations && after.counters[1] <= max_points,
                "{context}",
            );
            untouched(&initial, &after, &[1, 2]);
            continuation = next;
        }
    }
    Ok(())
}

pub(super) fn floor_history<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    let (_, continuation) = begin(client, dtype, &options(256, 2048))?.into_parts();
    let progress = client.resume_certified_mapped_selected(
        |query| roots(client, query),
        mapping,
        continuation,
        &owners(client, &[2])?,
        &tolerance(),
        &allowance(128, 256),
    )?;
    let (_, continuation) = progress.into_parts();
    let before = snapshot(&continuation)?;
    assert_eq!(before.status[2], 3);
    assert!(before.fields[6][1] != 0 && before.fields[4][1] != 0);
    let progress = client.resume_certified_mapped_selected(
        |query| {
            guard(&query.owner, &[0])?;
            roots(client, query)
        },
        |query| {
            guard(&query.owner, &[0])?;
            mapping(query)
        },
        continuation,
        &owners(client, &[0])?,
        &tolerance(),
        &allowance(2, 2),
    )?;
    let (_, continuation) = progress.into_parts();
    untouched(&before, &snapshot(&continuation)?, &[1, 2, 3]);
    Ok(())
}
