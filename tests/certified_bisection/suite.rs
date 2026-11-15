use super::callback;
use super::fixture::{self, Case, Client, Result, decode};
use numr::prelude::*;
use solvr::optimize::roots::{CertifiedBisectionAlgorithms, CertifiedRootStatus};
use std::cell::{Cell, RefCell};
use std::collections::HashSet;

fn endpoint<R, C>(
    client: &C,
    case: &Case,
    raw: u64,
    strided: bool,
) -> numr::error::Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    let value = decode(raw, case.dtype);
    if strided {
        return Ok(fixture::tensor::<R>(
            &[17.0, value, 23.0, 29.0],
            &[2, 2],
            case.dtype,
            client.device(),
        )?
        .narrow(0, 0, 1)?
        .narrow(1, 1, 1)?
        .squeeze(Some(1)));
    }
    fixture::tensor(&[value], &[1], case.dtype, client.device())
}

fn exact_difference(a: f64, b: f64) -> (f64, f64) {
    let sum = a - b;
    let recovered_b = sum - a;
    let error = (a - (sum - recovered_b)) + (-b - recovered_b);
    (sum, error)
}

fn check_width(case: &Case, lower: f64, upper: f64) {
    let atol = f64::from_bits(case.atol);
    let rtol = f64::from_bits(case.rtol);
    if atol == 0.0 && rtol == 0.0 {
        assert_eq!(lower, upper, "{} requires exact width", case.id);
        return;
    }
    let scale = lower.abs().max(upper.abs());
    let relative = rtol * scale;
    // Frozen relative-only acceptance rows use an exactly represented product.
    assert_eq!(
        rtol.mul_add(scale, -relative),
        0.0,
        "{} relative reference",
        case.id
    );
    let tolerance = atol + relative;
    let (_, tolerance_error) = exact_difference(tolerance, atol);
    assert_eq!(tolerance_error, 0.0);
    assert_eq!(tolerance - atol, relative);
    let (width, error) = exact_difference(upper, lower);
    assert!(
        width < tolerance || (width == tolerance && error <= 0.0),
        "{} width exceeds original f64 tolerance",
        case.id
    );
}

pub fn run<R, C>(client: &C, strided: bool) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R> + CertifiedBisectionAlgorithms<R>,
{
    assert_eq!(fixture::cases().count(), 40);
    for case in fixture::cases() {
        let lower = endpoint(client, case, case.lower, strided)?;
        let upper = endpoint(client, case, case.upper, strided)?;
        let derivative = endpoint(client, case, case.kappa, strided)?;
        let before = [
            fixture::bits(client, &lower)?,
            fixture::bits(client, &upper)?,
            fixture::bits(client, &derivative)?,
        ];
        let points = Cell::new(0usize);
        let queries = RefCell::new(Vec::new());
        let options = case.options();
        let result = client.bisect_certified(
            |query| {
                assert_eq!(query.owner.dtype(), DType::I64);
                assert_eq!(query.owner.shape(), query.point.shape());
                assert_eq!(query.point.dtype(), case.dtype);
                assert!(query.point.device().is_same(client.device()));
                assert!(query.owner.device().is_same(client.device()));
                points.set(points.get() + query.point.numel());
                queries
                    .borrow_mut()
                    .push((query.owner.clone(), query.point.clone()));
                callback::evaluate(client, case, query)
            },
            &lower,
            &upper,
            &derivative,
            &options,
        )?;
        assert_eq!(options.atol.to_bits(), case.atol);
        assert_eq!(options.rtol.to_bits(), case.rtol);
        assert_eq!(result.status().shape(), &[1]);
        assert_eq!(result.status().dtype(), DType::I64);
        assert_eq!(
            result.status().try_to_vec::<i64>()?,
            [case.status],
            "{} status",
            case.id
        );
        assert!(CertifiedRootStatus::from_code(case.status).is_some());
        assert_eq!(
            result.evaluated_points(),
            points.get(),
            "{} point accounting",
            case.id
        );
        assert!(result.evaluated_points() <= options.max_points);
        assert!(result.iterations() <= options.max_iterations);
        let owners = result.root_owner().try_to_vec::<i64>()?;
        let count = usize::from(case.certified);
        assert_eq!(result.root_owner().dtype(), DType::I64);
        assert_eq!(result.root_owner().shape(), &[count]);
        for value in [result.root_lower(), result.root_upper()] {
            assert_eq!(value.shape(), &[count]);
            assert_eq!(value.dtype(), case.dtype);
            assert!(value.device().is_same(client.device()));
        }
        if case.certified {
            assert_eq!(owners, [0], "{} original owner", case.id);
            let lo = fixture::read(result.root_lower())?[0];
            let hi = fixture::read(result.root_upper())?[0];
            assert!(lo.is_finite() && hi.is_finite() && lo <= hi);
            assert!(decode(case.lower, case.dtype) <= lo && hi <= decode(case.upper, case.dtype));
            // Exact rational or squared references determine neighboring representable endpoints independently.
            assert!(
                lo <= decode(case.root_lower, case.dtype),
                "{} excludes root from below",
                case.id
            );
            assert!(
                hi >= decode(case.root_upper, case.dtype),
                "{} excludes root from above",
                case.id
            );
            if case.status == 0 {
                check_width(case, lo, hi);
            }
            if case.id == "false_nominal_zero" {
                assert!(lo < 0.75 && hi > 0.75);
            }
        } else {
            assert!(owners.is_empty());
        }
        let mut sampled = HashSet::new();
        for (owner, point) in queries.borrow().iter() {
            assert!(owner.try_to_vec::<i64>()?.iter().all(|&value| value == 0));
            for x in fixture::read(point)? {
                let canonical_bits = if x == 0.0 { 0 } else { x.to_bits() };
                assert!(
                    sampled.insert((0i64, canonical_bits)),
                    "{} repeats an owner/point query",
                    case.id
                );
                assert!(decode(case.lower, case.dtype) <= x && x <= decode(case.upper, case.dtype));
            }
        }
        assert_eq!(fixture::bits(client, &lower)?, before[0]);
        assert_eq!(fixture::bits(client, &upper)?, before[1]);
        assert_eq!(fixture::bits(client, &derivative)?, before[2]);
    }
    Ok(())
}
