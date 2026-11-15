use super::callback::{self, Client};
use super::fixture::{self, Admission, Case, Result, decode};
use numr::prelude::*;
use solvr::integrate::{
    CertifiedSimpsonAlgorithms, CertifiedSimpsonPanel, CertifiedSimpsonQuery, IntegrateError,
    IntegrateResult,
};

fn input_bits<R, C>(client: &C, input: &Tensor<R>) -> numr::error::Result<Vec<u64>>
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    if input.numel() == 0 {
        return fixture::read_bits(input);
    }
    let copied = Tensor::empty(input.shape(), input.dtype(), client.device())?;
    client.copy_into(&copied, input)?;
    fixture::read_bits(&copied)
}

fn endpoints<R: Runtime<DType = DType>>(
    case: &Case,
    device: &R::Device,
    right: bool,
    strided: bool,
) -> numr::error::Result<Tensor<R>> {
    let values: Vec<_> = case
        .lanes
        .iter()
        .map(|lane| decode(if right { lane.b } else { lane.a }, case.dtype))
        .collect();
    if !strided || values.is_empty() {
        return fixture::tensor(&values, &[values.len()], case.dtype, device);
    }
    let paired: Vec<_> = values.iter().flat_map(|&value| [value, 19.0]).collect();
    Ok(
        fixture::tensor::<R>(&paired, &[values.len(), 2], case.dtype, device)?
            .narrow(1, 0, 1)?
            .squeeze(Some(1)),
    )
}
fn adjacent(case: &Case, owner: usize) -> bool {
    let lane = &case.lanes[owner];
    if case.dtype == DType::F32 {
        f32::from_bits(lane.a as u32).next_up().to_bits() == lane.b as u32
    } else {
        f64::from_bits(lane.a).next_up().to_bits() == lane.b
    }
}
fn check_owner_order(case: &Case, owners: &[i64], per_lane: usize) {
    let expected: Vec<_> = case
        .lanes
        .iter()
        .enumerate()
        .filter(|(_, lane)| lane.active)
        .flat_map(|(owner, _)| std::iter::repeat_n(owner as i64, per_lane))
        .collect();
    assert_eq!(owners, expected, "{} owner order", case.id);
}
fn check_queries<R: Runtime<DType = DType>>(
    case: &Case,
    query: &CertifiedSimpsonQuery<R>,
) -> IntegrateResult<()> {
    assert_eq!(query.owner.dtype(), DType::I64);
    let owners = query.owner.try_to_vec::<i64>()?;
    check_owner_order(case, &owners, 2 * case.panels + 1);
    let points = fixture::read(&query.point)?;
    let lower = fixture::read(&query.lower)?;
    let upper = fixture::read(&query.upper)?;
    for (i, &owner) in owners.iter().enumerate() {
        let owner = owner as usize;
        let a = decode(case.lanes[owner].a, case.dtype);
        let b = decode(case.lanes[owner].b, case.dtype);
        assert!(
            a <= lower[i] && lower[i] <= points[i] && points[i] <= upper[i] && upper[i] <= b,
            "{} query {i}",
            case.id
        );
        let node = i % (2 * case.panels + 1);
        if node == 0 {
            assert_eq!((points[i], lower[i], upper[i]), (a, a, a));
        } else if node == 2 * case.panels {
            assert_eq!((points[i], lower[i], upper[i]), (b, b, b));
        } else if adjacent(case, owner) {
            assert_eq!((lower[i], upper[i]), (a, b));
        }
    }
    Ok(())
}
fn check_panels<R: Runtime<DType = DType>>(
    case: &Case,
    query: &CertifiedSimpsonPanel<R>,
) -> IntegrateResult<()> {
    let owners = query.owner.try_to_vec::<i64>()?;
    check_owner_order(case, &owners, case.panels);
    let lower = fixture::read(&query.lower)?;
    let upper = fixture::read(&query.upper)?;
    for (i, &owner) in owners.iter().enumerate() {
        let owner = owner as usize;
        let a = decode(case.lanes[owner].a, case.dtype);
        let b = decode(case.lanes[owner].b, case.dtype);
        assert!(
            a <= lower[i] && lower[i] <= upper[i] && upper[i] <= b,
            "{} panel {i}",
            case.id
        );
        if adjacent(case, owner) {
            assert_eq!((lower[i], upper[i]), (a, b));
        }
    }
    Ok(())
}
pub fn run<R, C>(client: &C, strided: bool) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R> + CertifiedSimpsonAlgorithms<R>,
{
    assert_eq!(fixture::cases().count(), 40);
    for case in fixture::cases() {
        let left = endpoints(case, client.device(), false, strided)?;
        let right = endpoints(case, client.device(), true, strided)?;
        let left_bits = input_bits(client, &left)?;
        let right_bits = input_bits(client, &right)?;
        let mut value_calls = 0;
        let mut derivative_calls = 0;
        let mut value_points = 0;
        let mut derivative_points = 0;
        let mut retained = Vec::new();
        let result = client.certified_simpson(
            &left,
            &right,
            case.options(),
            |query| {
                value_calls += 1;
                value_points += query.point.numel();
                check_queries(case, query)?;
                let output = callback::evaluate(client, case, query)?;
                retained.extend([
                    output.value.clone(),
                    output.lower.clone(),
                    output.upper.clone(),
                ]);
                Ok(output)
            },
            |query| {
                derivative_calls += 1;
                derivative_points += query.lower.numel();
                check_panels(case, query)?;
                Ok(Tensor::full_scalar(
                    query.lower.shape(),
                    case.dtype,
                    decode(case.m4, case.dtype),
                    client.device(),
                )?)
            },
        );
        assert_eq!(
            input_bits(client, &left)?,
            left_bits,
            "{} left mutation",
            case.id
        );
        assert_eq!(
            input_bits(client, &right)?,
            right_bits,
            "{} right mutation",
            case.id
        );
        assert!(
            value_points <= case.function_ceiling,
            "{} value budget",
            case.id
        );
        assert!(
            derivative_points <= case.derivative_ceiling,
            "{} derivative budget",
            case.id
        );
        assert!(value_points + derivative_points <= case.max_points);
        let receipt = match result {
            Ok(receipt) => {
                assert!(
                    case.admission != Admission::Reject,
                    "{} requires a typed range rejection",
                    case.id
                );
                receipt
            }
            Err(error) => {
                assert!(
                    case.admission != Admission::Required,
                    "{} unexpected error: {error}",
                    case.id
                );
                assert!(
                    matches!(error, IntegrateError::ArithmeticRange { .. }),
                    "{} wrong rejection: {error}",
                    case.id
                );
                continue;
            }
        };
        let active = case.lanes.iter().filter(|lane| lane.active).count();
        assert_eq!(
            value_calls,
            usize::from(active != 0),
            "{} value calls",
            case.id
        );
        assert_eq!(
            derivative_calls,
            usize::from(active != 0),
            "{} derivative calls",
            case.id
        );
        assert_eq!(value_points, active * (2 * case.panels + 1));
        assert_eq!(derivative_points, active * case.panels);
        let outputs = [
            receipt.value(),
            receipt.lower(),
            receipt.upper(),
            receipt.abs_error_bound(),
            receipt.quadrature_error_bound(),
        ];
        for output in outputs {
            assert_eq!(output.shape(), &[case.lanes.len()]);
            assert_eq!(output.dtype(), case.dtype);
            assert!(output.device().is_same(client.device()));
            if output.numel() != 0 {
                assert!(!output.shares_storage_with(&left));
                assert!(!output.shares_storage_with(&right));
                for callback_output in &retained {
                    assert!(
                        !output.shares_storage_with(callback_output),
                        "{} callback alias",
                        case.id
                    );
                }
            }
        }
        let value = fixture::read(receipt.value())?;
        let lower = fixture::read(receipt.lower())?;
        let upper = fixture::read(receipt.upper())?;
        let error = fixture::read(receipt.abs_error_bound())?;
        let quadrature = fixture::read(receipt.quadrature_error_bound())?;
        for (i, lane) in case.lanes.iter().enumerate() {
            for field in [&value, &lower, &upper, &error, &quadrature] {
                assert!(field[i].is_finite(), "{} nonfinite receipt", case.id);
            }
            let truth_low = decode(lane.truth.0, case.dtype);
            let truth_high = decode(lane.truth.1, case.dtype);
            assert!(
                lower[i] <= truth_low && upper[i] >= truth_high,
                "{} lane{i}: [{},{}] excludes [{truth_low},{truth_high}]",
                case.id,
                lower[i],
                upper[i]
            );
            assert!(lower[i] <= value[i] && value[i] <= upper[i]);
            assert!(
                error[i]
                    >= (value[i] - truth_low)
                        .abs()
                        .max((value[i] - truth_high).abs()),
                "{} nominal error",
                case.id
            );
            assert!(
                quadrature[i] >= decode(lane.remainder.1, case.dtype),
                "{} lost remainder",
                case.id
            );
            assert!(quadrature[i] >= 0.0 && error[i] >= quadrature[i]);
            if case.m4 == 0 {
                assert_eq!(quadrature[i], 0.0);
            }
            if let Some((numerator, denominator)) = case.width_gate {
                assert!(
                    upper[i] - lower[i] <= numerator / denominator,
                    "{} width {} exceeds frozen gate",
                    case.id,
                    upper[i] - lower[i]
                );
            }
            if !lane.active {
                for output in outputs {
                    assert_eq!(fixture::read_bits(output)?[i], 0, "{} zero lane", case.id);
                }
            }
        }
        // Mutations at the API boundary cannot alter an already returned receipt.
        let before: Vec<_> = outputs
            .iter()
            .map(|field| fixture::read_bits(field))
            .collect::<numr::error::Result<_>>()?;
        for buffer in retained.iter().chain([&left, &right]) {
            if buffer.numel() == 0 {
                continue;
            }
            R::copy_to_device(
                &vec![0; buffer.numel() * buffer.dtype().size_in_bytes()],
                buffer.ptr(),
                client.device(),
            )?;
        }
        for (output, expected) in outputs.iter().zip(before) {
            assert_eq!(
                fixture::read_bits(output)?,
                expected,
                "{} retained storage",
                case.id
            );
        }
    }
    Ok(())
}
