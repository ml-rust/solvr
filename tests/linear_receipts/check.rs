use super::fixture::{self, Case, Result};
use super::observe::Observation;
use solvr::interpolate::InterpolateError;

pub fn observation(case: &Case, observed: &Observation, backend_defined: bool) -> Result {
    if backend_defined {
        let Err(error) = &observed.diagnostic else {
            return Err(format!(
                "{}: backend-defined arithmetic produces no checked receipt",
                case.id
            )
            .into());
        };
        if !fixture::arithmetic_limit(error, case.dtype)? {
            return Err(format!(
                "{}: expected typed provider-contract rejection, got {error}",
                case.id
            )
            .into());
        }
        let InterpolateError::UnsupportedArithmetic { failed_probes, .. } = error else {
            return Err("Missing typed provider-contract rejection".into());
        };
        if failed_probes.as_slice() != ["provider_contract"] {
            return Err(format!(
                "{}: expected only provider_contract, got {failed_probes:?}",
                case.id
            )
            .into());
        }
        return Ok(());
    }
    if let Err(error) = &observed.diagnostic
        && fixture::arithmetic_limit(error, case.dtype)?
    {
        return Err(format!(
            "{}: default IEEE basic provider requires receipt coverage: {error}",
            case.id
        )
        .into());
    }
    match (&observed.legacy, case.legacy_error) {
        (Ok(_), None) => {}
        (Err(InterpolateError::NumericalError { message }), Some(fragment))
            if message.contains(fragment) => {}
        (actual, expected) => {
            return Err(format!(
                "{} legacy result={actual:?},expected context={expected:?}",
                case.id
            )
            .into());
        }
    }
    let outside = observed.source_queries.iter().any(|index| {
        let value = case.query[*index];
        value < case.x[0] || value > case.x[case.x.len() - 1]
    });
    if outside {
        let Err(InterpolateError::OutOfDomain {
            point, min, max, ..
        }) = &observed.diagnostic
        else {
            return Err(format!("{} requires closed-support error", case.id).into());
        };
        assert!(
            observed
                .source_queries
                .iter()
                .any(|index| fixture::bits(case.query[*index], case.dtype)
                    == fixture::bits(*point, case.dtype))
        );
        assert_eq!(*min, case.x[0]);
        assert_eq!(*max, case.x[case.x.len() - 1]);
        return Ok(());
    }
    if case.legacy_error.is_some() {
        if observed.diagnostic.as_ref().err().map(fixture::error_label) != Some("numerical") {
            return Err(format!("{} requires diagnostic numerical rejection", case.id).into());
        }
        return Ok(());
    }
    let output = observed
        .diagnostic
        .as_ref()
        .map_err(|error| error.to_string())?;
    let legacy = observed
        .legacy
        .as_ref()
        .map_err(|error| error.to_string())?;
    assert_eq!(output.values.len(), observed.source_queries.len());
    assert_eq!(legacy.len(), output.values.len());
    for (lane, source) in observed.source_queries.iter().enumerate() {
        let (lower, upper, at_lower, at_upper, constant) = case.metadata[*source];
        assert_eq!(output.lower[lane], lower as i64, "{} lane={lane}", case.id);
        assert_eq!(output.upper[lane], upper as i64, "{} lane={lane}", case.id);
        assert_eq!(output.at_lower[lane], if at_lower { 1.0 } else { 0.0 });
        assert_eq!(output.at_upper[lane], if at_upper { 1.0 } else { 0.0 });
        assert_eq!(output.constant[lane], if constant { 1.0 } else { 0.0 });
        assert!(output.values[lane].is_finite());
        assert!(output.absolute_error[lane].is_finite() && output.absolute_error[lane] >= 0.0);
        assert_eq!(
            fixture::bits(output.values[lane], case.dtype),
            fixture::bits(legacy[lane], case.dtype),
            "{} lane={lane}",
            case.id
        );
        if at_lower || at_upper || constant {
            assert_eq!(output.absolute_error[lane], 0.0);
            let stored = if at_upper {
                case.y[upper]
            } else {
                case.y[lower]
            };
            assert_eq!(
                fixture::bits(output.values[lane], case.dtype),
                fixture::bits(stored, case.dtype)
            );
        }
    }
    Ok(())
}
