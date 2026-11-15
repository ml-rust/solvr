use super::{check, cpu::setup, fixture::*, observe, suite};
use numr::prelude::{CpuRuntime, DType};
use solvr::interpolate::{Interp1dAlgorithms, InterpMethod, LinearInterp1dDiagnostics};

#[test]
fn configured_cpu_receipts_preserve_values_metadata_and_exact_branches() -> Result {
    let (client, device) = setup();
    for dtype in [DType::F32, DType::F64] {
        assert_eq!(cases(dtype).len(), 20);
        suite::run(&client, &device, dtype, false)?;
    }
    Ok(())
}

#[test]
fn strided_original_inputs_preserve_receipt_metadata_and_legacy_bits() -> Result {
    let (client, device) = setup();
    for dtype in [DType::F32, DType::F64] {
        let case = cases(dtype)
            .into_iter()
            .find(|case| case.id.starts_with("unit_knots_shuffled"))
            .ok_or("Missing shuffled receipt input")?;
        let strided = |data: &[f64]| {
            let interleaved = data
                .iter()
                .flat_map(|value| [99.0, *value])
                .collect::<Vec<_>>();
            let base = tensor::<CpuRuntime>(&interleaved, &[data.len(), 2], dtype, &device)?;
            Ok::<_, numr::error::Error>(base.narrow(1, 1, 1)?.squeeze(Some(1)))
        };
        let x = strided(&case.x)?;
        let y = strided(&case.y)?;
        let query = strided(&case.query)?;
        assert!(!x.is_contiguous() && !y.is_contiguous() && !query.is_contiguous());
        let legacy = client
            .interp1d(&x, &y, &query, InterpMethod::Linear)
            .and_then(|tensor| Ok(values(&tensor)?));
        let diagnostic = client
            .interp1d_linear_with_receipts(&x, &y, &query)
            .and_then(|receipt| observe::project(&receipt, case.query.len(), dtype));
        check::observation(
            &case,
            &observe::Observation {
                source_queries: (0..case.query.len()).collect(),
                legacy,
                diagnostic,
            },
            false,
        )?;
    }
    Ok(())
}
