use crate::{check, cpu, fixture, observe};
use numr::prelude::*;
use solvr::interpolate::{Interp1dAlgorithms, InterpolateError, LinearInterp1dDiagnostics};
use std::io::{BufWriter, Write};

const HEADER: &str = "backend,dtype,case_id,layout,query_index,source_query_index,status,failed_probes,legacy_status,legacy_error,diagnostic_error,x_bits,y_bits,query_bits,value_bits,absolute_error_bits,legacy_bits,lower_index,upper_index,at_lower,at_upper,constant";

fn csv_row(writer: &mut impl Write, fields: &[String]) -> fixture::Result {
    for (index, field) in fields.iter().enumerate() {
        if index > 0 {
            write!(writer, ",")?;
        }
        if field
            .chars()
            .any(|value| matches!(value, ',' | '"' | '\n' | '\r'))
        {
            write!(writer, "\"{}\"", field.replace('"', "\"\""))?;
        } else {
            write!(writer, "{field}")?;
        }
    }
    writeln!(writer)?;
    Ok(())
}

fn rows(
    writer: &mut impl Write,
    backend: &str,
    case: &fixture::Case,
    layout: usize,
    observed: &observe::Observation,
) -> fixture::Result {
    let (status, probes, diagnostic_error) = match &observed.diagnostic {
        Ok(_) if observed.source_queries.is_empty() => ("empty", String::new(), String::new()),
        Ok(_) => ("success", String::new(), String::new()),
        Err(error) => {
            let probes =
                if let InterpolateError::UnsupportedArithmetic { failed_probes, .. } = error {
                    failed_probes.join("|")
                } else {
                    String::new()
                };
            (fixture::error_label(error), probes, error.to_string())
        }
    };
    let (legacy_status, legacy_error) = match &observed.legacy {
        Ok(_) => ("success", String::new()),
        Err(error) => (fixture::error_label(error), error.to_string()),
    };
    let x_bits = case
        .x
        .iter()
        .map(|value| fixture::bits(*value, case.dtype))
        .collect::<Vec<_>>()
        .join("|");
    let y_bits = case
        .y
        .iter()
        .map(|value| fixture::bits(*value, case.dtype))
        .collect::<Vec<_>>()
        .join("|");
    for lane in 0..observed.source_queries.len().max(1) {
        let source = observed.source_queries.get(lane).copied();
        let mut row = vec![
            backend.into(),
            format!("{:?}", case.dtype),
            case.id.into(),
            layout.to_string(),
            source.map_or_else(String::new, |_| lane.to_string()),
            source.map_or_else(String::new, |source| source.to_string()),
            status.into(),
            probes.clone(),
            legacy_status.into(),
            legacy_error.clone(),
            diagnostic_error.clone(),
            x_bits.clone(),
            y_bits.clone(),
            source.map_or_else(String::new, |source| {
                fixture::bits(case.query[source], case.dtype)
            }),
        ];
        let legacy_bits = observed
            .legacy
            .as_ref()
            .ok()
            .and_then(|values| values.get(lane))
            .map_or_else(String::new, |value| fixture::bits(*value, case.dtype));
        if let (Ok(output), Some(_)) = (&observed.diagnostic, source) {
            row.extend([
                fixture::bits(
                    *output.values.get(lane).ok_or("Missing receipt value")?,
                    case.dtype,
                ),
                fixture::bits(
                    *output
                        .absolute_error
                        .get(lane)
                        .ok_or("Missing receipt bound")?,
                    case.dtype,
                ),
                legacy_bits,
                output
                    .lower
                    .get(lane)
                    .ok_or("Missing lower index")?
                    .to_string(),
                output
                    .upper
                    .get(lane)
                    .ok_or("Missing upper index")?
                    .to_string(),
                output
                    .at_lower
                    .get(lane)
                    .ok_or("Missing lower equality mask")?
                    .to_string(),
                output
                    .at_upper
                    .get(lane)
                    .ok_or("Missing upper equality mask")?
                    .to_string(),
                output
                    .constant
                    .get(lane)
                    .ok_or("Missing constant mask")?
                    .to_string(),
            ]);
        } else {
            row.extend([String::new(), String::new(), legacy_bits]);
            row.extend(vec![String::new(); 5]);
        }
        csv_row(writer, &row)?;
    }
    Ok(())
}

fn backend<R, C>(
    writer: &mut impl Write,
    name: &str,
    client: &C,
    device: &R::Device,
    dtypes: &[DType],
) -> fixture::Result
where
    R: Runtime<DType = DType>,
    C: Interp1dAlgorithms<R> + LinearInterp1dDiagnostics<R>,
{
    let mut errors = Vec::new();
    for &dtype in dtypes {
        for case in fixture::cases(dtype) {
            for layout in fixture::LAYOUTS {
                let observed = observe::run(client, device, &case, layout)?;
                if let Err(error) = check::observation(&case, &observed, name == "wgpu") {
                    errors.push(error.to_string());
                }
                rows(writer, name, &case, layout, &observed)?;
            }
        }
    }
    writer.flush()?;
    if !errors.is_empty() {
        return Err(errors.join("\n").into());
    }
    Ok(())
}

pub fn run() -> fixture::Result {
    let mut writer = BufWriter::new(std::io::stdout().lock());
    writeln!(writer, "{HEADER}")?;
    let requested = std::env::args().nth(1);
    match requested.as_deref() {
        None | Some("cpu") => {
            let (client, device) = cpu::setup();
            backend(
                &mut writer,
                "cpu",
                &client,
                &device,
                &[DType::F32, DType::F64],
            )
        }
        #[cfg(feature = "cuda")]
        Some("cuda") => {
            let device = CudaDevice::new(0);
            let client = CudaClient::new(device.clone())?;
            client.synchronize();
            backend(
                &mut writer,
                "cuda",
                &client,
                &device,
                &[DType::F32, DType::F64],
            )
        }
        #[cfg(feature = "wgpu")]
        Some("wgpu") => {
            let device = WgpuDevice::new(0);
            let client = WgpuClient::new(device.clone())?;
            backend(&mut writer, "wgpu", &client, &device, &[DType::F32])
        }
        requested => Err(format!(
            "backend={requested:?}: select cpu or enable the requested cuda/wgpu feature"
        )
        .into()),
    }
}
