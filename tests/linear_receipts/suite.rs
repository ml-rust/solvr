use super::{check, fixture, observe};
use numr::prelude::*;
use solvr::interpolate::{Interp1dAlgorithms, LinearInterp1dDiagnostics};

pub fn run<R, C>(
    client: &C,
    device: &R::Device,
    dtype: DType,
    backend_defined: bool,
) -> fixture::Result
where
    R: Runtime<DType = DType>,
    C: Interp1dAlgorithms<R> + LinearInterp1dDiagnostics<R>,
{
    for case in fixture::cases(dtype) {
        for layout in fixture::LAYOUTS {
            let observed = observe::run(client, device, &case, layout)?;
            check::observation(&case, &observed, backend_defined)?;
        }
    }
    Ok(())
}
