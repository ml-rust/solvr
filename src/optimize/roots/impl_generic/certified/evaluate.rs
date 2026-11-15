//! Isolate callback query and result storage from the controller.

use super::native::RootClient;
use super::{metadata, native};
use crate::DType;
use crate::optimize::roots::traits::{CertifiedRootQuery, CertifiedRootValues};
use crate::optimize::{OptimizeError, OptimizeResult};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

pub(crate) fn callback<R, C, F>(
    client: &C,
    f: &F,
    owners: &Tensor<R>,
    point: &Tensor<R>,
) -> OptimizeResult<CertifiedRootValues<R>>
where
    R: Runtime<DType = DType>,
    C: RootClient<R>,
    F: Fn(&CertifiedRootQuery<R>) -> numr::error::Result<CertifiedRootValues<R>>,
{
    let query = CertifiedRootQuery {
        owner: native::owned(client, owners)?,
        point: native::owned(client, point)?,
    };
    let values = f(&query)?;
    for (name, value) in [
        ("callback lower", &values.lower),
        ("callback upper", &values.upper),
    ] {
        metadata::tensor(client, value, point.numel(), point.dtype(), name)?;
    }
    // Capture both outputs before another callback can reuse either buffer.
    let lower = native::owned(client, &values.lower)?;
    let upper = native::owned(client, &values.upper)?;
    let invalid = client.maximum(
        &native::nonfinite(client, &lower)?,
        &native::nonfinite(client, &upper)?,
    )?;
    let invalid = client.maximum(&invalid, &client.gt(&lower, &upper)?)?;
    if native::any(client, &invalid)? {
        return Err(OptimizeError::InvalidInput {
            context: "Certified root callback contains nonfinite or inverted bounds. Return finite ordered physical residual bounds.".into(),
        });
    }
    Ok(CertifiedRootValues { lower, upper })
}
