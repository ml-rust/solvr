//! Mapped convergence coordinates and additional continuation allowances.

use crate::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Original owners and certified parameter domains supplied to a mapping callback.
#[derive(Debug)]
pub struct CertifiedRootDomainQuery<R: Runtime<DType = DType>> {
    /// Original invocation-local I64 indices.
    pub owner: Tensor<R>,
    /// Parameter lower endpoints in the original numerical dtype.
    pub lower: Tensor<R>,
    /// Parameter upper endpoints in the original numerical dtype.
    pub upper: Tensor<R>,
}

/// Finite ordered bounds enclosing the mapped quantity throughout each supplied domain.
#[derive(Debug)]
pub struct CertifiedRootMappedValues<R: Runtime<DType = DType>> {
    pub lower: Tensor<R>,
    pub upper: Tensor<R>,
}

/// Additional work permitted during one continuation call.
///
/// These allowances never increase the original session ceilings.
#[derive(Clone, Copy, Debug)]
pub struct CertifiedRootAdvanceAllowance {
    pub max_iterations: usize,
    pub max_points: usize,
}
