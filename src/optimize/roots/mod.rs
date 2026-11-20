//! Independent scalar roots and multivariate nonlinear systems.

mod cpu;

#[cfg(feature = "cuda")]
mod cuda;

#[cfg(feature = "wgpu")]
mod wgpu;

pub mod impl_generic;
pub mod traits;

// Re-export traits
pub use traits::{
    AndersonAlgorithms, AndersonConvergence, AndersonOptions, BatchedBisectionAlgorithms,
    BatchedBisectionOptions, BatchedBracketResult, BatchedNewtonAlgorithms, BatchedNewtonOptions,
    BatchedNewtonResult, Broyden1Algorithms, LevenbergMarquardtAlgorithms, NewtonSystemAlgorithms,
    PowellHybridAlgorithms, RootOptions, RootTensorResult,
};

// Re-export result type from impl_generic
pub use impl_generic::TensorRootResult;
pub use traits::{
    CertifiedBisectionAlgorithms, CertifiedBisectionOptions, CertifiedRootProbe,
    CertifiedRootQuery, CertifiedRootResult, CertifiedRootStatus, CertifiedRootValues,
    CertifiedUpperBracketAlgorithms, CertifiedUpperBracketOptions, CertifiedUpperBracketResult,
    CertifiedUpperBracketStatus,
};
pub use traits::{
    CertifiedMappedBisectionAlgorithms, CertifiedMappedRootProgress, CertifiedRootAdvanceAllowance,
    CertifiedRootContinuation, CertifiedRootDomainQuery, CertifiedRootMappedValues,
    CertifiedRootTolerance,
};
