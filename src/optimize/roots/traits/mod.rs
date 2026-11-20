//! Root finding trait definitions.

pub mod anderson;
mod broyden;
mod levenberg_marquardt;
mod newton;
mod powell_hybrid;

pub use anderson::{AndersonAlgorithms, AndersonConvergence, AndersonOptions};
pub use broyden::Broyden1Algorithms;
pub use levenberg_marquardt::LevenbergMarquardtAlgorithms;
pub use newton::NewtonSystemAlgorithms;
pub use powell_hybrid::PowellHybridAlgorithms;

pub mod bisection;
pub mod root_options;
pub mod root_result;
pub use bisection::{BatchedBisectionAlgorithms, BatchedBisectionOptions, BatchedBracketResult};
pub use root_options::{RootOptions, RootTensorResult};

pub mod batched_newton;
pub use batched_newton::{BatchedNewtonAlgorithms, BatchedNewtonOptions, BatchedNewtonResult};

mod certified_bisection;
mod certified_bisection_types;
pub(crate) mod certified_client;
mod certified_mapped_bisection;
mod certified_mapped_progress;
mod certified_mapped_types;
mod certified_root_tolerance;
mod certified_upper_bracket;
mod certified_upper_bracket_types;
pub use certified_bisection::{CertifiedBisectionAlgorithms, CertifiedBisectionOptions};
pub use certified_bisection_types::{
    CertifiedRootProbe, CertifiedRootQuery, CertifiedRootResult, CertifiedRootStatus,
    CertifiedRootValues,
};
pub use certified_mapped_bisection::CertifiedMappedBisectionAlgorithms;
pub use certified_mapped_progress::{CertifiedMappedRootProgress, CertifiedRootContinuation};
pub use certified_mapped_types::{
    CertifiedRootAdvanceAllowance, CertifiedRootDomainQuery, CertifiedRootMappedValues,
};
pub use certified_root_tolerance::CertifiedRootTolerance;
pub use certified_upper_bracket::{CertifiedUpperBracketAlgorithms, CertifiedUpperBracketOptions};
pub use certified_upper_bracket_types::{CertifiedUpperBracketResult, CertifiedUpperBracketStatus};
