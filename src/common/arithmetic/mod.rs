//! Shared arithmetic checks and numerical bounds.

pub(crate) mod bounds;
pub(crate) mod envelope;
pub(crate) mod error;
pub(crate) mod interval;
pub(crate) mod precision;
pub(crate) mod reduction;

mod checked;
mod context;
mod expansion;

pub use checked::with_interval_arithmetic;
pub use context::IntervalContext;
pub use error::{ArithmeticError, ArithmeticResult};
pub use expansion::CertifiedExpansion;
pub use interval::Interval;
