pub mod akima;
pub mod cubic_spline;
pub mod interp1d;
pub mod interpnd;
pub mod pchip;

pub use akima::AkimaAlgorithms;
pub use cubic_spline::{CubicSplineAlgorithms, SplineBoundary};
pub use interp1d::{Interp1dAlgorithms, InterpMethod};
pub use interpnd::{ExtrapolateMode, InterpNdAlgorithms, InterpNdMethod};
pub use pchip::PchipAlgorithms;
