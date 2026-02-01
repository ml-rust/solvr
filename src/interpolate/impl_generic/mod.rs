pub mod akima;
pub mod cubic_spline;
pub mod interp1d;
pub mod interpnd;
pub mod pchip;

pub use akima::akima_slopes;
pub use cubic_spline::cubic_spline_coefficients;
pub use interp1d::interp1d_evaluate;
pub use interpnd::interpnd_evaluate;
pub use pchip::pchip_slopes;
