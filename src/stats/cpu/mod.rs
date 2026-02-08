//! CPU implementation of statistical algorithms.
#![allow(unused_imports)]

mod descriptive;
mod hypothesis;
mod information;
mod regression;
mod robust;

pub use descriptive::*;
pub use hypothesis::*;
pub use information::*;
pub use regression::*;
pub use robust::*;
