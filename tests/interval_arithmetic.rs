//! Public interval arithmetic uses native clients and exact reference bounds.

#[path = "interval_arithmetic/admission.rs"]
mod admission;
#[path = "interval_arithmetic/arithmetic.rs"]
mod arithmetic;
#[cfg(any(feature = "cuda", feature = "wgpu"))]
#[path = "interval_arithmetic/backends.rs"]
mod backends;
#[path = "interval_arithmetic/cpu.rs"]
mod cpu;
#[path = "interval_arithmetic/division_identity.rs"]
mod division_identity;
#[path = "interval_arithmetic/exactness.rs"]
mod exactness;
#[path = "interval_arithmetic/ownership.rs"]
mod ownership;
#[path = "interval_arithmetic/product_ratio_lower.rs"]
mod product_ratio_lower;
#[path = "interval_arithmetic/product_ratio_lower_admission.rs"]
mod product_ratio_lower_admission;
#[path = "interval_arithmetic/sign.rs"]
mod sign;
#[path = "interval_arithmetic/support.rs"]
mod support;

#[path = "interval_arithmetic/lattice_exactness.rs"]
mod lattice_exactness;
