//! Public expansion arithmetic retains cancellation residuals across native operations.

#[path = "certified_expansion/admission.rs"]
mod admission;
#[path = "certified_expansion/arithmetic.rs"]
mod arithmetic;
#[path = "certified_expansion/backends.rs"]
mod backends;
#[path = "certified_expansion/fixture.rs"]
mod fixture;
#[path = "certified_expansion/interval_lift.rs"]
mod interval_lift;
#[path = "certified_expansion/ownership.rs"]
mod ownership;
#[path = "certified_expansion/product_identity.rs"]
mod product_identity;
#[path = "certified_expansion/reference_f32.rs"]
mod reference_f32;
#[path = "certified_expansion/reference_f64.rs"]
mod reference_f64;
#[path = "certified_expansion/sqrt.rs"]
mod sqrt;
