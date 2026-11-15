//! Actual-client receipt execution or explicit checked arithmetic limitations.

#[cfg(any(feature = "cuda", feature = "wgpu"))]
#[path = "linear_receipts/backend.rs"]
mod backend;
#[cfg(any(feature = "cuda", feature = "wgpu"))]
#[path = "linear_receipts/cases/mod.rs"]
mod cases;
#[cfg(any(feature = "cuda", feature = "wgpu"))]
#[path = "linear_receipts/check.rs"]
mod check;
#[cfg(any(feature = "cuda", feature = "wgpu"))]
#[path = "linear_receipts/fixture.rs"]
mod fixture;
#[cfg(any(feature = "cuda", feature = "wgpu"))]
#[path = "linear_receipts/observe.rs"]
mod observe;
#[cfg(any(feature = "cuda", feature = "wgpu"))]
#[path = "linear_receipts/suite.rs"]
mod suite;
