//! Frozen exact references and native callback contracts for certified integration.

#[path = "certified_simpson/admission.rs"]
mod admission;
#[cfg(any(feature = "cuda", feature = "wgpu"))]
#[path = "certified_simpson/backends.rs"]
mod backends;
#[path = "certified_simpson/callback.rs"]
mod callback;
#[path = "certified_simpson/cpu.rs"]
mod cpu;
#[path = "certified_simpson/f32_limits.rs"]
mod f32_limits;
#[path = "certified_simpson/f32_values.rs"]
mod f32_values;
#[path = "certified_simpson/f64_limits.rs"]
mod f64_limits;
#[path = "certified_simpson/f64_precision.rs"]
mod f64_precision;
#[path = "certified_simpson/f64_values.rs"]
mod f64_values;
#[path = "certified_simpson/fixture.rs"]
mod fixture;
#[cfg(target_arch = "x86_64")]
#[path = "certified_simpson/modes.rs"]
mod modes;
#[path = "certified_simpson/ownership.rs"]
mod ownership;
#[path = "certified_simpson/suite.rs"]
mod suite;
