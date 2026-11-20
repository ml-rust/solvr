//! Mapped stopping and retained certified-root continuation.

#[path = "certified_mapped_bisection/admission.rs"]
mod admission;
#[path = "certified_mapped_bisection/backends.rs"]
mod backends;
#[path = "certified_mapped_bisection/behavior.rs"]
mod behavior;
#[path = "certified_mapped_bisection/fixture.rs"]
mod fixture;
#[path = "certified_mapped_bisection/ownership.rs"]
mod ownership;
#[path = "certified_mapped_bisection/resume.rs"]
mod resume;
#[path = "certified_bisection/shapes.rs"]
#[allow(dead_code)]
mod shapes;
#[path = "certified_mapped_bisection/slack_secant.rs"]
mod slack_secant;
#[path = "certified_mapped_bisection/tolerance.rs"]
mod tolerance;
