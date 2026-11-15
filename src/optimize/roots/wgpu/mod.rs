//! WebGPU implementations for root finding algorithms.

mod anderson;
mod broyden;
mod levenberg_marquardt;
mod newton;
mod powell_hybrid;

mod bisection;

pub mod batched_newton;

mod certified_bisection;
mod certified_mapped_bisection;
mod certified_upper_bracket;
