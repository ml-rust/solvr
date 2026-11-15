//! Fixed Simpson integration with conditional certified error bounds.

mod admission;
mod assemble;
mod error;
mod evaluate;
mod integration;
mod queries;

pub use integration::certified_simpson_impl;
