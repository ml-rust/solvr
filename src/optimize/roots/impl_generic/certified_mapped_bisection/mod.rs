//! Persistent certified root refinement in caller-supplied stopping coordinates.

mod controller;
mod policy;
mod state;

pub use controller::{
    bisect_certified_mapped_impl, refresh_certified_mapped_selected_impl,
    resume_certified_mapped_impl, resume_certified_mapped_selected_impl,
};

mod selection;

#[cfg(test)]
mod selected_tests;
