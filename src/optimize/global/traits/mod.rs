//! Trait definitions for global optimization algorithms.

pub mod basin_hopping;
pub mod differential_evolution;
pub mod dual_annealing;
mod options;
pub mod simulated_annealing;

pub use basin_hopping::{BasinHoppingAlgorithms, BasinHoppingResult};
pub use differential_evolution::{DifferentialEvolutionAlgorithms, DifferentialEvolutionResult};
pub use dual_annealing::{DualAnnealingAlgorithms, DualAnnealingResult};
pub use options::GlobalOptions;
pub use simulated_annealing::{SimulatedAnnealingAlgorithms, SimulatedAnnealingResult};
