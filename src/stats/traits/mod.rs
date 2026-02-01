//! Statistical algorithm traits.
//!
//! This module defines three focused trait groups:
//! - DescriptiveStatisticsAlgorithms - Computing statistics (mean, variance, skewness, etc.)
//! - HypothesisTestingAlgorithms - Statistical hypothesis tests
//! - RegressionAlgorithms - Regression analysis

mod descriptive;
mod hypothesis;
mod regression;

pub use descriptive::DescriptiveStatisticsAlgorithms;
pub use hypothesis::HypothesisTestingAlgorithms;
pub use regression::RegressionAlgorithms;
