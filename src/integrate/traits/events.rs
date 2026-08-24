//! Event function traits and types for ODE solvers.
//!
//! Event functions allow detection of specific conditions during integration,
//! such as zero-crossings, impacts, or state thresholds.
//!
//! # Example
//!
//! ```ignore
//! # use numr::runtime::Runtime;
//! # use numr::ops::TensorOps;
//! # use numr::runtime::RuntimeClient;
//! # use numr::tensor::Tensor;
//! # use numr::error::Result;
//! use solvr::integrate::traits::EventFunction;
//!
//! // Detect when y[0] crosses zero (e.g., ball hitting ground)
//! struct GroundImpact;
//!
//! impl<R: Runtime<DType = DType>, C> EventFunction<R, C> for GroundImpact
//! where
//!     C: TensorOps<R> + RuntimeClient<R>,
//! {
//!     fn evaluate(&self, _client: &C, _t: f64, y: &Tensor<R>) -> Result<f64> {
//!         // g(t, y) = y[0], event when height = 0
//!         let val = y.get_scalar(0)?;
//!         Ok(val)
//!     }
//! }
//! ```
use crate::DType;

use numr::error::Result;
use numr::ops::TensorOps;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Trait for event functions used in ODE integration.
///
/// An event function g(t, y) triggers when it crosses zero.
/// The solver detects sign changes and refines the exact crossing time.
pub trait EventFunction<R: Runtime<DType = DType>, C>: Send + Sync
where
    C: TensorOps<R> + RuntimeClient<R>,
{
    /// Evaluate the event function at (t, y).
    ///
    /// # Arguments
    /// * `client` - Runtime client for tensor operations
    /// * `t` - Current time (scalar)
    /// * `y` - Current state (1-D tensor)
    ///
    /// # Returns
    /// The value of the event function. An event is detected when this value
    /// crosses zero between steps.
    fn evaluate(&self, client: &C, t: f64, y: &Tensor<R>) -> Result<f64>;
}

/// Wrapper for closure-based event functions.
///
/// Allows using closures as event functions without defining a struct.
pub struct EventFn<R, C, F>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + RuntimeClient<R>,
    F: Fn(&C, f64, &Tensor<R>) -> Result<f64> + Send + Sync,
{
    f: F,
    _marker: std::marker::PhantomData<(R, C)>,
}

impl<R, C, F> EventFn<R, C, F>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + RuntimeClient<R>,
    F: Fn(&C, f64, &Tensor<R>) -> Result<f64> + Send + Sync,
{
    /// Create a new event function from a closure.
    pub fn new(f: F) -> Self {
        Self {
            f,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<R, C, F> EventFunction<R, C> for EventFn<R, C, F>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + RuntimeClient<R>,
    F: Fn(&C, f64, &Tensor<R>) -> Result<f64> + Send + Sync,
{
    fn evaluate(&self, client: &C, t: f64, y: &Tensor<R>) -> Result<f64> {
        (self.f)(client, t, y)
    }
}

/// Collection of event functions with their specifications.
///
/// Manages multiple event functions and their detection parameters.
pub struct EventSet<'a, R, C>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + RuntimeClient<R>,
{
    /// Event functions.
    pub functions: Vec<&'a dyn EventFunction<R, C>>,
    /// Specifications for each event function.
    pub specs: Vec<crate::integrate::ode::EventSpec>,
}

impl<'a, R, C> EventSet<'a, R, C>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + RuntimeClient<R>,
{
    /// Create a new empty event set.
    pub fn new() -> Self {
        Self {
            functions: Vec::new(),
            specs: Vec::new(),
        }
    }

    /// Add an event function with default specification.
    pub fn add(&mut self, event: &'a dyn EventFunction<R, C>) {
        self.functions.push(event);
        self.specs.push(crate::integrate::ode::EventSpec::default());
    }

    /// Add an event function with custom specification.
    pub fn add_with_spec(
        &mut self,
        event: &'a dyn EventFunction<R, C>,
        spec: crate::integrate::ode::EventSpec,
    ) {
        self.functions.push(event);
        self.specs.push(spec);
    }

    /// Check if the event set is empty.
    pub fn is_empty(&self) -> bool {
        self.functions.is_empty()
    }

    /// Get the number of event functions.
    pub fn len(&self) -> usize {
        self.functions.len()
    }

    /// Evaluate all event functions at (t, y).
    pub fn evaluate_all(&self, client: &C, t: f64, y: &Tensor<R>) -> Result<Vec<f64>> {
        self.functions
            .iter()
            .map(|f| f.evaluate(client, t, y))
            .collect()
    }
}

impl<'a, R, C> Default for EventSet<'a, R, C>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + RuntimeClient<R>,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    fn setup() -> (CpuDevice, CpuClient) {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        (device, client)
    }

    #[test]
    fn test_event_fn_closure() {
        let (device, client) = setup();

        // Event function that detects y[0] = 0
        let event = EventFn::new(|_c: &CpuClient, _t: f64, y: &Tensor<CpuRuntime>| {
            let y_data: Vec<f64> = y.to_vec();
            Ok(y_data[0])
        });

        let y = Tensor::<CpuRuntime>::from_slice(&[1.0, 2.0], &[2], &device).unwrap();
        let val = event.evaluate(&client, 0.0, &y).unwrap();
        assert!((val - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_event_set() {
        use crate::integrate::ode::EventSpec;
        let (device, client) = setup();

        let event1 = EventFn::new(|_c: &CpuClient, _t: f64, y: &Tensor<CpuRuntime>| {
            let y_data: Vec<f64> = y.to_vec();
            Ok(y_data[0])
        });

        let event2 = EventFn::new(|_c: &CpuClient, _t: f64, y: &Tensor<CpuRuntime>| {
            let y_data: Vec<f64> = y.to_vec();
            Ok(y_data[1] - 1.0) // Detect y[1] = 1
        });

        let mut event_set = EventSet::<CpuRuntime, CpuClient>::new();
        event_set.add(&event1);
        event_set.add_with_spec(&event2, EventSpec::terminal());

        assert_eq!(event_set.len(), 2);
        assert!(!event_set.is_empty());

        let y = Tensor::<CpuRuntime>::from_slice(&[0.5, 1.5], &[2], &device).unwrap();
        let vals = event_set.evaluate_all(&client, 0.0, &y).unwrap();

        assert!((vals[0] - 0.5).abs() < 1e-10);
        assert!((vals[1] - 0.5).abs() < 1e-10);
    }
}
