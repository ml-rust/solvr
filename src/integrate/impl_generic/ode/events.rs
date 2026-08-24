//! Event detection and root refinement for ODE solvers.
//!
//! Provides zero-crossing detection within integration steps and
//! accurate root refinement using Brent's method with dense output.
use crate::DType;

use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use crate::integrate::error::IntegrateResult;
use crate::integrate::ode::{EventDirection, EventOptions, EventRecord, EventSpec};
use crate::integrate::traits::EventFunction;
use crate::optimize::scalar::{ScalarOptions, brentq};

use super::dense_output::{DenseOutputStep, dense_eval};

/// Result of checking events within a step.
pub struct EventCheckResult<R: Runtime<DType = DType>> {
    /// Events detected in this step (sorted by time).
    pub events: Vec<EventRecord<R>>,

    /// Whether a terminal event was found.
    pub has_terminal: bool,

    /// Index of the first terminal event (if any).
    pub terminal_index: Option<usize>,

    /// Time of the first terminal event (if any).
    pub terminal_time: Option<f64>,
}

/// Check for events within a step using sign-change detection.
///
/// This function evaluates all event functions at both endpoints of the step,
/// detects sign changes (respecting direction filters), and refines the exact
/// event time using root finding with dense output interpolation.
///
/// # Arguments
///
/// * `client` - Runtime client for tensor operations
/// * `event_fns` - Slice of event functions to evaluate
/// * `specs` - Event specifications (terminal, direction)
/// * `step` - Dense output step containing interpolation data
/// * `g_old` - Event function values at t_old
/// * `opts` - Event detection options
///
/// # Returns
///
/// `EventCheckResult` containing all detected events and terminal status.
pub fn check_events<R, C, E>(
    client: &C,
    event_fns: &[&E],
    specs: &[EventSpec],
    step: &DenseOutputStep<R>,
    g_old: &[f64],
    opts: &EventOptions,
) -> IntegrateResult<EventCheckResult<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
    E: EventFunction<R, C> + ?Sized,
{
    let mut events = Vec::new();
    let mut has_terminal = false;
    let mut terminal_index = None;
    let mut terminal_time = None;

    // Evaluate events at the new endpoint
    let g_new = evaluate_events(client, event_fns, step.t_new, &step.y_new)?;

    // Check each event for sign change
    for (i, ((&g0, &g1), spec)) in g_old.iter().zip(g_new.iter()).zip(specs.iter()).enumerate() {
        if let Some(event) = detect_sign_change(client, event_fns[i], step, i, g0, g1, spec, opts)?
        {
            // Check if this event is terminal and comes before any previous terminal
            if spec.terminal && (!has_terminal || event.t < terminal_time.unwrap_or(f64::MAX)) {
                terminal_time = Some(event.t);
                terminal_index = Some(i);
                has_terminal = true;
            }
            events.push(event);
        }
    }

    // Sort events by time
    events.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap_or(std::cmp::Ordering::Equal));

    Ok(EventCheckResult {
        events,
        has_terminal,
        terminal_index,
        terminal_time,
    })
}

/// Evaluate all event functions at a given (t, y).
pub fn evaluate_events<R, C, E>(
    client: &C,
    event_fns: &[&E],
    t: f64,
    y: &Tensor<R>,
) -> IntegrateResult<Vec<f64>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + RuntimeClient<R>,
    E: EventFunction<R, C> + ?Sized,
{
    let mut values = Vec::with_capacity(event_fns.len());
    for event_fn in event_fns {
        let val = event_fn.evaluate(client, t, y).map_err(to_integrate_err)?;
        values.push(val);
    }
    Ok(values)
}

/// Detect a sign change and refine the event time if one is found.
#[allow(clippy::too_many_arguments)]
fn detect_sign_change<R, C, E>(
    client: &C,
    event_fn: &E,
    step: &DenseOutputStep<R>,
    event_index: usize,
    g_old: f64,
    g_new: f64,
    spec: &EventSpec,
    opts: &EventOptions,
) -> IntegrateResult<Option<EventRecord<R>>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
    E: EventFunction<R, C> + ?Sized,
{
    // Check for sign change
    let sign_change = g_old * g_new < 0.0;
    if !sign_change {
        return Ok(None);
    }

    // Check direction filter
    let direction_ok = match spec.direction {
        EventDirection::Any => true,
        EventDirection::Increasing => g_old < 0.0 && g_new > 0.0,
        EventDirection::Decreasing => g_old > 0.0 && g_new < 0.0,
    };

    if !direction_ok {
        return Ok(None);
    }

    // Refine the event time using Brent's method with dense output
    let (t_event, y_event, g_event) =
        refine_event_time(client, event_fn, step, g_old, g_new, opts)?;

    Ok(Some(EventRecord {
        t: t_event,
        y: y_event,
        event_index,
        event_value: g_event,
    }))
}

/// Refine the exact event time using Brent's method.
///
/// Uses the dense output to evaluate g(t, y(t)) at arbitrary points
/// within the step for accurate root finding.
fn refine_event_time<R, C, E>(
    client: &C,
    event_fn: &E,
    step: &DenseOutputStep<R>,
    g_old: f64,
    g_new: f64,
    opts: &EventOptions,
) -> IntegrateResult<(f64, Tensor<R>, f64)>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
    E: EventFunction<R, C> + ?Sized,
{
    // Define the event function for root finding
    // g(t) = event_fn(t, y_interp(t))
    let event_at_t = |t: f64| -> f64 {
        // Interpolate y at time t
        let y_interp = match dense_eval(client, step, t) {
            Ok(y) => y,
            Err(_) => return f64::NAN,
        };

        // Evaluate event function
        event_fn.evaluate(client, t, &y_interp).unwrap_or(f64::NAN)
    };

    // Use Brent's method for root finding
    let scalar_opts = ScalarOptions {
        max_iter: opts.max_root_iter,
        tol: opts.root_tol,
        rtol: opts.root_tol,
    };

    // Ensure we have proper bracket (swap if needed)
    let (a, b) = if step.t_old < step.t_new {
        (step.t_old, step.t_new)
    } else {
        (step.t_new, step.t_old)
    };

    // Handle edge cases where root is at endpoint
    if g_old.abs() < opts.root_tol {
        return Ok((step.t_old, step.y_old.clone(), g_old));
    }
    if g_new.abs() < opts.root_tol {
        return Ok((step.t_new, step.y_new.clone(), g_new));
    }

    match brentq(event_at_t, a, b, &scalar_opts) {
        Ok(result) => {
            // Get the state at the refined time
            let y_event = dense_eval(client, step, result.root).map_err(to_integrate_err)?;
            Ok((result.root, y_event, result.function_value))
        }
        Err(_) => {
            // Fallback: use linear interpolation for time estimate
            let theta = g_old / (g_old - g_new);
            let t_event = step.t_old + theta * step.h();
            let y_event = dense_eval(client, step, t_event).map_err(to_integrate_err)?;
            let g_event = event_fn
                .evaluate(client, t_event, &y_event)
                .map_err(to_integrate_err)?;
            Ok((t_event, y_event, g_event))
        }
    }
}

/// Handle terminal event: truncate the step to the event time.
///
/// Returns the adjusted (t_stop, y_stop) pair.
pub fn handle_terminal_event<R, C>(
    client: &C,
    step: &DenseOutputStep<R>,
    event: &EventRecord<R>,
) -> IntegrateResult<(f64, Tensor<R>)>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
    // If the event is at the end of the step, use the step endpoint
    if (event.t - step.t_new).abs() < 1e-14 {
        return Ok((step.t_new, step.y_new.clone()));
    }

    // Otherwise, interpolate to the event time
    let y_stop = dense_eval(client, step, event.t).map_err(to_integrate_err)?;
    Ok((event.t, y_stop))
}

fn to_integrate_err(e: numr::error::Error) -> crate::integrate::error::IntegrateError {
    crate::integrate::error::IntegrateError::InvalidInput {
        context: format!("Event evaluation error: {}", e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integrate::EventFn;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    fn setup() -> (CpuDevice, CpuClient) {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        (device, client)
    }

    #[test]
    fn test_evaluate_events() {
        let (device, client) = setup();

        let event1 = EventFn::new(|_c: &CpuClient, _t: f64, y: &Tensor<CpuRuntime>| {
            let y_data: Vec<f64> = y.to_vec();
            Ok(y_data[0]) // g(t, y) = y[0]
        });

        let event2 = EventFn::new(|_c: &CpuClient, _t: f64, y: &Tensor<CpuRuntime>| {
            let y_data: Vec<f64> = y.to_vec();
            Ok(y_data[0] - 0.5) // g(t, y) = y[0] - 0.5
        });

        let events: Vec<&dyn EventFunction<CpuRuntime, CpuClient>> = vec![&event1, &event2];
        let y = Tensor::<CpuRuntime>::from_slice(&[1.0], &[1], &device).unwrap();

        let values = evaluate_events(&client, &events, 0.0, &y).unwrap();
        assert_eq!(values.len(), 2);
        assert!((values[0] - 1.0).abs() < 1e-10);
        assert!((values[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_check_events_detects_crossing() {
        let (device, client) = setup();

        // Event: y[0] = 0, which should happen at t = 0.5 for y(t) = 1 - 2t
        let event = EventFn::new(|_c: &CpuClient, _t: f64, y: &Tensor<CpuRuntime>| {
            let y_data: Vec<f64> = y.to_vec();
            Ok(y_data[0])
        });

        // Create step for y(t) = 1 - 2t from t=0 to t=1
        // y(0) = 1, y(1) = -1, f(t) = -2
        let y_old = Tensor::<CpuRuntime>::from_slice(&[1.0], &[1], &device).unwrap();
        let y_new = Tensor::<CpuRuntime>::from_slice(&[-1.0], &[1], &device).unwrap();
        let f_old = Tensor::<CpuRuntime>::from_slice(&[-2.0], &[1], &device).unwrap();
        let f_new = Tensor::<CpuRuntime>::from_slice(&[-2.0], &[1], &device).unwrap();

        let step = DenseOutputStep::new(0.0, 1.0, y_old, y_new, f_old, f_new);

        let events: Vec<&dyn EventFunction<CpuRuntime, CpuClient>> = vec![&event];
        let specs = vec![EventSpec::default()];
        let g_old = vec![1.0]; // g(0) = y(0) = 1

        let result = check_events(
            &client,
            &events,
            &specs,
            &step,
            &g_old,
            &EventOptions::default(),
        )
        .unwrap();

        assert_eq!(result.events.len(), 1);
        let detected = &result.events[0];
        assert!(
            (detected.t - 0.5).abs() < 1e-6,
            "Event at t = {}",
            detected.t
        );
        assert!(detected.event_value.abs() < 1e-6);
    }

    #[test]
    fn test_check_events_direction_filter() {
        let (device, client) = setup();

        // Same setup as above, but with direction filter
        let event = EventFn::new(|_c: &CpuClient, _t: f64, y: &Tensor<CpuRuntime>| {
            let y_data: Vec<f64> = y.to_vec();
            Ok(y_data[0])
        });

        let y_old = Tensor::<CpuRuntime>::from_slice(&[1.0], &[1], &device).unwrap();
        let y_new = Tensor::<CpuRuntime>::from_slice(&[-1.0], &[1], &device).unwrap();
        let f_old = Tensor::<CpuRuntime>::from_slice(&[-2.0], &[1], &device).unwrap();
        let f_new = Tensor::<CpuRuntime>::from_slice(&[-2.0], &[1], &device).unwrap();

        let step = DenseOutputStep::new(0.0, 1.0, y_old, y_new, f_old, f_new);
        let events: Vec<&dyn EventFunction<CpuRuntime, CpuClient>> = vec![&event];
        let g_old = vec![1.0];

        // Increasing direction - should NOT detect (we're going from + to -)
        let specs = vec![EventSpec::default().direction(EventDirection::Increasing)];
        let result = check_events(
            &client,
            &events,
            &specs,
            &step,
            &g_old,
            &EventOptions::default(),
        )
        .unwrap();
        assert_eq!(result.events.len(), 0);

        // Decreasing direction - should detect (we're going from + to -)
        let specs = vec![EventSpec::default().direction(EventDirection::Decreasing)];
        let result = check_events(
            &client,
            &events,
            &specs,
            &step,
            &g_old,
            &EventOptions::default(),
        )
        .unwrap();
        assert_eq!(result.events.len(), 1);
    }

    #[test]
    fn test_check_events_terminal() {
        let (device, client) = setup();

        let event = EventFn::new(|_c: &CpuClient, _t: f64, y: &Tensor<CpuRuntime>| {
            let y_data: Vec<f64> = y.to_vec();
            Ok(y_data[0])
        });

        let y_old = Tensor::<CpuRuntime>::from_slice(&[1.0], &[1], &device).unwrap();
        let y_new = Tensor::<CpuRuntime>::from_slice(&[-1.0], &[1], &device).unwrap();
        let f_old = Tensor::<CpuRuntime>::from_slice(&[-2.0], &[1], &device).unwrap();
        let f_new = Tensor::<CpuRuntime>::from_slice(&[-2.0], &[1], &device).unwrap();

        let step = DenseOutputStep::new(0.0, 1.0, y_old, y_new, f_old, f_new);
        let events: Vec<&dyn EventFunction<CpuRuntime, CpuClient>> = vec![&event];
        let specs = vec![EventSpec::terminal()];
        let g_old = vec![1.0];

        let result = check_events(
            &client,
            &events,
            &specs,
            &step,
            &g_old,
            &EventOptions::default(),
        )
        .unwrap();

        assert!(result.has_terminal);
        assert_eq!(result.terminal_index, Some(0));
        assert!(result.terminal_time.is_some());
    }
}
