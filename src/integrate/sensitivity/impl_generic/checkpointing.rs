//! Checkpoint management for adjoint sensitivity analysis.
//!
//! Manages memory-efficient storage of ODE solution checkpoints
//! during forward integration for use during backward adjoint pass.

use numr::runtime::Runtime;
use numr::tensor::Tensor;

use crate::integrate::sensitivity::traits::{Checkpoint, CheckpointStrategy};

/// Manages checkpoints for adjoint sensitivity computation.
///
/// During forward integration, checkpoints are stored at strategic times.
/// During backward adjoint integration, the forward solution is reconstructed
/// by reintegrating between checkpoints as needed.
#[derive(Debug, Clone)]
pub struct CheckpointManager<R: Runtime> {
    /// Stored checkpoints in forward time order.
    checkpoints: Vec<Checkpoint<R>>,
    /// Target number of checkpoints.
    #[allow(dead_code)]
    n_checkpoints: usize,
    /// Strategy for checkpoint placement.
    #[allow(dead_code)]
    strategy: CheckpointStrategy,
    /// Time span [t0, T].
    #[allow(dead_code)]
    t_span: [f64; 2],
    /// Planned checkpoint times (computed at initialization).
    checkpoint_times: Vec<f64>,
}

impl<R: Runtime> CheckpointManager<R> {
    /// Create a new checkpoint manager.
    ///
    /// # Arguments
    ///
    /// * `n_checkpoints` - Number of checkpoints to store (excluding t0 and T)
    /// * `strategy` - Strategy for placing checkpoints
    /// * `t_span` - Integration interval [t0, T]
    pub fn new(n_checkpoints: usize, strategy: CheckpointStrategy, t_span: [f64; 2]) -> Self {
        let checkpoint_times = Self::compute_checkpoint_times(n_checkpoints, strategy, t_span);

        Self {
            checkpoints: Vec::with_capacity(n_checkpoints + 2),
            n_checkpoints,
            strategy,
            t_span,
            checkpoint_times,
        }
    }

    /// Compute the times at which to place checkpoints.
    fn compute_checkpoint_times(
        n_checkpoints: usize,
        strategy: CheckpointStrategy,
        t_span: [f64; 2],
    ) -> Vec<f64> {
        let [t0, tf] = t_span;
        let dt = tf - t0;

        // Always include t0 and tf
        let n_interior = n_checkpoints.saturating_sub(2);

        match strategy {
            CheckpointStrategy::Uniform => {
                // Uniformly spaced checkpoints
                let n_total = n_interior + 2;
                (0..n_total)
                    .map(|i| t0 + dt * (i as f64) / ((n_total - 1) as f64))
                    .collect()
            }
            CheckpointStrategy::Logarithmic => {
                // Logarithmically spaced - more checkpoints near t0
                // Useful when solution varies rapidly at the beginning
                let mut times = vec![t0];
                if n_interior > 0 {
                    for i in 1..=n_interior {
                        let alpha = (i as f64) / ((n_interior + 1) as f64);
                        // Logarithmic spacing: t = t0 + dt * (e^(alpha*k) - 1) / (e^k - 1)
                        // where k controls concentration (higher = more at start)
                        let k = 3.0;
                        let t = t0 + dt * ((alpha * k).exp() - 1.0) / (k.exp() - 1.0);
                        times.push(t);
                    }
                }
                times.push(tf);
                times
            }
            CheckpointStrategy::Adaptive => {
                // For now, fall back to uniform
                // A true adaptive strategy would place checkpoints based on
                // solution variation during the forward pass
                let n_total = n_interior + 2;
                (0..n_total)
                    .map(|i| t0 + dt * (i as f64) / ((n_total - 1) as f64))
                    .collect()
            }
        }
    }

    /// Get the planned checkpoint times.
    pub fn checkpoint_times(&self) -> &[f64] {
        &self.checkpoint_times
    }

    /// Add a checkpoint at the given time and state.
    pub fn add_checkpoint(&mut self, t: f64, y: Tensor<R>) {
        self.checkpoints.push(Checkpoint::new(t, y));
    }

    /// Get all stored checkpoints.
    pub fn checkpoints(&self) -> &[Checkpoint<R>] {
        &self.checkpoints
    }

    /// Get the number of stored checkpoints.
    pub fn len(&self) -> usize {
        self.checkpoints.len()
    }

    /// Check if no checkpoints are stored.
    pub fn is_empty(&self) -> bool {
        self.checkpoints.is_empty()
    }

    /// Find the checkpoint interval containing time t.
    ///
    /// Returns (checkpoint_before, checkpoint_after) where
    /// checkpoint_before.t <= t <= checkpoint_after.t
    ///
    /// For backward adjoint integration, we need the checkpoint
    /// just before the current time to start reintegrating from.
    pub fn find_interval(&self, t: f64) -> Option<(usize, usize)> {
        if self.checkpoints.is_empty() {
            return None;
        }

        // Find the largest checkpoint time <= t
        let mut before_idx = 0;
        for (i, ck) in self.checkpoints.iter().enumerate() {
            if ck.t <= t {
                before_idx = i;
            } else {
                break;
            }
        }

        let after_idx = (before_idx + 1).min(self.checkpoints.len() - 1);

        Some((before_idx, after_idx))
    }

    /// Get checkpoint at index.
    pub fn get(&self, index: usize) -> Option<&Checkpoint<R>> {
        self.checkpoints.get(index)
    }

    /// Iterate checkpoints in reverse order (for backward pass).
    pub fn iter_reverse(&self) -> impl Iterator<Item = &Checkpoint<R>> {
        self.checkpoints.iter().rev()
    }

    /// Check if a time is close to a planned checkpoint time.
    pub fn should_checkpoint(&self, t: f64, tol: f64) -> bool {
        self.checkpoint_times.iter().any(|&tc| (t - tc).abs() < tol)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    fn make_checkpoint(t: f64) -> Checkpoint<CpuRuntime> {
        let device = CpuDevice::new();
        let y = Tensor::<CpuRuntime>::from_slice(&[t], &[1], &device);
        Checkpoint::new(t, y)
    }

    #[test]
    fn test_uniform_checkpoint_times() {
        let manager =
            CheckpointManager::<CpuRuntime>::new(5, CheckpointStrategy::Uniform, [0.0, 1.0]);

        let times = manager.checkpoint_times();
        assert_eq!(times.len(), 5);
        assert!((times[0] - 0.0).abs() < 1e-10);
        assert!((times[1] - 0.25).abs() < 1e-10);
        assert!((times[2] - 0.5).abs() < 1e-10);
        assert!((times[3] - 0.75).abs() < 1e-10);
        assert!((times[4] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_logarithmic_checkpoint_times() {
        let manager =
            CheckpointManager::<CpuRuntime>::new(5, CheckpointStrategy::Logarithmic, [0.0, 1.0]);

        let times = manager.checkpoint_times();
        assert_eq!(times.len(), 5);
        assert!((times[0] - 0.0).abs() < 1e-10);
        assert!((times[4] - 1.0).abs() < 1e-10);

        // Logarithmic: points should be denser near t=0
        let dt1 = times[1] - times[0];
        let dt2 = times[2] - times[1];
        let dt3 = times[3] - times[2];
        assert!(dt1 < dt2, "dt1={} should be < dt2={}", dt1, dt2);
        assert!(dt2 < dt3, "dt2={} should be < dt3={}", dt2, dt3);
    }

    #[test]
    fn test_add_and_find_checkpoint() {
        let mut manager =
            CheckpointManager::<CpuRuntime>::new(5, CheckpointStrategy::Uniform, [0.0, 1.0]);

        manager.add_checkpoint(0.0, make_checkpoint(0.0).y);
        manager.add_checkpoint(0.5, make_checkpoint(0.5).y);
        manager.add_checkpoint(1.0, make_checkpoint(1.0).y);

        assert_eq!(manager.len(), 3);

        // Find interval containing t=0.3
        let (before, after) = manager.find_interval(0.3).unwrap();
        assert_eq!(before, 0);
        assert_eq!(after, 1);

        // Find interval containing t=0.7
        let (before, after) = manager.find_interval(0.7).unwrap();
        assert_eq!(before, 1);
        assert_eq!(after, 2);
    }

    #[test]
    fn test_should_checkpoint() {
        let manager =
            CheckpointManager::<CpuRuntime>::new(5, CheckpointStrategy::Uniform, [0.0, 1.0]);

        let tol = 1e-8;
        assert!(manager.should_checkpoint(0.0, tol));
        assert!(manager.should_checkpoint(0.25, tol));
        assert!(manager.should_checkpoint(0.5, tol));
        assert!(!manager.should_checkpoint(0.3, tol));
    }
}
