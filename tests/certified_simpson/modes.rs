//! Isolated MXCSR checks run on x86_64. Other architectures omit this mode test.

use numr::prelude::*;
use solvr::integrate::{CertifiedSimpsonAlgorithms, CertifiedSimpsonOptions, IntegrateError};

struct Restore(u32);
impl Drop for Restore {
    fn drop(&mut self) {
        // SAFETY: The saved register retains every reserved MXCSR bit.
        unsafe {
            std::arch::asm!("ldmxcsr [{}]", in(reg) &self.0, options(nostack, preserves_flags));
        }
    }
}
#[test]
fn altered_cpu_arithmetic_rejects_before_callbacks_even_for_empty_batches() {
    std::thread::spawn(|| {
        let mut original = 0u32;
        // SAFETY: MXCSR writes four bytes into a valid local variable.
        unsafe { std::arch::asm!("stmxcsr [{}]", in(reg) &mut original, options(nostack, preserves_flags)); }
        let _restore = Restore(original);
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        let empty = Tensor::<CpuRuntime>::empty(&[0], DType::F64, &device).expect("empty tensor");
        let nonempty = Tensor::<CpuRuntime>::from_slice(&[0.0f64], &[1], &device).expect("endpoint tensor");
        for mode in [(original & !(3 << 13)) | (1 << 13), original | (1 << 15) | (1 << 6)] {
            // SAFETY: Mode changes retain all reserved bits and restore through the guard.
            unsafe { std::arch::asm!("ldmxcsr [{}]", in(reg) &mode, options(nostack, preserves_flags)); }
            for endpoints in [&empty, &nonempty] {
                let error = client.certified_simpson(endpoints, endpoints, CertifiedSimpsonOptions { panels: 1, max_points: 1 },
                    |_| panic!("unsupported mode reaches value callback"),
                    |_| panic!("unsupported mode reaches derivative callback")).expect_err("unsupported mode accepted");
                assert!(matches!(error, IntegrateError::UnsupportedArithmetic { dtype: DType::F64, failed_probes } if !failed_probes.is_empty() && !failed_probes.contains(&"provider_contract")));
            }
        }
    }).join().expect("isolated mode test panicked");
}
