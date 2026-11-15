use super::scalar_tolerance::lower;
use crate::DType;
use crate::optimize::{OptimizeError, OptimizeResult};

// Frozen exact-rational floor references, independent of the conversion implementation.
const CASES: &[(u64, u32)] = &[
    (0x0000000000000000, 0x00000000), // positive_zero
    (0x8000000000000000, 0x80000000), // negative_zero
    (0x3fb999999999999a, 0x3dcccccc), // decimal_tenth
    (0x3fb99999a0000000, 0x3dcccccd), // represented_f32_tenth
    (0x3fb9999980000000, 0x3dcccccc), // f32_tenth_predecessor
    (0x36a0000000000000, 0x00000001), // smallest_f32_subnormal
    (0x3690000000000000, 0x00000000), // half_smallest_f32_subnormal
    (0x0000000000000001, 0x00000000), // smallest_f64_subnormal
    (0x3810000000000000, 0x00800000), // f32_minimum_normal
    (0x47efffffe0000000, 0x7f7fffff), // f32_maximum
    (0x47efffffe0000001, 0x7f7fffff), // above_f32_maximum
    (0x7fefffffffffffff, 0x7f7fffff), // f64_maximum
];

#[test]
fn original_scalar_tolerances_round_down() -> OptimizeResult<()> {
    for &(original, expected) in CASES {
        let value = f64::from_bits(original);
        assert_eq!(lower(DType::F64, value)?.to_bits(), original);
        assert_eq!(
            lower(DType::F32, value)?.to_bits(),
            f64::from(f32::from_bits(expected)).to_bits()
        );
        assert!(lower(DType::F32, value)? <= value);
    }
    Ok(())
}

#[test]
fn malformed_scalar_tolerances_return_typed_errors() {
    for value in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(matches!(
            lower(DType::F32, value),
            Err(OptimizeError::InvalidParameter { .. })
        ));
    }
    assert!(matches!(
        lower(DType::I64, 1.0),
        Err(OptimizeError::InvalidParameter { .. })
    ));
}
