//! Original-bit interpolation reference inputs.
use super::types::CaseF64;

pub const F64_VALUES: &[CaseF64] = &[
    CaseF64 {
        id: "unit_knots_shuffled_f64",
        x: &[
            f64::from_bits(0x0000000000000000),
            f64::from_bits(0x3ff0000000000000),
            f64::from_bits(0x4008000000000000),
        ],
        y: &[
            f64::from_bits(0x3ff0000000000000),
            f64::from_bits(0x4000000000000000),
            f64::from_bits(0x4010000000000000),
        ],
        query: &[
            f64::from_bits(0x4008000000000000),
            f64::from_bits(0x3fe0000000000000),
            f64::from_bits(0x0000000000000000),
            f64::from_bits(0x3ff0000000000000),
            f64::from_bits(0x4000000000000000),
            f64::from_bits(0x3ff0000000000000),
        ],
        legacy_error: None,
        metadata: &[
            (1, 2, false, true, false),
            (0, 1, false, false, false),
            (0, 1, true, false, false),
            (0, 1, false, true, false),
            (1, 2, false, false, false),
            (0, 1, false, true, false),
        ],
    },
    CaseF64 {
        id: "signed_cancellation_f64",
        x: &[
            f64::from_bits(0x0000000000000000),
            f64::from_bits(0x3ff0000000000000),
        ],
        y: &[
            f64::from_bits(0xbff0000000000000),
            f64::from_bits(0x3ff0000000000000),
        ],
        query: &[
            f64::from_bits(0x3fd0000000000000),
            f64::from_bits(0x3fe0000000000000),
            f64::from_bits(0x3fe8000000000000),
            f64::from_bits(0x0000000000000000),
            f64::from_bits(0x3ff0000000000000),
        ],
        legacy_error: None,
        metadata: &[
            (0, 1, false, false, false),
            (0, 1, false, false, false),
            (0, 1, false, false, false),
            (0, 1, true, false, false),
            (0, 1, false, true, false),
        ],
    },
    CaseF64 {
        id: "positive_cancellation_f64",
        x: &[
            f64::from_bits(0x3fdffffffffffffe),
            f64::from_bits(0x4000000000000000),
        ],
        y: &[
            f64::from_bits(0x3ff0000000000000),
            f64::from_bits(0x0000000000000000),
        ],
        query: &[f64::from_bits(0x3fffffffffffffff)],
        legacy_error: None,
        metadata: &[(0, 1, false, false, false)],
    },
    CaseF64 {
        id: "nonzero_ordinate_witness_f64",
        x: &[
            f64::from_bits(0x0000000000000000),
            f64::from_bits(0x3ff0000000000000),
        ],
        y: &[
            f64::from_bits(0x8000000000000001),
            f64::from_bits(0x3ff0000000000000),
        ],
        query: &[
            f64::from_bits(0x3fd0000000000000),
            f64::from_bits(0x3fe8000000000000),
            f64::from_bits(0x0000000000000000),
            f64::from_bits(0x3ff0000000000000),
        ],
        legacy_error: None,
        metadata: &[
            (0, 1, false, false, false),
            (0, 1, false, false, false),
            (0, 1, true, false, false),
            (0, 1, false, true, false),
        ],
    },
    CaseF64 {
        id: "large_signed_ordinates_f64",
        x: &[
            f64::from_bits(0x0000000000000000),
            f64::from_bits(0x3ff0000000000000),
        ],
        y: &[
            f64::from_bits(0xffcfffffffffffff),
            f64::from_bits(0x7fcfffffffffffff),
        ],
        query: &[
            f64::from_bits(0x3fd0000000000000),
            f64::from_bits(0x3fe0000000000000),
            f64::from_bits(0x3fe8000000000000),
        ],
        legacy_error: None,
        metadata: &[
            (0, 1, false, false, false),
            (0, 1, false, false, false),
            (0, 1, false, false, false),
        ],
    },
    CaseF64 {
        id: "constant_unused_overflow_f64",
        x: &[
            f64::from_bits(0x0000000000000000),
            f64::from_bits(0x3ff0000000000000),
            f64::from_bits(0x4000000000000000),
        ],
        y: &[
            f64::from_bits(0xffefffffffffffff),
            f64::from_bits(0x7fefffffffffffff),
            f64::from_bits(0x7fefffffffffffff),
        ],
        query: &[
            f64::from_bits(0x3ff0000000000000),
            f64::from_bits(0x3ff8000000000000),
            f64::from_bits(0x4000000000000000),
        ],
        legacy_error: None,
        metadata: &[
            (0, 1, false, true, false),
            (1, 2, false, false, true),
            (1, 2, false, true, true),
        ],
    },
    CaseF64 {
        id: "knots_unused_overflow_f64",
        x: &[
            f64::from_bits(0x0000000000000000),
            f64::from_bits(0x3ff0000000000000),
        ],
        y: &[
            f64::from_bits(0xffefffffffffffff),
            f64::from_bits(0x7fefffffffffffff),
        ],
        query: &[
            f64::from_bits(0x3ff0000000000000),
            f64::from_bits(0x0000000000000000),
            f64::from_bits(0x3ff0000000000000),
        ],
        legacy_error: None,
        metadata: &[
            (0, 1, false, true, false),
            (0, 1, true, false, false),
            (0, 1, false, true, false),
        ],
    },
    CaseF64 {
        id: "subnormal_increment_f64",
        x: &[
            f64::from_bits(0x0000000000000000),
            f64::from_bits(0x3ff0000000000000),
        ],
        y: &[
            f64::from_bits(0x0000000000000000),
            f64::from_bits(0x0000000000000002),
        ],
        query: &[
            f64::from_bits(0x3fe8000000000000),
            f64::from_bits(0x0000000000000000),
            f64::from_bits(0x3ff0000000000000),
        ],
        legacy_error: None,
        metadata: &[
            (0, 1, false, false, false),
            (0, 1, true, false, false),
            (0, 1, false, true, false),
        ],
    },
    CaseF64 {
        id: "tiny_grid_width_f64",
        x: &[
            f64::from_bits(0x0000000000000000),
            f64::from_bits(0x0010000000000000),
        ],
        y: &[
            f64::from_bits(0xbff0000000000000),
            f64::from_bits(0x3ff0000000000000),
        ],
        query: &[
            f64::from_bits(0x0004000000000000),
            f64::from_bits(0x0008000000000000),
            f64::from_bits(0x000c000000000000),
        ],
        legacy_error: None,
        metadata: &[
            (0, 1, false, false, false),
            (0, 1, false, false, false),
            (0, 1, false, false, false),
        ],
    },
    CaseF64 {
        id: "huge_grid_width_f64",
        x: &[
            f64::from_bits(0xffcfffffffffffff),
            f64::from_bits(0x7fcfffffffffffff),
        ],
        y: &[
            f64::from_bits(0xbff0000000000000),
            f64::from_bits(0x3ff0000000000000),
        ],
        query: &[
            f64::from_bits(0x0000000000000000),
            f64::from_bits(0xffbfffffffffffff),
            f64::from_bits(0x7fbfffffffffffff),
        ],
        legacy_error: None,
        metadata: &[
            (0, 1, false, false, false),
            (0, 1, false, false, false),
            (0, 1, false, false, false),
        ],
    },
];
