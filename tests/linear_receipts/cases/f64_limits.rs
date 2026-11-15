//! Original-bit interpolation reference inputs.
use super::types::CaseF64;

pub const F64_LIMITS: &[CaseF64] = &[
    CaseF64 {
        id: "empty_queries_f64",
        x: &[
            f64::from_bits(0x0000000000000000),
            f64::from_bits(0x3ff0000000000000),
        ],
        y: &[
            f64::from_bits(0xffefffffffffffff),
            f64::from_bits(0x7fefffffffffffff),
        ],
        query: &[],
        legacy_error: None,
        metadata: &[],
    },
    CaseF64 {
        id: "outside_closed_support_f64",
        x: &[
            f64::from_bits(0x0000000000000000),
            f64::from_bits(0x3ff0000000000000),
        ],
        y: &[
            f64::from_bits(0x3ff0000000000000),
            f64::from_bits(0x4008000000000000),
        ],
        query: &[
            f64::from_bits(0xbff0000000000000),
            f64::from_bits(0x4000000000000000),
        ],
        legacy_error: None,
        metadata: &[(0, 1, false, false, false), (0, 1, false, false, false)],
    },
    CaseF64 {
        id: "grid_width_overflow_f64",
        x: &[
            f64::from_bits(0xffefffffffffffff),
            f64::from_bits(0x7fefffffffffffff),
        ],
        y: &[
            f64::from_bits(0x0000000000000000),
            f64::from_bits(0x3ff0000000000000),
        ],
        query: &[
            f64::from_bits(0xffefffffffffffff),
            f64::from_bits(0x7fefffffffffffff),
        ],
        legacy_error: Some("adjacent grid width"),
        metadata: &[(0, 1, true, false, false), (0, 1, false, true, false)],
    },
    CaseF64 {
        id: "active_ordinate_overflow_f64",
        x: &[
            f64::from_bits(0x0000000000000000),
            f64::from_bits(0x3ff0000000000000),
        ],
        y: &[
            f64::from_bits(0xffefffffffffffff),
            f64::from_bits(0x7fefffffffffffff),
        ],
        query: &[f64::from_bits(0x3fe0000000000000)],
        legacy_error: Some("active value difference"),
        metadata: &[(0, 1, false, false, false)],
    },
    CaseF64 {
        id: "active_offset_overflow_f64",
        x: &[
            f64::from_bits(0xffefffffffffffff),
            f64::from_bits(0xffdfffffffffffff),
        ],
        y: &[
            f64::from_bits(0x0000000000000000),
            f64::from_bits(0x3ff0000000000000),
        ],
        query: &[f64::from_bits(0x7fefffffffffffff)],
        legacy_error: Some("active query offset"),
        metadata: &[(0, 1, false, false, false)],
    },
    CaseF64 {
        id: "coordinate_overflow_f64",
        x: &[
            f64::from_bits(0x0000000000000000),
            f64::from_bits(0x0000000000000001),
        ],
        y: &[
            f64::from_bits(0x0000000000000000),
            f64::from_bits(0x3ff0000000000000),
        ],
        query: &[f64::from_bits(0x7fefffffffffffff)],
        legacy_error: Some("linear coordinate"),
        metadata: &[(0, 1, false, false, false)],
    },
    CaseF64 {
        id: "coordinate_underflow_f64",
        x: &[
            f64::from_bits(0x0000000000000000),
            f64::from_bits(0x7fefffffffffffff),
        ],
        y: &[
            f64::from_bits(0x0000000000000000),
            f64::from_bits(0x3ff0000000000000),
        ],
        query: &[f64::from_bits(0x0000000000000001)],
        legacy_error: Some("linear coordinate"),
        metadata: &[(0, 1, false, false, false)],
    },
    CaseF64 {
        id: "increment_underflow_f64",
        x: &[
            f64::from_bits(0x0000000000000000),
            f64::from_bits(0x3ff0000000000000),
        ],
        y: &[
            f64::from_bits(0x0000000000000000),
            f64::from_bits(0x0000000000000001),
        ],
        query: &[f64::from_bits(0x3fe0000000000000)],
        legacy_error: Some("linear value increment"),
        metadata: &[(0, 1, false, false, false)],
    },
    CaseF64 {
        id: "increment_overflow_f64",
        x: &[
            f64::from_bits(0x0000000000000000),
            f64::from_bits(0x3ff0000000000000),
        ],
        y: &[
            f64::from_bits(0x0000000000000000),
            f64::from_bits(0x7fefffffffffffff),
        ],
        query: &[f64::from_bits(0x4000000000000000)],
        legacy_error: Some("linear value increment"),
        metadata: &[(0, 1, false, false, false)],
    },
    CaseF64 {
        id: "output_overflow_f64",
        x: &[
            f64::from_bits(0x0000000000000000),
            f64::from_bits(0x3ff0000000000000),
        ],
        y: &[
            f64::from_bits(0x7fdfffffffffffff),
            f64::from_bits(0x7fefffffffffffff),
        ],
        query: &[f64::from_bits(0x4000000000000000)],
        legacy_error: Some("linear interpolated value"),
        metadata: &[(0, 1, false, false, false)],
    },
];
