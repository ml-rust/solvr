//! Original-bit interpolation reference inputs.
use super::types::CaseF32;

pub const F32_VALUES: &[CaseF32] = &[
    CaseF32 {
        id: "unit_knots_shuffled_f32",
        x: &[
            f32::from_bits(0x00000000),
            f32::from_bits(0x3f800000),
            f32::from_bits(0x40400000),
        ],
        y: &[
            f32::from_bits(0x3f800000),
            f32::from_bits(0x40000000),
            f32::from_bits(0x40800000),
        ],
        query: &[
            f32::from_bits(0x40400000),
            f32::from_bits(0x3f000000),
            f32::from_bits(0x00000000),
            f32::from_bits(0x3f800000),
            f32::from_bits(0x40000000),
            f32::from_bits(0x3f800000),
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
    CaseF32 {
        id: "signed_cancellation_f32",
        x: &[f32::from_bits(0x00000000), f32::from_bits(0x3f800000)],
        y: &[f32::from_bits(0xbf800000), f32::from_bits(0x3f800000)],
        query: &[
            f32::from_bits(0x3e800000),
            f32::from_bits(0x3f000000),
            f32::from_bits(0x3f400000),
            f32::from_bits(0x00000000),
            f32::from_bits(0x3f800000),
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
    CaseF32 {
        id: "positive_cancellation_f32",
        x: &[f32::from_bits(0x3efffffe), f32::from_bits(0x40000000)],
        y: &[f32::from_bits(0x3f800000), f32::from_bits(0x00000000)],
        query: &[f32::from_bits(0x3fffffff)],
        legacy_error: None,
        metadata: &[(0, 1, false, false, false)],
    },
    CaseF32 {
        id: "nonzero_ordinate_witness_f32",
        x: &[f32::from_bits(0x00000000), f32::from_bits(0x3f800000)],
        y: &[f32::from_bits(0x80000001), f32::from_bits(0x3f800000)],
        query: &[
            f32::from_bits(0x3e800000),
            f32::from_bits(0x3f400000),
            f32::from_bits(0x00000000),
            f32::from_bits(0x3f800000),
        ],
        legacy_error: None,
        metadata: &[
            (0, 1, false, false, false),
            (0, 1, false, false, false),
            (0, 1, true, false, false),
            (0, 1, false, true, false),
        ],
    },
    CaseF32 {
        id: "large_signed_ordinates_f32",
        x: &[f32::from_bits(0x00000000), f32::from_bits(0x3f800000)],
        y: &[f32::from_bits(0xfe7fffff), f32::from_bits(0x7e7fffff)],
        query: &[
            f32::from_bits(0x3e800000),
            f32::from_bits(0x3f000000),
            f32::from_bits(0x3f400000),
        ],
        legacy_error: None,
        metadata: &[
            (0, 1, false, false, false),
            (0, 1, false, false, false),
            (0, 1, false, false, false),
        ],
    },
    CaseF32 {
        id: "constant_unused_overflow_f32",
        x: &[
            f32::from_bits(0x00000000),
            f32::from_bits(0x3f800000),
            f32::from_bits(0x40000000),
        ],
        y: &[
            f32::from_bits(0xff7fffff),
            f32::from_bits(0x7f7fffff),
            f32::from_bits(0x7f7fffff),
        ],
        query: &[
            f32::from_bits(0x3f800000),
            f32::from_bits(0x3fc00000),
            f32::from_bits(0x40000000),
        ],
        legacy_error: None,
        metadata: &[
            (0, 1, false, true, false),
            (1, 2, false, false, true),
            (1, 2, false, true, true),
        ],
    },
    CaseF32 {
        id: "knots_unused_overflow_f32",
        x: &[f32::from_bits(0x00000000), f32::from_bits(0x3f800000)],
        y: &[f32::from_bits(0xff7fffff), f32::from_bits(0x7f7fffff)],
        query: &[
            f32::from_bits(0x3f800000),
            f32::from_bits(0x00000000),
            f32::from_bits(0x3f800000),
        ],
        legacy_error: None,
        metadata: &[
            (0, 1, false, true, false),
            (0, 1, true, false, false),
            (0, 1, false, true, false),
        ],
    },
    CaseF32 {
        id: "subnormal_increment_f32",
        x: &[f32::from_bits(0x00000000), f32::from_bits(0x3f800000)],
        y: &[f32::from_bits(0x00000000), f32::from_bits(0x00000002)],
        query: &[
            f32::from_bits(0x3f400000),
            f32::from_bits(0x00000000),
            f32::from_bits(0x3f800000),
        ],
        legacy_error: None,
        metadata: &[
            (0, 1, false, false, false),
            (0, 1, true, false, false),
            (0, 1, false, true, false),
        ],
    },
    CaseF32 {
        id: "tiny_grid_width_f32",
        x: &[f32::from_bits(0x00000000), f32::from_bits(0x00800000)],
        y: &[f32::from_bits(0xbf800000), f32::from_bits(0x3f800000)],
        query: &[
            f32::from_bits(0x00200000),
            f32::from_bits(0x00400000),
            f32::from_bits(0x00600000),
        ],
        legacy_error: None,
        metadata: &[
            (0, 1, false, false, false),
            (0, 1, false, false, false),
            (0, 1, false, false, false),
        ],
    },
    CaseF32 {
        id: "huge_grid_width_f32",
        x: &[f32::from_bits(0xfe7fffff), f32::from_bits(0x7e7fffff)],
        y: &[f32::from_bits(0xbf800000), f32::from_bits(0x3f800000)],
        query: &[
            f32::from_bits(0x00000000),
            f32::from_bits(0xfdffffff),
            f32::from_bits(0x7dffffff),
        ],
        legacy_error: None,
        metadata: &[
            (0, 1, false, false, false),
            (0, 1, false, false, false),
            (0, 1, false, false, false),
        ],
    },
];
