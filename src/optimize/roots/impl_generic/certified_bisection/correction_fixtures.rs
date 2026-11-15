// Frozen original-bit operands and exact-rational directed reference endpoints.
pub(super) struct CorrectionCase {
    pub id: &'static str,
    pub active: bool,
    pub operands: [u64; 6],
    pub enclosure: [u64; 2],
    pub exact: [Option<u64>; 2],
}

pub(super) const F32: &[CorrectionCase] = &[
    CorrectionCase {
        id: "finite_regular",
        active: true,
        operands: [
            0xc0000000, 0x40400000, 0x3f000000, 0xbe800000, 0x3f000000, 0x40000000,
        ],
        enclosure: [0x3e800000, 0x3f200000],
        exact: [None, None],
    },
    CorrectionCase {
        id: "quotient_overflow",
        active: true,
        operands: [
            0xbf800000, 0x3f800000, 0x00000000, 0xff7fffff, 0x7f7fffff, 0x00000001,
        ],
        enclosure: [0xbf800000, 0x3f800000],
        exact: [None, None],
    },
    CorrectionCase {
        id: "quotient_underflow",
        active: true,
        operands: [
            0xbf800000, 0x3f800000, 0x00000000, 0x80000001, 0x00000001, 0x40000000,
        ],
        enclosure: [0x80000001, 0x00000001],
        exact: [None, None],
    },
    CorrectionCase {
        id: "upper_endpoint_overflow",
        active: true,
        operands: [
            0x00000000, 0x7f7fffff, 0x7f7fffff, 0xfeffffff, 0x00000000, 0x3f800000,
        ],
        enclosure: [0x7f7fffff, 0x7f7fffff],
        exact: [Some(0x7f7fffff), None],
    },
    CorrectionCase {
        id: "lower_endpoint_overflow",
        active: true,
        operands: [
            0xff7fffff, 0x00000000, 0xff7fffff, 0x80000000, 0x7effffff, 0x3f800000,
        ],
        enclosure: [0xff7fffff, 0xff7fffff],
        exact: [None, Some(0xff7fffff)],
    },
    CorrectionCase {
        id: "finite_quotient_next_up_infinity",
        active: true,
        operands: [
            0xbf800000, 0x3f800000, 0x00000000, 0xff7fffff, 0x7f7fffff, 0x3f800000,
        ],
        enclosure: [0xbf800000, 0x3f800000],
        exact: [None, None],
    },
    CorrectionCase {
        id: "signed_zero_corrections",
        active: true,
        operands: [
            0xbf800000, 0x3f800000, 0x3e800000, 0x80000000, 0x00000000, 0x00000001,
        ],
        enclosure: [0x3e800000, 0x3e800000],
        exact: [Some(0x3e800000), Some(0x3e800000)],
    },
    CorrectionCase {
        id: "one_sided_zero",
        active: true,
        operands: [
            0xbf800000, 0x3f800000, 0x00000000, 0xbf000000, 0x80000000, 0x3f800000,
        ],
        enclosure: [0x00000000, 0x3f000000],
        exact: [Some(0x00000000), None],
    },
    CorrectionCase {
        id: "maximum_finite_domain",
        active: true,
        operands: [
            0xff7fffff, 0x7f7fffff, 0x00000000, 0xbf800000, 0x3f800000, 0x3f800000,
        ],
        enclosure: [0xbf800000, 0x3f800000],
        exact: [None, None],
    },
    CorrectionCase {
        id: "inactive_invalid_operands",
        active: false,
        operands: [
            0xbf800000, 0x3f800000, 0x7fc00001, 0x7fc00001, 0x7f800000, 0x00000000,
        ],
        enclosure: [0xbf800000, 0x3f800000],
        exact: [Some(0xbf800000), Some(0x3f800000)],
    },
];

pub(super) const F64: &[CorrectionCase] = &[
    CorrectionCase {
        id: "finite_regular",
        active: true,
        operands: [
            0xc000000000000000,
            0x4008000000000000,
            0x3fe0000000000000,
            0xbfd0000000000000,
            0x3fe0000000000000,
            0x4000000000000000,
        ],
        enclosure: [0x3fd0000000000000, 0x3fe4000000000000],
        exact: [None, None],
    },
    CorrectionCase {
        id: "quotient_overflow",
        active: true,
        operands: [
            0xbff0000000000000,
            0x3ff0000000000000,
            0x0000000000000000,
            0xffefffffffffffff,
            0x7fefffffffffffff,
            0x0000000000000001,
        ],
        enclosure: [0xbff0000000000000, 0x3ff0000000000000],
        exact: [None, None],
    },
    CorrectionCase {
        id: "quotient_underflow",
        active: true,
        operands: [
            0xbff0000000000000,
            0x3ff0000000000000,
            0x0000000000000000,
            0x8000000000000001,
            0x0000000000000001,
            0x4000000000000000,
        ],
        enclosure: [0x8000000000000001, 0x0000000000000001],
        exact: [None, None],
    },
    CorrectionCase {
        id: "upper_endpoint_overflow",
        active: true,
        operands: [
            0x0000000000000000,
            0x7fefffffffffffff,
            0x7fefffffffffffff,
            0xffdfffffffffffff,
            0x0000000000000000,
            0x3ff0000000000000,
        ],
        enclosure: [0x7fefffffffffffff, 0x7fefffffffffffff],
        exact: [Some(0x7fefffffffffffff), None],
    },
    CorrectionCase {
        id: "lower_endpoint_overflow",
        active: true,
        operands: [
            0xffefffffffffffff,
            0x0000000000000000,
            0xffefffffffffffff,
            0x8000000000000000,
            0x7fdfffffffffffff,
            0x3ff0000000000000,
        ],
        enclosure: [0xffefffffffffffff, 0xffefffffffffffff],
        exact: [None, Some(0xffefffffffffffff)],
    },
    CorrectionCase {
        id: "finite_quotient_next_up_infinity",
        active: true,
        operands: [
            0xbff0000000000000,
            0x3ff0000000000000,
            0x0000000000000000,
            0xffefffffffffffff,
            0x7fefffffffffffff,
            0x3ff0000000000000,
        ],
        enclosure: [0xbff0000000000000, 0x3ff0000000000000],
        exact: [None, None],
    },
    CorrectionCase {
        id: "signed_zero_corrections",
        active: true,
        operands: [
            0xbff0000000000000,
            0x3ff0000000000000,
            0x3fd0000000000000,
            0x8000000000000000,
            0x0000000000000000,
            0x0000000000000001,
        ],
        enclosure: [0x3fd0000000000000, 0x3fd0000000000000],
        exact: [Some(0x3fd0000000000000), Some(0x3fd0000000000000)],
    },
    CorrectionCase {
        id: "one_sided_zero",
        active: true,
        operands: [
            0xbff0000000000000,
            0x3ff0000000000000,
            0x0000000000000000,
            0xbfe0000000000000,
            0x8000000000000000,
            0x3ff0000000000000,
        ],
        enclosure: [0x0000000000000000, 0x3fe0000000000000],
        exact: [Some(0x0000000000000000), None],
    },
    CorrectionCase {
        id: "maximum_finite_domain",
        active: true,
        operands: [
            0xffefffffffffffff,
            0x7fefffffffffffff,
            0x0000000000000000,
            0xbff0000000000000,
            0x3ff0000000000000,
            0x3ff0000000000000,
        ],
        enclosure: [0xbff0000000000000, 0x3ff0000000000000],
        exact: [None, None],
    },
    CorrectionCase {
        id: "inactive_invalid_operands",
        active: false,
        operands: [
            0xbff0000000000000,
            0x3ff0000000000000,
            0x7ff8000000000001,
            0x7ff8000000000001,
            0x7ff0000000000000,
            0x0000000000000000,
        ],
        enclosure: [0xbff0000000000000, 0x3ff0000000000000],
        exact: [Some(0xbff0000000000000), Some(0x3ff0000000000000)],
    },
];
