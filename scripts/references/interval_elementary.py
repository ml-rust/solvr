"""Print the F64 bracket of acos and log for the interval elementary tests.

Run: uv run --with mpmath python scripts/references/interval_elementary.py
Each row is (input bits, largest F64 <= f(x), smallest F64 >= f(x)) at 100 digits.
Paste the output into tests/interval_arithmetic/elementary.rs.
"""

import math
import struct

import mpmath

mpmath.mp.dps = 100


def bits(value: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", value))[0]


def from_bits(value: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", value))[0]


def bracket(true: mpmath.mpf) -> tuple[float, float]:
    nearest = float(true)
    exact = mpmath.mpf(nearest)
    if exact == true:
        return nearest, nearest
    if exact > true:
        return math.nextafter(nearest, -math.inf), nearest
    return nearest, math.nextafter(nearest, math.inf)


def acos_inputs() -> list[float]:
    edge = 2.0**-4
    values = [
        -1.0,
        1.0,
        0.0,
        -0.0,
        0.5,
        -0.5,
        edge,
        -edge,
        math.nextafter(edge, 0.0),
        -math.nextafter(edge, 0.0),
        math.nextafter(edge, 1.0),
        -math.nextafter(edge, 1.0),
        math.nextafter(1.0, 0.0),
        math.nextafter(-1.0, 0.0),
        2.0**-30,
        -(2.0**-30),
        0.9375,
        -0.8764,
        0.3,
        -0.0371,
        from_bits(1),
    ]
    return values


def log_inputs() -> list[float]:
    return [
        1.0,
        2.0,
        0.5,
        10.0,
        3.7,
        math.nextafter(1.0, 2.0),
        math.nextafter(1.0, 0.0),
        from_bits(1),
        2.0**-1022,
        1e-300,
        1.7976931348623157e308,
        math.e,
    ]


def emit(name: str, function, inputs: list[float]) -> None:
    print(f"pub const {name}: [(u64, u64, u64); {len(inputs)}] = [")
    for value in inputs:
        down, up = bracket(function(mpmath.mpf(value)))
        print(f"    (0x{bits(value):016x}, 0x{bits(down):016x}, 0x{bits(up):016x}),")
    print("];")


emit("ACOS", mpmath.acos, acos_inputs())
emit("LOG", mpmath.log, log_inputs())
