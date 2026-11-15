//! Exact-bit Linear receipt observations on selected actual backends.

#[path = "../tests/linear_receipts/cases/mod.rs"]
mod cases;
#[path = "../tests/linear_receipts/check.rs"]
mod check;
#[path = "../tests/linear_receipts/cpu.rs"]
mod cpu;
#[path = "../tests/linear_receipts/fixture.rs"]
mod fixture;
#[path = "../tests/linear_receipts/observe.rs"]
mod observe;
#[path = "linear_receipts_reference/observer.rs"]
mod observer;

fn main() -> fixture::Result {
    observer::run()
}
