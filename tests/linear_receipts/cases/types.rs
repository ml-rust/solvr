//! Original-bit reference records and expected bracket metadata.

pub struct CaseF32 {
    pub id: &'static str,
    pub x: &'static [f32],
    pub y: &'static [f32],
    pub query: &'static [f32],
    pub legacy_error: Option<&'static str>,
    pub metadata: &'static [(usize, usize, bool, bool, bool)],
}

pub struct CaseF64 {
    pub id: &'static str,
    pub x: &'static [f64],
    pub y: &'static [f64],
    pub query: &'static [f64],
    pub legacy_error: Option<&'static str>,
    pub metadata: &'static [(usize, usize, bool, bool, bool)],
}
