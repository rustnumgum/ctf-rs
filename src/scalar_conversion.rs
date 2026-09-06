//! Explicit scalar conversion required by CTF's cast_double coefficient paths.
//! This is an algebra capability, not a numerical backend selection interface.
use crate::algebra::{Arithmetic,Complex,Semiring};

/// Convert a real algorithmic coefficient into this algebra. Custom algebras
/// define their own conversion rather than receiving an implicit f64 fallback.
/// Integer arithmetic truncates fractional coefficients as the upstream cast
/// does; callers must provide finite coefficients representable by the scalar.
pub trait CastFromF64: Semiring {
    fn cast_f64(&self,value:f64)->Self::Element;
}
macro_rules! real {
    ($($t:ty),*) => {$(impl CastFromF64 for Arithmetic<$t> {
        fn cast_f64(&self,value:f64)->$t {value as $t}
    })*};
}
real!(f32,f64,i32,i64);
macro_rules! complex {
    ($($t:ty),*) => {$(impl CastFromF64 for Arithmetic<Complex<$t>> {
        fn cast_f64(&self,value:f64)->Complex<$t> {Complex{re:value as $t,im:0.}}
    })*};
}
complex!(f32,f64);
