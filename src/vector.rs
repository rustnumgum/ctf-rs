//! Native one-dimensional value construction from `interface/vector.cxx`.
//!
//! `Vector` itself is not duplicated: a one-dimensional `Tensor` already
//! carries its shape, distribution, algebra, and storage.  The source
//! interface's one non-constructor algorithm, `arange`, is implemented here
//! directly against that tensor representation.

use crate::{
    algebra::{Arithmetic, Complex, Monoid, Ring, Semiring, Wire},
    context::Context,
    mapping::Distribution,
    tensor::Tensor,
};

/// Scalar types for which the source `arange` cast from a global index is
/// meaningful.  The conversion is deliberately explicit instead of relying
/// on a lossy or blanket `From<usize>` bound (which Rust does not provide for
/// floating-point types).
pub trait ArangeElement: Clone + PartialEq + Wire {
    fn from_index(index: usize) -> Self;
}

macro_rules! arange_integer {
    ($($ty:ty),* $(,)?) => {
        $(impl ArangeElement for $ty {
            fn from_index(index: usize) -> Self { index as $ty }
        })*
    };
}

macro_rules! arange_float {
    ($($ty:ty),* $(,)?) => {
        $(impl ArangeElement for $ty {
            fn from_index(index: usize) -> Self { index as $ty }
        })*
    };
}

arange_integer!(i8, i16, i32, i64);
arange_float!(f32, f64);

impl ArangeElement for Complex<f32> {
    fn from_index(index: usize) -> Self { Self::new(index as f32, 0.0) }
}

impl ArangeElement for Complex<f64> {
    fn from_index(index: usize) -> Self { Self::new(index as f64, 0.0) }
}

/// Construct a distributed dense vector with source column-major values
/// `start + index * step`.
///
/// The returned tensor uses the crate's canonical cyclic rank-one
/// distribution.  Every valid local slot, including replicas, is generated
/// from its global key; padding is left at the algebraic zero.  This is the
/// direct Rust equivalent of `Vector::arange`, without a C++-style Vector
/// subclass or an implicit write/redistribution round trip.
pub fn arange<'c, 'r, T>(
    context: &'c Context<'r>,
    start: T,
    len: usize,
    step: T,
) -> Tensor<'c, 'r, Arithmetic<T>>
where
    T: ArangeElement,
    Arithmetic<T>: Ring<Element = T> + Clone,
{
    let distribution = Distribution::cyclic(vec![len], context.size());
    let algebra = Arithmetic::<T>::new();
    let mut vector = Tensor::new(context, distribution.clone(), algebra.clone());
    let rank = context.rank();

    for (offset, value) in vector.data.iter_mut().enumerate() {
        if let Some(index) = distribution.global_key(rank, offset) {
            let index_value = T::from_index(index);
            let scaled = algebra.multiply(&index_value, &step);
            *value = algebra.add(&start, &scaled);
        }
    }
    vector
}
