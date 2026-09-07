//! Native scalar value access from `interface/scalar.cxx`.
//!
//! A zero-order `Tensor` is the scalar representation in this crate.  The
//! source-specific behavior that is not already supplied by `Tensor` is kept
//! as these typed operations: setting changes only the canonical root, while
//! reading broadcasts that root value to every rank.

use std::slice;

use crate::{algebra::{Monoid, Wire}, tensor::Tensor};

fn assert_scalar<A: Monoid>(scalar: &Tensor<'_, '_, A>) {
    assert!(scalar.distribution().shape.is_empty());
    assert_eq!(scalar.distribution().global_len(), 1);
    assert_eq!(scalar.local_storage().len(), 1);
}

/// Set a zero-order tensor's canonical value on rank zero.
///
/// This intentionally mirrors `Scalar::set_val`: non-root replicas are not
/// silently rewritten.  A subsequent [`value`] call is the synchronization
/// point, matching the source root-broadcast behavior.
pub fn set_value<A>(scalar: &mut Tensor<'_, '_, A>, value: A::Element)
where
    A: Monoid,
{
    assert_scalar(scalar);
    if scalar.context().rank() == 0 {
        scalar.data[0] = value;
    }
}

/// Read a zero-order tensor's value, broadcasting rank zero's entry.
///
/// `Wire` is the native Rust communication contract in place of the source's
/// raw `MPI_CHAR` byte broadcast.
pub fn value<A>(scalar: &Tensor<'_, '_, A>) -> A::Element
where
    A: Monoid,
    A::Element: Wire,
{
    assert_scalar(scalar);
    let mut result = scalar.local_storage()[0].clone();
    scalar.context().broadcast(0, slice::from_mut(&mut result));
    result
}
