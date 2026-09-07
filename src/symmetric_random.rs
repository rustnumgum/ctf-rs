// Adapted from cc4s interface/tensor.cxx::fill_random_base and
// redistribution/pad.cxx::zero_padding.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.

use crate::{
    algebra::{Arithmetic, Complex, Group, Monoid, Semiring},
    random::Generator,
};

use super::SymmetricTensor;

/// Clear every allocated slot that is padding, a packed hole, or a
/// noncanonical AS/SH coordinate.  The source fills the complete allocation
/// first and performs this cleanup afterwards, so this helper must not draw
/// from the generator.
fn zero_out_padding<A: Group>(tensor: &mut SymmetricTensor<'_, '_, A>) {
    let mut valid = vec![false; tensor.data.len()];
    for (offset, _) in tensor.distribution.local_pairs(tensor.context.rank()) {
        valid[offset] = true;
    }
    let zero = tensor.algebra.zero();
    for (value, keep) in tensor.data.iter_mut().zip(valid) {
        if !keep {
            *value = zero.clone();
        }
    }
}

macro_rules! real_fill_random {
    ($scalar:ty, $sample:expr) => {
        impl SymmetricTensor<'_, '_, Arithmetic<$scalar>> {
            /// Fill the full packed allocation, then clear source padding and
            /// noncanonical holes without consuming additional random draws.
            pub fn fill_random(
                &mut self,
                minimum: $scalar,
                maximum: $scalar,
                generator: &mut Generator,
            ) {
                let algebra = Arithmetic::<$scalar>::new();
                let negative_minimum = algebra.negate(&minimum);
                let span = algebra.add(&maximum, &negative_minimum);
                for value in &mut self.data {
                    let random: $scalar = ($sample)(generator.unit_interval());
                    let scaled = algebra.multiply(&random, &span);
                    *value = algebra.add(&scaled, &minimum);
                }
                zero_out_padding(self);
            }
        }
    };
}

real_fill_random!(f32, |value: f64| value as f32);
real_fill_random!(f64, |value: f64| value);

macro_rules! complex_fill_random {
    ($scalar:ty) => {
        impl SymmetricTensor<'_, '_, Arithmetic<Complex<$scalar>>> {
            /// Complex random fills use the source real sample cast and the
            /// same typed semiring multiply/add order as dense tensors.
            pub fn fill_random(
                &mut self,
                minimum: Complex<$scalar>,
                maximum: Complex<$scalar>,
                generator: &mut Generator,
            ) {
                let algebra = Arithmetic::<Complex<$scalar>>::new();
                let negative_minimum = algebra.negate(&minimum);
                let span = algebra.add(&maximum, &negative_minimum);
                for value in &mut self.data {
                    let random = Complex::new(generator.unit_interval() as $scalar, 0 as $scalar);
                    let scaled = algebra.multiply(&random, &span);
                    *value = algebra.add(&scaled, &minimum);
                }
                zero_out_padding(self);
            }
        }
    };
}

complex_fill_random!(f32);
complex_fill_random!(f64);
