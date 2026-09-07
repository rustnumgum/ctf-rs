// Adapted from cc4s CTF interface/{common,tensor}.cxx.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! MT19937-64 generation and source-compatible dense/sparse random filling.

use crate::{
    algebra::{Arithmetic, Complex, Group, Monoid, Semiring, Wire},
    sparse::SparseTensor,
    tensor::Tensor,
};

const N: usize = 312;
const M: usize = 156;
const MATRIX_A: u64 = 0xb502_6f5a_a966_19e9;
const UPPER_MASK: u64 = u64::MAX << 31;
const LOWER_MASK: u64 = !UPPER_MASK;

pub struct Generator {
    state: [u64; N],
    index: usize,
}

impl Generator {
    pub fn new(seed: u64) -> Self {
        let mut state = [0; N];
        state[0] = seed;
        for i in 1..N {
            state[i] = 6_364_136_223_846_793_005u64
                .wrapping_mul(state[i - 1] ^ (state[i - 1] >> 62))
                .wrapping_add(i as u64);
        }
        Self { state, index: N }
    }

    fn twist(&mut self) {
        for i in 0..N {
            let value = (self.state[i] & UPPER_MASK)
                | (self.state[(i + 1) % N] & LOWER_MASK);
            let mut next = value >> 1;
            if value & 1 != 0 {
                next ^= MATRIX_A;
            }
            self.state[i] = self.state[(i + M) % N] ^ next;
        }
        self.index = 0;
    }

    pub fn next_u64(&mut self) -> u64 {
        if self.index == N {
            self.twist();
        }
        let mut value = self.state[self.index];
        self.index += 1;
        value ^= (value >> 29) & 0x5555_5555_5555_5555;
        value ^= (value << 17) & 0x71d6_7fff_eda6_0000;
        value ^= (value << 37) & 0xfff7_eee0_0000_0000;
        value ^= value >> 43;
        value
    }

    pub fn unit_interval(&mut self) -> f64 {
        self.next_u64() as f64 / u64::MAX as f64
    }
}

macro_rules! fill_random {
    ($scalar:ty, $sample:expr) => {
        impl Tensor<'_, '_, Arithmetic<$scalar>> {
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
                let rank = self.context().rank();
                let distribution = self.distribution().clone();
                for (offset, value) in self.data.iter_mut().enumerate() {
                    if distribution.global_key(rank, offset).is_none() {
                        *value = algebra.zero();
                    }
                }
            }
        }
    };
}

fill_random!(f32, |value: f64| value as f32);
fill_random!(f64, |value: f64| value);
fill_random!(Complex<f32>, |value: f64| Complex::new(
    value as f32,
    0.0
));
fill_random!(Complex<f64>, |value: f64| Complex::new(value, 0.0));

// Source integer filling casts only after the double sample is multiplied
// by the integer span; casting the sample first would almost always give zero.
macro_rules! fill_random_integer {
    ($scalar:ty) => {
        impl Tensor<'_, '_, Arithmetic<$scalar>> {
            pub fn fill_random(
                &mut self,
                minimum: $scalar,
                maximum: $scalar,
                generator: &mut Generator,
            ) {
                let span = maximum.wrapping_sub(minimum);
                for value in &mut self.data {
                    let scaled = (generator.unit_interval() * span as f64) as $scalar;
                    *value = scaled.wrapping_add(minimum);
                }
                let rank = self.context().rank();
                let distribution = self.distribution().clone();
                for (offset, value) in self.data.iter_mut().enumerate() {
                    if distribution.global_key(rank, offset).is_none() {
                        *value = 0;
                    }
                }
            }
        }
    };
}

fill_random_integer!(i32);
fill_random_integer!(i64);

fn local_generation_count(
    total_size: usize,
    fraction: f64,
    processes: usize,
    rank: usize,
) -> usize {
    let mut series = total_size as f64 * fraction;
    let mut expected = 0.0;
    for divisor in 2..20 {
        expected += series;
        series *= fraction / divisor as f64;
    }
    let generated = (expected + 0.5) as usize;
    generated / processes + usize::from(generated % processes > rank)
}

fn fill_dense_sparse<A>(
    tensor: &mut Tensor<'_, '_, A>,
    fraction: f64,
    generator: &mut Generator,
    mut sample: impl FnMut(bool, f64) -> A::Element,
) where
    A: Semiring + Clone,
    A::Element: Wire,
{
    let algebra = tensor.algebra().clone();
    let total_size = tensor.distribution().shape.iter().product::<usize>();
    let generated = local_generation_count(
        total_size,
        fraction,
        tensor.context().size(),
        tensor.context().rank(),
    );
    tensor.data.fill(algebra.zero());
    let mut candidates = Vec::with_capacity(generated);
    for _ in 0..generated {
        let key = (generator.unit_interval() * total_size as f64) as usize;
        candidates.push((key, algebra.one()));
    }
    tensor.write_add(&candidates);
    let zero = algebra.zero();
    tensor.transform(|_, value| {
        let draw = generator.unit_interval();
        *value = sample(*value != zero, draw);
    });
}

fn fill_sparse_sparse<A>(
    tensor: &mut SparseTensor<'_, '_, A>,
    fraction: f64,
    generator: &mut Generator,
    mut sample: impl FnMut(bool, f64) -> A::Element,
) where
    A: Semiring + Clone,
    A::Element: Wire,
{
    let algebra = tensor.algebra().clone();
    let total_size = tensor.distribution().shape.iter().product::<usize>();
    let generated = local_generation_count(
        total_size,
        fraction,
        tensor.context().size(),
        tensor.context().rank(),
    );
    tensor.sparsify(|_| false);
    let mut candidates = Vec::with_capacity(generated);
    for _ in 0..generated {
        let key = (generator.unit_interval() * total_size as f64) as usize;
        candidates.push((key, algebra.one()));
    }
    tensor.write_add(&candidates);
    let zero = algebra.zero();
    tensor.transform_stored(|_, value| {
        let draw = generator.unit_interval();
        *value = sample(*value != zero, draw);
    });
}

macro_rules! fill_random_sparse_numeric {
    ($scalar:ty, $cast:expr) => {
        impl Tensor<'_, '_, Arithmetic<$scalar>> {
            pub fn fill_random_sparse(
                &mut self,
                minimum: $scalar,
                maximum: $scalar,
                fraction: f64,
                generator: &mut Generator,
            ) {
                let algebra = Arithmetic::<$scalar>::new();
                let span = algebra.add(&maximum, &algebra.negate(&minimum));
                fill_dense_sparse(self, fraction, generator, move |present, draw| {
                    let random: $scalar = ($cast)(draw);
                    let value = algebra.add(&algebra.multiply(&random, &span), &minimum);
                    let indicator = if present { algebra.one() } else { algebra.zero() };
                    algebra.multiply(&indicator, &value)
                });
            }
        }

        impl SparseTensor<'_, '_, Arithmetic<$scalar>> {
            pub fn fill_random_sparse(
                &mut self,
                minimum: $scalar,
                maximum: $scalar,
                fraction: f64,
                generator: &mut Generator,
            ) {
                let algebra = Arithmetic::<$scalar>::new();
                let span = algebra.add(&maximum, &algebra.negate(&minimum));
                fill_sparse_sparse(self, fraction, generator, move |present, draw| {
                    let random: $scalar = ($cast)(draw);
                    let value = algebra.add(&algebra.multiply(&random, &span), &minimum);
                    let indicator = if present { algebra.one() } else { algebra.zero() };
                    algebra.multiply(&indicator, &value)
                });
            }
        }
    };
}

fill_random_sparse_numeric!(f32, |value: f64| value as f32);
fill_random_sparse_numeric!(f64, |value: f64| value);
fill_random_sparse_numeric!(Complex<f32>, |value: f64| Complex::new(
    value as f32,
    0.0
));
fill_random_sparse_numeric!(Complex<f64>, |value: f64| Complex::new(value, 0.0));
fill_random_sparse_numeric!(i32, |value: f64| value as i32);
fill_random_sparse_numeric!(i64, |value: f64| value as i64);

impl Tensor<'_, '_, Arithmetic<bool>> {
    pub fn fill_random_sparse(
        &mut self,
        minimum: bool,
        maximum: bool,
        fraction: f64,
        generator: &mut Generator,
    ) {
        fill_dense_sparse(self, fraction, generator, move |present, draw| {
            present && if draw != 0.0 { maximum } else { minimum }
        });
    }
}

impl SparseTensor<'_, '_, Arithmetic<bool>> {
    pub fn fill_random_sparse(
        &mut self,
        minimum: bool,
        maximum: bool,
        fraction: f64,
        generator: &mut Generator,
    ) {
        fill_sparse_sparse(self, fraction, generator, move |present, draw| {
            present && if draw != 0.0 { maximum } else { minimum }
        });
    }
}
