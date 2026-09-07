// Adapted from cc4s CTF interface/tensor.cxx.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! Dense and sparse tensor norms for the source's specialized scalar families.

use crate::{
    algebra::{Arithmetic, Complex, Monoid},
    sparse::SparseTensor,
    tensor::Tensor,
};

#[derive(Clone)]
struct Maximum(f64);

impl Monoid for Maximum {
    type Element = f64;

    fn zero(&self) -> f64 {
        self.0
    }

    fn add(&self, left: &f64, right: &f64) -> f64 {
        if right > left { *right } else { *left }
    }
}

fn dense_logical_sum<A: Monoid>(
    tensor: &Tensor<'_, '_, A>,
    magnitude: impl Fn(&A::Element) -> f64,
) -> f64 {
    let rank = tensor.context().rank();
    let mut result = 0.0;
    for (key, value) in tensor.local_pairs() {
        if tensor.distribution().owner(key) == rank {
            result += magnitude(&value);
        }
    }
    tensor.context().sum_f64(std::slice::from_mut(&mut result));
    result
}

fn sparse_logical_sum<A: Monoid>(
    tensor: &SparseTensor<'_, '_, A>,
    magnitude: impl Fn(&A::Element) -> f64,
) -> f64 {
    let rank = tensor.context().rank();
    let mut result = 0.0;
    for (key, value) in tensor.local_pairs() {
        if tensor.distribution().owner(key) == rank {
            result += magnitude(&value);
        }
    }
    tensor.context().sum_f64(std::slice::from_mut(&mut result));
    result
}

fn dense_logical_max<A: Monoid>(
    tensor: &Tensor<'_, '_, A>,
    identity: f64,
    magnitude: impl Fn(&A::Element) -> f64,
) -> f64 {
    let rank = tensor.context().rank();
    let maximum = Maximum(identity);
    let mut result = maximum.zero();
    for (key, value) in tensor.local_pairs() {
        if tensor.distribution().owner(key) == rank {
            result = maximum.add(&result, &magnitude(&value));
        }
    }
    tensor
        .context()
        .all_reduce_monoid(&maximum, std::slice::from_mut(&mut result), true);
    result
}

fn sparse_logical_max<A: Monoid>(
    tensor: &SparseTensor<'_, '_, A>,
    identity: f64,
    magnitude: impl Fn(&A::Element) -> f64,
) -> f64 {
    let rank = tensor.context().rank();
    let maximum = Maximum(identity);
    let mut result = maximum.zero();
    for (key, value) in tensor.local_pairs() {
        if tensor.distribution().owner(key) == rank {
            result = maximum.add(&result, &magnitude(&value));
        }
    }
    tensor
        .context()
        .all_reduce_monoid(&maximum, std::slice::from_mut(&mut result), true);
    result
}

fn dense_manual_norm2<A: Monoid>(
    tensor: &Tensor<'_, '_, A>,
    to_double: impl Fn(&A::Element) -> f64,
) -> f64 {
    let mut squared = 0.0;
    // Source manual_norm2 traverses the whole local allocation: padding and
    // replicas are intentionally neither removed nor canonicalized.
    for value in tensor.local_storage() {
        let value = to_double(value);
        squared += value * value;
    }
    tensor.context().sum_f64(std::slice::from_mut(&mut squared));
    squared.sqrt()
}

fn sparse_manual_norm2<A: Monoid>(
    tensor: &SparseTensor<'_, '_, A>,
    to_double: impl Fn(&A::Element) -> f64,
) -> f64 {
    let mut squared = 0.0;
    // Sparse manual_norm2 likewise traverses every locally stored pair,
    // including replicas, rather than reducing logical canonical owners.
    for (_, value) in tensor.local_pairs() {
        let value = to_double(&value);
        squared += value * value;
    }
    tensor.context().sum_f64(std::slice::from_mut(&mut squared));
    squared.sqrt()
}

macro_rules! real_norms {
    ($scalar:ty, $norm1:expr, $max_identity:expr, $norm_infty:expr, $norm2:expr) => {
        impl Tensor<'_, '_, Arithmetic<$scalar>> {
            pub fn norm1(&self) -> f64 {
                dense_logical_sum(self, $norm1)
            }

            pub fn norm2(&self) -> f64 {
                dense_manual_norm2(self, $norm2)
            }

            pub fn norm_infty(&self) -> f64 {
                dense_logical_max(self, $max_identity, $norm_infty)
            }
        }

        impl SparseTensor<'_, '_, Arithmetic<$scalar>> {
            pub fn norm1(&self) -> f64 {
                sparse_logical_sum(self, $norm1)
            }

            pub fn norm2(&self) -> f64 {
                sparse_manual_norm2(self, $norm2)
            }

            pub fn norm_infty(&self) -> f64 {
                sparse_logical_max(self, $max_identity, $norm_infty)
            }
        }
    };
}

real_norms!(
    i8,
    |value: &i8| (*value as i16).abs() as f64,
    i8::MIN as f64,
    |value: &i8| value.wrapping_abs() as f64,
    |value: &i8| *value as f64
);
real_norms!(
    i16,
    |value: &i16| (*value as i32).abs() as f64,
    i16::MIN as f64,
    |value: &i16| value.wrapping_abs() as f64,
    |value: &i16| *value as f64
);
real_norms!(
    i32,
    |value: &i32| (*value as i64).abs() as f64,
    i32::MIN as f64,
    |value: &i32| value.wrapping_abs() as f64,
    |value: &i32| *value as f64
);
real_norms!(
    i64,
    |value: &i64| value.unsigned_abs() as f64,
    i64::MIN as f64,
    |value: &i64| value.wrapping_abs() as f64,
    |value: &i64| *value as f64
);
real_norms!(
    f32,
    |value: &f32| value.abs() as f64,
    f32::MIN_POSITIVE as f64,
    |value: &f32| value.abs() as f64,
    |value: &f32| *value as f64
);
real_norms!(
    f64,
    |value: &f64| value.abs(),
    f64::MIN_POSITIVE,
    |value: &f64| value.abs(),
    |value: &f64| *value
);

macro_rules! norm2_only {
    ($scalar:ty, $to_double:expr) => {
        impl Tensor<'_, '_, Arithmetic<$scalar>> {
            pub fn norm2(&self) -> f64 {
                dense_manual_norm2(self, $to_double)
            }
        }

        impl SparseTensor<'_, '_, Arithmetic<$scalar>> {
            pub fn norm2(&self) -> f64 {
                sparse_manual_norm2(self, $to_double)
            }
        }
    };
}

norm2_only!(bool, |value: &bool| if *value { 1.0 } else { 0.0 });
norm2_only!(Complex<f32>, |value: &Complex<f32>| value.re.hypot(value.im) as f64);
norm2_only!(Complex<f64>, |value: &Complex<f64>| value.re.hypot(value.im));
