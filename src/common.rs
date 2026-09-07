// Adapted from cc4s CTF interface/common.{h,cxx}.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! Remaining CPU helpers from the pinned interface common layer.
//!
//! The other active responsibilities in that layer already have native Rust
//! owners: symmetry and reduction kinds are typed in `symmetry`/`algebra`,
//! packed sizes are in `util`, communication is in `context` and private MPI
//! FFI, allocation accounting is in `memcontrol`, models are in `cost`, random
//! generation is in `random`, and flop accounting is in `flop_counter`.
//! `conv_idx` is already embedded where labels are normalized. The C++ status
//! integers, assertion/backtrace macros, raw allocation/conversion routines,
//! global debug switches, and raw-MPI datatype/ABI helpers are intentionally
//! excluded rather than exposed as compatibility wrappers.

use std::ops::Add;

/// Return the source's consecutive byte labels starting at `b'a' + start`.
pub fn default_indices(order: usize, start: usize) -> Vec<u8> {
    (0..order)
        .map(|axis| (b'a' as usize + start + axis) as u8)
        .collect()
}

/// Decode a column-major linear index using the source `cvrt_idx` recurrence.
pub fn decode_index(lengths: &[i64], index: i64) -> Vec<i64> {
    let mut remainder = index;
    let mut coordinates = Vec::with_capacity(lengths.len());
    for &length in lengths {
        coordinates.push(remainder % length);
        remainder /= length;
    }
    coordinates
}

/// Encode column-major coordinates using the source `cvrt_idx` recurrence.
pub fn encode_index(lengths: &[i64], coordinates: &[i64]) -> i64 {
    let mut index = 0;
    let mut stride = 1;
    for axis in 0..lengths.len() {
        index += coordinates[axis] * stride;
        stride *= lengths[axis];
    }
    index
}

/// Inclusive scan of `values[0], values[stride], ...` below `n`.
///
/// This retains the source's recursive upsweep/downsweep order. `T::default()`
/// is required to be the additive zero when the same type is passed to
/// [`parallel_prefix`] or [`prefix`].
pub fn parallel_postfix<T>(n: usize, stride: usize, values: &mut [T])
where
    T: Add<Output = T> + Copy,
{
    postfix(n, stride, 0, values);
}

fn postfix<T>(n: usize, stride: usize, base: usize, values: &mut [T])
where
    T: Add<Output = T> + Copy,
{
    if (n + stride - 1) / stride <= 2 {
        if (n + stride - 1) / stride == 2 {
            values[base + stride] = values[base + stride] + values[base];
        }
    } else {
        let stride2 = 2 * stride;
        for i in (stride..n).step_by(stride2) {
            values[base + i] = values[base + i] + values[base + i - stride];
        }
        postfix(n - stride, stride2, base + stride, values);
        for i in (stride..n - stride).step_by(stride2) {
            values[base + i + stride] = values[base + i + stride] + values[base + i];
        }
    }
}

/// Exclusive scan of `values[0], values[stride], ...` below `n`, in place.
pub fn parallel_prefix<T>(n: usize, stride: usize, values: &mut [T])
where
    T: Add<Output = T> + Copy + Default,
{
    exclusive_prefix(n, stride, 0, values);
}

fn exclusive_prefix<T>(n: usize, stride: usize, base: usize, values: &mut [T])
where
    T: Add<Output = T> + Copy + Default,
{
    if n / stride < 2 {
        if (n - 1) / stride >= 1 {
            values[base + stride] = values[base];
        }
        values[base] = T::default();
    } else {
        let stride2 = 2 * stride;
        for i in (stride..n).step_by(stride2) {
            values[base + i] = values[base + i] + values[base + i - stride];
        }
        let nsub = (n + stride - 1) / stride;
        if nsub % 2 != 0 {
            values[base + (nsub - 1) * stride] = values[base + (nsub - 2) * stride];
        }
        exclusive_prefix(n - stride, stride2, base + stride, values);
        if nsub % 2 != 0 {
            let current = base + (nsub - 1) * stride;
            values[current] = values[current] + values[current - stride];
        }
        for i in (stride..n).step_by(stride2) {
            let left = base + i - stride;
            let right = base + i;
            let value = values[left];
            values[left] = values[right];
            values[right] = values[right] + value;
        }
    }
}

/// Copy `input` and compute its exclusive prefix sum.
pub fn prefix<T>(input: &[T]) -> Vec<T>
where
    T: Add<Output = T> + Copy + Default,
{
    let mut output = input.to_vec();
    parallel_prefix(output.len(), 1, &mut output);
    output
}
