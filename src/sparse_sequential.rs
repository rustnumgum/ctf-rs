// Adapted from cc4s CTF contraction/sp_seq_ctr.cxx at
// f69cbb46e23bc2f39cda5722ce096f56301dab4f.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! Local nonsymmetric sparse-A/dense-B/dense-C contraction.
//!
//! Sparse keys are strictly increasing, unique, unpadded column-major offsets
//! into `shape_a`. Dense buffers are contiguous unpadded column-major storage.
//! Labels must be ASCII and unique within each operand; repeated labels are
//! projected before this kernel. As in the source sparse recursion, an A-only
//! label is invalid, while B-only and C-only labels are traversed locally.

use crate::algebra::Semiring;

struct Plan {
    dimensions: Vec<usize>,
    axis_a: Vec<Option<usize>>,
    offsets_b: Vec<usize>,
    offsets_c: Vec<usize>,
    strides_a: Vec<usize>,
}

impl Plan {
    fn new(
        shape_a: &[usize],
        indices_a: &str,
        shape_b: &[usize],
        indices_b: &str,
        shape_c: &[usize],
        indices_c: &str,
    ) -> Self {
        let operands = [
            (shape_a, indices_a),
            (shape_b, indices_b),
            (shape_c, indices_c),
        ];
        let mut labels = Vec::new();
        let mut dimensions = Vec::new();
        for (shape, indices) in operands {
            assert!(indices.is_ascii());
            assert_eq!(shape.len(), indices.len());
            for (axis, (&dimension, label)) in shape.iter().zip(indices.bytes()).enumerate() {
                assert!(!indices.as_bytes()[..axis].contains(&label),
                    "sparse sequential kernel requires unique labels per operand");
                if let Some(global) = labels.iter().position(|&candidate| candidate == label) {
                    assert_eq!(dimensions[global], dimension);
                } else {
                    labels.push(label);
                    dimensions.push(dimension);
                }
            }
        }

        for label in indices_a.bytes() {
            assert!(indices_b.as_bytes().contains(&label)
                || indices_c.as_bytes().contains(&label),
                "sparse sequential kernel does not accept A-only labels");
        }

        let axis_a = labels.iter().map(|label| {
            indices_a.bytes().position(|candidate| candidate == *label)
        }).collect();
        let offsets_b = label_offsets(&labels, shape_b, indices_b);
        let offsets_c = label_offsets(&labels, shape_c, indices_c);
        let strides_a = strides(shape_a);
        Self { dimensions, axis_a, offsets_b, offsets_c, strides_a }
    }
}

fn checked_len(shape: &[usize]) -> usize {
    shape.iter().try_fold(1usize, |length, &dimension| {
        length.checked_mul(dimension)
    }).expect("tensor size overflow")
}

fn strides(shape: &[usize]) -> Vec<usize> {
    let mut stride = 1usize;
    shape.iter().map(|&dimension| {
        let current = stride;
        stride = stride.checked_mul(dimension).expect("tensor size overflow");
        current
    }).collect()
}

fn label_offsets(labels: &[u8], shape: &[usize], indices: &str) -> Vec<usize> {
    let operand_strides = strides(shape);
    labels.iter().map(|label| {
        indices.bytes().position(|candidate| candidate == *label)
            .map_or(0, |axis| operand_strides[axis])
    }).collect()
}

fn recurse<E, F: Fn(&E, &E, &mut E)>(
    plan: &Plan,
    level: Option<usize>,
    index: &mut [usize],
    a: &[(usize, E)],
    b: &[E],
    c: &mut [E],
    update: &F,
) {
    if a.is_empty() {
        return;
    }
    let Some(label) = level else {
        let offset_b: usize = index.iter().zip(&plan.offsets_b)
            .map(|(&coordinate, &stride)| coordinate * stride).sum();
        let offset_c: usize = index.iter().zip(&plan.offsets_c)
            .map(|(&coordinate, &stride)| coordinate * stride).sum();
        for (_, value_a) in a {
            update(value_a, &b[offset_b], &mut c[offset_c]);
        }
        return;
    };
    let next = label.checked_sub(1);
    if let Some(axis) = plan.axis_a[label] {
        let stride = plan.strides_a[axis];
        let dimension = plan.dimensions[label];
        let mut start = 0;
        while start < a.len() {
            let coordinate = (a[start].0 / stride) % dimension;
            let mut end = start + 1;
            while end < a.len() && (a[end].0 / stride) % dimension == coordinate {
                end += 1;
            }
            index[label] = coordinate;
            recurse(plan, next, index, &a[start..end], b, c, update);
            start = end;
        }
    } else {
        for coordinate in 0..plan.dimensions[label] {
            index[label] = coordinate;
            recurse(plan, next, index, a, b, c, update);
        }
    }
}

/// Execute `C = beta*C + alpha*(A*B)` using the pinned source's sparse-key
/// recursive traversal. For non-scalars, beta left-multiplies C, while alpha
/// right-multiplies each `A*B` product; the contribution is the left argument
/// of addition. The rank-zero scalar source path instead right-multiplies C by
/// beta, and that distinct ordering is retained here.
pub fn sequential<A: Semiring>(
    algebra: &A,
    shape_a: &[usize],
    indices_a: &str,
    a: &[(usize, A::Element)],
    shape_b: &[usize],
    indices_b: &str,
    b: &[A::Element],
    shape_c: &[usize],
    indices_c: &str,
    c: &mut [A::Element],
    alpha: &A::Element,
    beta: &A::Element,
) {
    let plan = Plan::new(shape_a, indices_a, shape_b, indices_b, shape_c, indices_c);
    let length_a = checked_len(shape_a);
    assert_eq!(b.len(), checked_len(shape_b));
    assert_eq!(c.len(), checked_len(shape_c));
    for pair in a.windows(2) {
        assert!(pair[0].0 < pair[1].0, "sparse A keys must be strictly increasing");
    }
    assert!(a.iter().all(|(key, _)| *key < length_a));

    if plan.dimensions.is_empty() {
        assert_eq!(a.len(), 1);
        let product = algebra.multiply(&a[0].1, &b[0]);
        let scaled = algebra.multiply(&product, alpha);
        let previous = algebra.multiply(&c[0], beta);
        c[0] = algebra.add(&scaled, &previous);
        return;
    }

    let one = algebra.one();
    if beta != &one {
        let zero = algebra.zero();
        if beta == &zero {
            c.fill(zero);
        } else {
            for value in c.iter_mut() {
                *value = algebra.multiply(beta, value);
            }
        }
    }

    let mut index = vec![0; plan.dimensions.len()];
    recurse(&plan, plan.dimensions.len().checked_sub(1), &mut index, a, b, c,
        &|value_a, value_b, output| {
            let product = algebra.multiply(value_a, value_b);
            let scaled = algebra.multiply(&product, alpha);
            *output = algebra.add(&scaled, output);
        });
}

/// Custom-function branch of the pinned sparse sequential kernel. The source
/// can reach `func->acc_f` only when A is scalar and the global index space is
/// nonempty, and then only with multiplicative-identity alpha. Accumulation is
/// `old_C + function(A, B)`. The all-scalar source branch does not invoke the
/// custom function at all; it uses ordinary semiring multiplication and retains
/// the scalar right-beta ordering.
pub fn sequential_function<
    A: Semiring,
    F: Fn(&A::Element, &A::Element) -> A::Element,
>(
    algebra: &A,
    shape_a: &[usize],
    indices_a: &str,
    a: &[(usize, A::Element)],
    shape_b: &[usize],
    indices_b: &str,
    b: &[A::Element],
    shape_c: &[usize],
    indices_c: &str,
    c: &mut [A::Element],
    alpha: &A::Element,
    beta: &A::Element,
    function: F,
) {
    let plan = Plan::new(shape_a, indices_a, shape_b, indices_b, shape_c, indices_c);
    let length_a = checked_len(shape_a);
    assert_eq!(b.len(), checked_len(shape_b));
    assert_eq!(c.len(), checked_len(shape_c));
    for pair in a.windows(2) {
        assert!(pair[0].0 < pair[1].0, "sparse A keys must be strictly increasing");
    }
    assert!(a.iter().all(|(key, _)| *key < length_a));

    if plan.dimensions.is_empty() {
        assert_eq!(a.len(), 1);
        let product = algebra.multiply(&a[0].1, &b[0]);
        let scaled = algebra.multiply(&product, alpha);
        let previous = algebra.multiply(&c[0], beta);
        c[0] = algebra.add(&scaled, &previous);
        return;
    }

    let one = algebra.one();
    if beta != &one {
        let zero = algebra.zero();
        if beta == &zero {
            c.fill(zero);
        } else {
            for value in c.iter_mut() {
                *value = algebra.multiply(beta, value);
            }
        }
    }

    let mut index = vec![0; plan.dimensions.len()];
    recurse(&plan, plan.dimensions.len().checked_sub(1), &mut index, a, b, c,
        &|value_a, value_b, output| {
            assert!(indices_a.is_empty(),
                "source custom sparse sequential kernel requires scalar A");
            assert!(alpha == &one,
                "source custom sparse sequential kernel requires identity alpha");
            let value = function(value_a, value_b);
            *output = algebra.add(output, &value);
        });
}
