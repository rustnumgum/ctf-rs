// Port of the scalar reference kernel in contraction/sym_seq_ctr.cxx.
// Copyright (c) 2011, Edgar Solomonik, all rights reserved. See LICENSE.
//! Raw local packed contraction. Buffers use `sy_packed_size` storage, so AS
//! and SH diagonal slots are physical holes here. Orbit handling, diagonal
//! extraction, and padding cleanup belong to the upper contraction layers.

use crate::algebra::{Group, Semiring};
use crate::symmetry::{Layout, Symmetry};

struct IndexSpace {
    dimensions: Vec<usize>,
    operands: Vec<Vec<usize>>,
}

impl IndexSpace {
    fn new(operands: &[(&Layout, &str)]) -> Self {
        let mut labels = Vec::new();
        let mut dimensions = Vec::new();
        let mut operand_labels = Vec::with_capacity(operands.len());

        for &(layout, indices) in operands {
            assert!(indices.is_ascii());
            assert_eq!(layout.shape().len(), indices.len());
            let mut axes = Vec::with_capacity(indices.len());
            for (&dimension, label) in layout.shape().iter().zip(indices.bytes()) {
                let index = if let Some(index) = labels.iter().position(|&old| old == label) {
                    assert_eq!(dimensions[index], dimension);
                    index
                } else {
                    labels.push(label);
                    dimensions.push(dimension);
                    labels.len() - 1
                };
                axes.push(index);
            }
            operand_labels.push(axes);
        }

        Self {
            dimensions,
            operands: operand_labels,
        }
    }

    /// Traverse global labels from high to low, applying the same inclusive
    /// lower/upper qbounds as `sym_seq_ctr_loop` for every non-NS link.
    fn for_each_canonical(&self, layouts: &[&Layout], mut visit: impl FnMut(&[usize])) {
        let mut constraints = Vec::new();
        for (&layout, axes) in layouts.iter().zip(&self.operands) {
            for axis in 0..layout.links().len().saturating_sub(1) {
                if layout.links()[axis] != Symmetry::NS {
                    constraints.push((axes[axis], axes[axis + 1]));
                }
            }
        }

        fn descend(
            level: usize,
            dimensions: &[usize],
            constraints: &[(usize, usize)],
            index: &mut [usize],
            visit: &mut impl FnMut(&[usize]),
        ) {
            let mut lower = 0;
            let mut upper = dimensions[level];
            for &(left, right) in constraints {
                if left == level && right > level {
                    upper = upper.min(index[right] + 1);
                } else if right == level && left > level {
                    lower = lower.max(index[left]);
                }
            }
            for value in lower..upper {
                index[level] = value;
                if level == 0 {
                    visit(index);
                } else {
                    descend(level - 1, dimensions, constraints, index, visit);
                }
            }
        }

        if self.dimensions.is_empty() {
            visit(&[]);
        } else {
            let mut index = vec![0; self.dimensions.len()];
            descend(
                self.dimensions.len() - 1,
                &self.dimensions,
                &constraints,
                &mut index,
                &mut visit,
            );
        }
    }
}

fn symmetric_group_size(n: usize, order: usize) -> usize {
    let mut product = 1u128;
    for i in 0..order {
        product = product * (n + i) as u128 / (i + 1) as u128;
    }
    product.try_into().unwrap()
}

/// Offset in raw `sy_packed_size` storage. In particular, AS and SH use the
/// weakly ordered SY rank rather than semantic signed/structural-zero access.
fn symmetric_offset(layout: &Layout, axes: &[usize], index: &[usize]) -> usize {
    let mut offset = 0;
    let mut stride = 1;
    let mut start = 0;

    for (end, &link) in layout.links().iter().enumerate() {
        if link != Symmetry::NS {
            continue;
        }
        let mut rank = 0;
        for (within_group, &label) in axes[start..=end].iter().enumerate() {
            rank += symmetric_group_size(index[label], within_group + 1);
        }
        offset += stride * rank;
        stride *= symmetric_group_size(layout.shape()[start], end - start + 1);
        start = end + 1;
    }
    offset
}

/// Local scalar `sym_seq_ctr_ref`:
/// `C = alpha * (A contract B) + beta * C`, with the source's actual operand
/// ordering preserved for noncommutative algebras. The product is formed as
/// `(A * B) * alpha`; nonscalar C is prescaled as `beta * C`, while the source
/// scalar special case uses `C * beta`.
///
/// Repeated C labels do not narrow the beta prescale: just like the reference,
/// this raw kernel scales the entire physical C buffer. A caller that contracts
/// into a diagonal/subset must isolate that output in an upper layer.
pub fn sequential<A: Group + Semiring>(
    algebra: &A,
    layout_a: &Layout,
    indices_a: &str,
    a: &[A::Element],
    layout_b: &Layout,
    indices_b: &str,
    b: &[A::Element],
    layout_c: &Layout,
    indices_c: &str,
    c: &mut [A::Element],
    alpha: &A::Element,
    beta: &A::Element,
) {
    sequential_function(
        algebra,
        layout_a,
        indices_a,
        a,
        layout_b,
        indices_b,
        b,
        layout_c,
        indices_c,
        c,
        alpha,
        beta,
        &|a, b| algebra.multiply(a, b),
    );
}

/// Custom packed contraction with the same traversal and coefficient ordering
/// as [`sequential`], replacing only the elementwise product with `function`.
pub fn sequential_function<A: Group + Semiring>(
    algebra: &A,
    layout_a: &Layout,
    indices_a: &str,
    a: &[A::Element],
    layout_b: &Layout,
    indices_b: &str,
    b: &[A::Element],
    layout_c: &Layout,
    indices_c: &str,
    c: &mut [A::Element],
    alpha: &A::Element,
    beta: &A::Element,
    function: &impl Fn(&A::Element, &A::Element) -> A::Element,
) {
    assert_eq!(a.len(), layout_a.symmetric_len());
    assert_eq!(b.len(), layout_b.symmetric_len());
    assert_eq!(c.len(), layout_c.symmetric_len());

    let space = IndexSpace::new(&[
        (layout_a, indices_a),
        (layout_b, indices_b),
        (layout_c, indices_c),
    ]);

    if space.dimensions.is_empty() {
        let product = function(&a[0], &b[0]);
        let scaled = algebra.multiply(&product, alpha);
        let old = algebra.multiply(&c[0], beta);
        c[0] = algebra.add(&scaled, &old);
        return;
    }

    if *beta == algebra.zero() {
        c.fill(algebra.zero());
    } else if *beta != algebra.one() {
        for value in c.iter_mut() {
            *value = algebra.multiply(beta, value);
        }
    }

    let alpha_is_one = *alpha == algebra.one();
    space.for_each_canonical(&[layout_a, layout_b, layout_c], |index| {
        let offset_a = symmetric_offset(layout_a, &space.operands[0], index);
        let offset_b = symmetric_offset(layout_b, &space.operands[1], index);
        let offset_c = symmetric_offset(layout_c, &space.operands[2], index);
        let product = function(&a[offset_a], &b[offset_b]);
        let scaled = if alpha_is_one {
            product
        } else {
            algebra.multiply(&product, alpha)
        };
        c[offset_c] = algebra.add(&scaled, &c[offset_c]);
    });
}
