// Port of the scalar reference kernel in summation/sym_seq_sum.cxx.
// Copyright (c) 2011, Edgar Solomonik, all rights reserved. See LICENSE.
//! Raw local packed summation. This kernel only intersects the canonical local
//! domains; `sym_sum_tsr`-level orbit processing and padding cleanup are separate.

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

/// Offset in `sy_packed_size` storage. AS and SH deliberately retain the SY
/// diagonal slots here; those slots are raw local holes, not signed reads.
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

fn has_repeated(indices: &str) -> bool {
    indices
        .bytes()
        .enumerate()
        .any(|(i, label)| indices.as_bytes()[..i].contains(&label))
}

#[derive(Clone, Copy)]
struct GroupInfo {
    start: usize,
    end: usize,
    extent: usize,
}

fn groups(layout: &Layout) -> Vec<GroupInfo> {
    let mut start = 0;
    layout
        .links()
        .iter()
        .enumerate()
        .filter_map(|(axis, &link)| {
            if link != Symmetry::NS {
                return None;
            }
            let end = axis + 1;
            let group = GroupInfo {
                start,
                end,
                extent: symmetric_group_size(layout.shape()[start], end - start),
            };
            start = end;
            Some(group)
        })
        .collect()
}

struct FoldedSum {
    groups_a: Vec<GroupInfo>,
    groups_b: Vec<GroupInfo>,
    order_a: Vec<usize>,
    order_b: Vec<usize>,
    residual_layout_a: Layout,
    residual_layout_b: Layout,
    residual_indices_a: String,
    residual_indices_b: String,
    inner_stride: usize,
}

impl FoldedSum {
    fn new(
        layout_a: &Layout,
        indices_a: &str,
        layout_b: &Layout,
        indices_b: &str,
    ) -> Option<Self> {
        if has_repeated(indices_a) || has_repeated(indices_b) {
            return None;
        }
        let groups_a = groups(layout_a);
        let groups_b = groups(layout_b);
        let mut selected = Vec::new();
        for (ga, group_a) in groups_a.iter().enumerate() {
            let labels_a = &indices_a.as_bytes()[group_a.start..group_a.end];
            let Some((gb, group_b)) = groups_b.iter().enumerate().find(|(_, group_b)| {
                &indices_b.as_bytes()[group_b.start..group_b.end] == labels_a
            }) else {
                continue;
            };
            if layout_a.shape()[group_a.start..group_a.end]
                == layout_b.shape()[group_b.start..group_b.end]
                && layout_a.links()[group_a.start..group_a.end]
                    == layout_b.links()[group_b.start..group_b.end]
            {
                selected.push((ga, gb));
            }
        }
        if selected.is_empty() {
            return None;
        }

        let order_a = selected
            .iter()
            .map(|&(ga, _)| ga)
            .chain((0..groups_a.len()).filter(|ga| !selected.iter().any(|&(old, _)| old == *ga)))
            .collect();
        let order_b = selected
            .iter()
            .map(|&(_, gb)| gb)
            .chain((0..groups_b.len()).filter(|gb| !selected.iter().any(|&(_, old)| old == *gb)))
            .collect();
        let residual = |layout: &Layout, indices: &str, operand: usize| {
            let groups = if operand == 0 { &groups_a } else { &groups_b };
            let mut shape = Vec::new();
            let mut links = Vec::new();
            let mut labels = Vec::new();
            for (group_index, group) in groups.iter().enumerate() {
                let folded = selected.iter().any(|&(ga, gb)| {
                    if operand == 0 {
                        ga == group_index
                    } else {
                        gb == group_index
                    }
                });
                if !folded {
                    shape.extend_from_slice(&layout.shape()[group.start..group.end]);
                    links.extend_from_slice(&layout.links()[group.start..group.end]);
                    labels.extend_from_slice(&indices.as_bytes()[group.start..group.end]);
                }
            }
            (Layout::new(shape, links), String::from_utf8(labels).unwrap())
        };
        let (residual_layout_a, residual_indices_a) = residual(layout_a, indices_a, 0);
        let (residual_layout_b, residual_indices_b) = residual(layout_b, indices_b, 1);
        let inner_stride = selected
            .iter()
            .map(|&(ga, _)| groups_a[ga].extent)
            .product();
        Some(Self {
            groups_a,
            groups_b,
            order_a,
            order_b,
            residual_layout_a,
            residual_layout_b,
            residual_indices_a,
            residual_indices_b,
            inner_stride,
        })
    }

    fn execute<A: Semiring>(
        &self,
        algebra: &A,
        a: &[A::Element],
        b: &mut [A::Element],
        alpha: &A::Element,
        beta: &A::Element,
    ) {
        let shape_a: Vec<_> = self.groups_a.iter().map(|group| group.extent).collect();
        let shape_b: Vec<_> = self.groups_b.iter().map(|group| group.extent).collect();
        let packed_a = pack(a, &shape_a, &self.order_a);
        let mut packed_b = pack(b, &shape_b, &self.order_b);
        if *beta == algebra.zero() {
            packed_b.fill(algebra.zero());
        } else if *beta != algebra.one() {
            for value in &mut packed_b {
                *value = algebra.multiply(beta, value);
            }
        }

        let space = IndexSpace::new(&[
            (&self.residual_layout_a, &self.residual_indices_a),
            (&self.residual_layout_b, &self.residual_indices_b),
        ]);
        space.for_each_canonical(
            [&self.residual_layout_a, &self.residual_layout_b].as_slice(),
            |index| {
                let outer_a = symmetric_offset(&self.residual_layout_a, &space.operands[0], index);
                let outer_b = symmetric_offset(&self.residual_layout_b, &space.operands[1], index);
                for inner in 0..self.inner_stride {
                    let value = algebra.multiply(&packed_a[outer_a * self.inner_stride + inner], alpha);
                    let output = &mut packed_b[outer_b * self.inner_stride + inner];
                    *output = algebra.add(&value, output);
                }
            },
        );
        unpack(&packed_b, b, &shape_b, &self.order_b);
    }
}

fn strides(shape: &[usize]) -> Vec<usize> {
    let mut stride = 1;
    shape
        .iter()
        .map(|&dimension| {
            let result = stride;
            stride *= dimension;
            result
        })
        .collect()
}

fn pack<T: Clone>(source: &[T], shape: &[usize], order: &[usize]) -> Vec<T> {
    let source_strides = strides(shape);
    let packed_shape: Vec<_> = order.iter().map(|&axis| shape[axis]).collect();
    let mut packed = source.to_vec();
    for (packed_offset, value) in packed.iter_mut().enumerate() {
        let mut remainder = packed_offset;
        let mut source_offset = 0;
        for (&axis, &dimension) in order.iter().zip(&packed_shape) {
            source_offset += remainder % dimension * source_strides[axis];
            remainder /= dimension;
        }
        *value = source[source_offset].clone();
    }
    packed
}

fn unpack<T: Clone>(packed: &[T], target: &mut [T], shape: &[usize], order: &[usize]) {
    let target_strides = strides(shape);
    let packed_shape: Vec<_> = order.iter().map(|&axis| shape[axis]).collect();
    for (packed_offset, value) in packed.iter().enumerate() {
        let mut remainder = packed_offset;
        let mut target_offset = 0;
        for (&axis, &dimension) in order.iter().zip(&packed_shape) {
            target_offset += remainder % dimension * target_strides[axis];
            remainder /= dimension;
        }
        target[target_offset] = value.clone();
    }
}

/// Local scalar `sym_seq_sum_ref`: `B = A * alpha + beta * B` under indexed
/// summation, with source operand order preserved for noncommutative algebras.
pub fn sequential<A: Group + Semiring>(
    algebra: &A,
    layout_a: &Layout,
    indices_a: &str,
    a: &[A::Element],
    layout_b: &Layout,
    indices_b: &str,
    b: &mut [A::Element],
    alpha: &A::Element,
    beta: &A::Element,
) {
    assert_eq!(a.len(), layout_a.symmetric_len());
    assert_eq!(b.len(), layout_b.symmetric_len());

    let repeated = has_repeated(indices_a) || has_repeated(indices_b);
    if let Some(plan) = FoldedSum::new(layout_a, indices_a, layout_b, indices_b) {
        plan.execute(algebra, a, b, alpha, beta);
        return;
    }
    if repeated {
        if *beta != algebra.one() {
            // SCAL_B scales only the indexed output diagonal when either map
            // repeats; entries outside that selection remain untouched.
            let output = IndexSpace::new(&[(layout_b, indices_b)]);
            output.for_each_canonical(&[layout_b], |index| {
                let offset = symmetric_offset(layout_b, &output.operands[0], index);
                b[offset] = algebra.multiply(beta, &b[offset]);
            });
        }
    } else if *beta == algebra.zero() {
        b.fill(algebra.zero());
    } else if *beta != algebra.one() {
        for value in b.iter_mut() {
            *value = algebra.multiply(beta, value);
        }
    }

    let space = IndexSpace::new(&[(layout_a, indices_a), (layout_b, indices_b)]);
    space.for_each_canonical(&[layout_a, layout_b], |index| {
        let offset_a = symmetric_offset(layout_a, &space.operands[0], index);
        let offset_b = symmetric_offset(layout_b, &space.operands[1], index);
        // The source multiplies A by alpha on the right, then adds that value
        // to B with the scaled input as the left addition operand.
        let scaled = algebra.multiply(&a[offset_a], alpha);
        b[offset_b] = algebra.add(&scaled, &b[offset_b]);
    });
}
