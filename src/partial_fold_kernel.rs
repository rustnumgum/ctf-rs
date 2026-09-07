// Adapted from cc4s contraction/{ctr_tsr,sym_seq_ctr}.cxx and
// shared/iter_tsr.h. Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! CPU f64 residual-index traversal around a selected partial folded GEMM.

use crate::{
    contraction::{Folded, folded_f64},
    fold_layout::{Direction,storage_len},
    linalg::LocalKernels,
    partial_fold::Descriptor,
    symmetry::Symmetry,
};

struct Operand {
    lengths: Vec<usize>,
    links: Vec<Symmetry>,
    indices: Vec<usize>,
}

fn normalized(indices: [&str; 3]) -> [Vec<usize>; 3] {
    let mut labels = Vec::new();
    std::array::from_fn(|operand| {
        indices[operand]
            .bytes()
            .map(|label| {
                if let Some(id) = labels.iter().position(|&old| old == label) {
                    id
                } else {
                    labels.push(label);
                    labels.len() - 1
                }
            })
            .collect()
    })
}

fn group_ranges(links: &[Symmetry]) -> Vec<std::ops::Range<usize>> {
    let mut start = 0;
    links
        .iter()
        .enumerate()
        .filter_map(|(end, &link)| {
            if link != Symmetry::NS {
                return None;
            }
            let range = start..end + 1;
            start = end + 1;
            Some(range)
        })
        .collect()
}

fn residual_operand(
    shape: &[usize],
    links: &[Symmetry],
    indices: Vec<usize>,
    descriptor: &Descriptor,
    operand: usize,
) -> Operand {
    let layout = &descriptor.layouts[operand];
    assert_eq!(shape.len(), links.len());
    assert_eq!(
        storage_len(shape,links),
        layout.group_lengths.iter().product()
    );
    let mut lengths = shape.to_vec();
    let mut residual_links = links.to_vec();
    let groups = group_ranges(links);
    for &group in &layout.inner_ordering[..layout.folded_shape.len()] {
        for axis in groups[group].clone() {
            lengths[axis] = 1;
            residual_links[axis] = Symmetry::NS;
        }
    }
    Operand {
        lengths,
        links: residual_links,
        indices,
    }
}

// shared/iter_tsr.h::RESET_IDX. The returned offset counts folded GEMM blocks,
// not scalar elements.
fn reset_offset(operand: &Operand, global: &[usize]) -> usize {
    if operand.indices.is_empty() {
        return 0;
    }
    let mut dynamic_lengths = operand.lengths.clone();
    let mut offset = global[operand.indices[0]];
    let mut group_start = 0;
    let mut leading = 1;
    for axis in 1..operand.indices.len() {
        if operand.links[axis - 1] == Symmetry::NS {
            group_start = axis;
            leading = storage_len(
                &dynamic_lengths[..axis],
                &operand.links[..axis],
            );
            offset += leading * global[operand.indices[axis]];
        } else if global[operand.indices[axis]] != 0 {
            let coordinate = global[operand.indices[axis]];
            let mut count = 1;
            dynamic_lengths[axis] = coordinate;
            loop {
                dynamic_lengths[axis - count] = coordinate;
                count += 1;
                if axis < count || operand.links[axis - count] == Symmetry::NS {
                    break;
                }
            }
            offset += leading
                * storage_len(
                    &dynamic_lengths[group_start..=axis],
                    &operand.links[group_start..=axis],
                );
            for restored in axis + 1 - count..=axis {
                dynamic_lengths[restored] = operand.lengths[restored];
            }
        }
    }
    offset
}

// Pinned default build takes iter_tsr.h's non-SEQ branch: every non-NS link
// uses the nondecreasing comparison, including AS and SH.
fn symmetry_passes(operand: &Operand, global: &[usize]) -> bool {
    (0..operand.links.len()).all(|axis| {
        operand.links[axis] == Symmetry::NS
            || global[operand.indices[axis + 1]] >= global[operand.indices[axis]]
    })
}

fn inverse(operands: &[Operand; 3]) -> Vec<[Option<usize>; 3]> {
    let count = operands
        .iter()
        .flat_map(|operand| operand.indices.iter())
        .copied()
        .max()
        .map_or(0, |label| label + 1);
    let mut result = vec![[None; 3]; count];
    for operand in 0..3 {
        for (axis, &label) in operands[operand].indices.iter().enumerate() {
            result[label][operand] = Some(axis);
        }
    }
    result
}

/// Execute one packed local partial fold. `a`, `b`, and `c` each contain one
/// original local block (not multiple virtual blocks). The descriptor must be
/// the eligible result for the supplied shapes/links/indices.
#[allow(clippy::too_many_arguments)]
pub fn execute<K: LocalKernels>(
    descriptor: &Descriptor,
    shapes: [&[usize]; 3],
    links: [&[Symmetry]; 3],
    indices: [&str; 3],
    a: &[f64],
    b: &[f64],
    c: &mut [f64],
    alpha: f64,
    beta: f64,
) {
    let normalized = normalized(indices);
    let operands: [Operand; 3] = std::array::from_fn(|operand| {
        residual_operand(
            shapes[operand],
            links[operand],
            normalized[operand].clone(),
            descriptor,
            operand,
        )
    });
    let original_sizes: [usize; 3] = std::array::from_fn(|operand| {
        storage_len(shapes[operand],links[operand])
    });
    assert_eq!(a.len(), original_sizes[0]);
    assert_eq!(b.len(), original_sizes[1]);
    assert_eq!(c.len(), original_sizes[2]);
    let packed_a = descriptor.layouts[0].transpose(a, 1, Direction::Forward);
    let packed_b = descriptor.layouts[1].transpose(b, 1, Direction::Forward);
    let mut packed_c = descriptor.layouts[2].transpose(c, 1, Direction::Forward);
    if beta == 0. {
        packed_c.fill(0.);
    } else if beta != 1. {
        for value in &mut packed_c {
            *value *= beta;
        }
    }

    let strides = [
        descriptor.m * descriptor.k * descriptor.batches,
        descriptor.k * descriptor.n * descriptor.batches,
        descriptor.m * descriptor.n * descriptor.batches,
    ];
    let inverse = inverse(&operands);
    let mut global = vec![0usize; inverse.len()];
    let mut offsets = [0usize; 3];
    let mut symmetry_pass = true;
    loop {
        // sym_seq_ctr_inr intentionally executes the all-zero tuple before its
        // first CHECK_SYM, including for AS/SH residual groups.
        if symmetry_pass {
            let starts: [usize;3] = std::array::from_fn(|operand| offsets[operand] * strides[operand]);
            folded_f64::<K>(
                Folded {
                    m: descriptor.m,
                    n: descriptor.n,
                    k: descriptor.k,
                    batches: descriptor.batches,
                    trans_a: descriptor.trans_a,
                    trans_b: descriptor.trans_b,
                    transposed_output: descriptor.transposed_output,
                },
                &packed_a[starts[0]..starts[0] + strides[0]],
                &packed_b[starts[1]..starts[1] + strides[1]],
                &mut packed_c[starts[2]..starts[2] + strides[2]],
                alpha,
                1.,
            );
        }

        let mut changed = None;
        for label in 0..inverse.len() {
            let mut maximum = usize::MAX;
            for operand in 0..3 {
                if let Some(axis) = inverse[label][operand] {
                    maximum = maximum.min(operands[operand].lengths[axis]);
                }
            }
            assert!(global[label] < maximum);
            global[label] += 1;
            if global[label] >= maximum {
                global[label] = 0;
            }
            if global[label] != 0 {
                changed = Some(label);
                break;
            }
        }
        if changed.is_none() {
            break;
        }
        symmetry_pass = symmetry_passes(&operands[0], &global)
            && symmetry_passes(&operands[1], &global)
            && symmetry_passes(&operands[2], &global);
        if symmetry_pass {
            offsets = std::array::from_fn(|operand| reset_offset(&operands[operand], &global));
        }
    }
    let restored = descriptor.layouts[2].transpose(&packed_c, 1, Direction::Backward);
    c.copy_from_slice(&restored);
}
