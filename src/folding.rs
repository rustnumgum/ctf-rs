// Adapted from cc4s CTF contraction/contraction.cxx.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! Local dense nonsymmetric contraction folding.
//!
//! The index grouping and matrix layout follow `get_len_ordering`,
//! `calc_fold_lnmk`, and permutation 0 in CTF's
//! `src/contraction/contraction.cxx` at
//! `f69cbb46e23bc2f39cda5722ce096f56301dab4f`: AB, AC, BC, and ABC labels
//! become `k`, `m`, `n`, and independent GEMM batches `l`, respectively.

use crate::{
    contraction::{folded_f64, Folded},
    linalg::{LocalKernels, Transpose},
};

/// Tensor named by a rejected contraction description.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Operand {
    A,
    B,
    C,
}

/// Reason that a local contraction cannot be represented by the bounded fold.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Rejected {
    NonAsciiIndices { operand: Operand },
    RankMismatch {
        operand: Operand,
        shape: usize,
        indices: usize,
    },
    RepeatedIndex { operand: Operand, label: char },
    DimensionMismatch { label: char },
    OneOperandLabel { operand: Operand, label: char },
    SizeOverflow,
}

/// A fully foldable local dense f64 contraction.
///
/// Packed buffers have column-major layouts `A[k,m,l]`, `B[k,n,l]`, and
/// `C[m,n,l]`. Labels within a group use A's source order for k/m/l and B's
/// source order for n, so differently ordered operands still share one logical
/// flattened coordinate.
#[derive(Clone, Debug)]
pub struct Plan {
    shape_a: Vec<usize>,
    shape_b: Vec<usize>,
    shape_c: Vec<usize>,
    order_a: Vec<usize>,
    order_b: Vec<usize>,
    order_c: Vec<usize>,
    m: usize,
    n: usize,
    k: usize,
    batches: usize,
    size_a: usize,
    size_b: usize,
    size_c: usize,
}

impl Plan {
    /// Classify and plan a unique-index dense contraction.
    pub fn new(shapes: [&[usize]; 3], indices: [&str; 3]) -> Result<Self, Rejected> {
        let operands = [Operand::A, Operand::B, Operand::C];
        for i in 0..3 {
            if !indices[i].is_ascii() {
                return Err(Rejected::NonAsciiIndices {
                    operand: operands[i],
                });
            }
            if shapes[i].len() != indices[i].len() {
                return Err(Rejected::RankMismatch {
                    operand: operands[i],
                    shape: shapes[i].len(),
                    indices: indices[i].len(),
                });
            }
            let mut seen = [false; 256];
            for label in indices[i].bytes() {
                if seen[label as usize] {
                    return Err(Rejected::RepeatedIndex {
                        operand: operands[i],
                        label: label as char,
                    });
                }
                seen[label as usize] = true;
            }
        }

        let mut masks = [0_u8; 256];
        let mut dimensions = [None; 256];
        for i in 0..3 {
            for (&dimension, label) in shapes[i].iter().zip(indices[i].bytes()) {
                if let Some(previous) = dimensions[label as usize] {
                    if previous != dimension {
                        return Err(Rejected::DimensionMismatch {
                            label: label as char,
                        });
                    }
                } else {
                    dimensions[label as usize] = Some(dimension);
                }
                masks[label as usize] |= 1 << i;
            }
        }
        for (label, &mask) in masks.iter().enumerate() {
            let operand = match mask {
                0 | 0b011 | 0b101 | 0b110 | 0b111 => continue,
                0b001 => Operand::A,
                0b010 => Operand::B,
                0b100 => Operand::C,
                _ => unreachable!(),
            };
            return Err(Rejected::OneOperandLabel {
                operand,
                label: label as u8 as char,
            });
        }

        let labels_a = indices[0].as_bytes();
        let labels_b = indices[1].as_bytes();
        let k_labels = labels_with_mask(labels_a, &masks, 0b011);
        let m_labels = labels_with_mask(labels_a, &masks, 0b101);
        let n_labels = labels_with_mask(labels_b, &masks, 0b110);
        let l_labels = labels_with_mask(labels_a, &masks, 0b111);

        let k = label_product(&k_labels, &dimensions)?;
        let m = label_product(&m_labels, &dimensions)?;
        let n = label_product(&n_labels, &dimensions)?;
        let batches = label_product(&l_labels, &dimensions)?;
        let size_a = checked_product(shapes[0])?;
        let size_b = checked_product(shapes[1])?;
        let size_c = checked_product(shapes[2])?;
        let packed_size_a = k
            .checked_mul(m)
            .and_then(|size| size.checked_mul(batches))
            .ok_or(Rejected::SizeOverflow)?;
        let packed_size_b = k
            .checked_mul(n)
            .and_then(|size| size.checked_mul(batches))
            .ok_or(Rejected::SizeOverflow)?;
        let packed_size_c = m
            .checked_mul(n)
            .and_then(|size| size.checked_mul(batches))
            .ok_or(Rejected::SizeOverflow)?;
        debug_assert_eq!(size_a, packed_size_a);
        debug_assert_eq!(size_b, packed_size_b);
        debug_assert_eq!(size_c, packed_size_c);

        let mut packed_a = Vec::new();
        packed_a.extend_from_slice(&k_labels);
        packed_a.extend_from_slice(&m_labels);
        packed_a.extend_from_slice(&l_labels);
        let mut packed_b = Vec::new();
        packed_b.extend_from_slice(&k_labels);
        packed_b.extend_from_slice(&n_labels);
        packed_b.extend_from_slice(&l_labels);
        let mut packed_c = Vec::new();
        packed_c.extend_from_slice(&m_labels);
        packed_c.extend_from_slice(&n_labels);
        packed_c.extend_from_slice(&l_labels);

        Ok(Self {
            shape_a: shapes[0].to_vec(),
            shape_b: shapes[1].to_vec(),
            shape_c: shapes[2].to_vec(),
            order_a: axis_order(labels_a, &packed_a),
            order_b: axis_order(labels_b, &packed_b),
            order_c: axis_order(indices[2].as_bytes(), &packed_c),
            m,
            n,
            k,
            batches,
            size_a,
            size_b,
            size_c,
        })
    }

    /// Pack, execute the existing batched GEMM kernel, and restore C's layout.
    pub fn execute<K: LocalKernels>(
        &self,
        a: &[f64],
        b: &[f64],
        c: &mut [f64],
        alpha: f64,
        beta: f64,
    ) {
        assert_eq!(a.len(), self.size_a);
        assert_eq!(b.len(), self.size_b);
        assert_eq!(c.len(), self.size_c);

        let packed_a = pack(a, &self.shape_a, &self.order_a);
        let packed_b = pack(b, &self.shape_b, &self.order_b);
        let mut packed_c = pack(c, &self.shape_c, &self.order_c);
        folded_f64::<K>(
            Folded {
                m: self.m,
                n: self.n,
                k: self.k,
                batches: self.batches,
                trans_a: Transpose::Yes,
                trans_b: Transpose::No,
                transposed_output: false,
            },
            &packed_a,
            &packed_b,
            &mut packed_c,
            alpha,
            beta,
        );
        unpack(&packed_c, c, &self.shape_c, &self.order_c);
    }
}

fn labels_with_mask(source: &[u8], masks: &[u8; 256], mask: u8) -> Vec<u8> {
    source
        .iter()
        .copied()
        .filter(|&label| masks[label as usize] == mask)
        .collect()
}

fn label_product(labels: &[u8], dimensions: &[Option<usize>; 256]) -> Result<usize, Rejected> {
    labels.iter().try_fold(1_usize, |product, &label| {
        product
            .checked_mul(dimensions[label as usize].unwrap())
            .ok_or(Rejected::SizeOverflow)
    })
}

fn checked_product(shape: &[usize]) -> Result<usize, Rejected> {
    shape.iter().try_fold(1_usize, |product, &dimension| {
        product
            .checked_mul(dimension)
            .ok_or(Rejected::SizeOverflow)
    })
}

fn axis_order(source_labels: &[u8], packed_labels: &[u8]) -> Vec<usize> {
    packed_labels
        .iter()
        .map(|label| source_labels.iter().position(|candidate| candidate == label).unwrap())
        .collect()
}

fn pack(source: &[f64], source_shape: &[usize], order: &[usize]) -> Vec<f64> {
    let source_strides = strides(source_shape);
    let packed_shape: Vec<_> = order.iter().map(|&axis| source_shape[axis]).collect();
    let mut packed = vec![0.; source.len()];
    for (packed_offset, value) in packed.iter_mut().enumerate() {
        let mut remainder = packed_offset;
        let mut source_offset = 0;
        for (&axis, &dimension) in order.iter().zip(&packed_shape) {
            let coordinate = remainder % dimension;
            remainder /= dimension;
            source_offset += coordinate * source_strides[axis];
        }
        *value = source[source_offset];
    }
    packed
}

fn unpack(packed: &[f64], destination: &mut [f64], source_shape: &[usize], order: &[usize]) {
    let source_strides = strides(source_shape);
    let packed_shape: Vec<_> = order.iter().map(|&axis| source_shape[axis]).collect();
    for (packed_offset, &value) in packed.iter().enumerate() {
        let mut remainder = packed_offset;
        let mut destination_offset = 0;
        for (&axis, &dimension) in order.iter().zip(&packed_shape) {
            let coordinate = remainder % dimension;
            remainder /= dimension;
            destination_offset += coordinate * source_strides[axis];
        }
        destination[destination_offset] = value;
    }
}

fn strides(shape: &[usize]) -> Vec<usize> {
    let mut stride = 1;
    shape
        .iter()
        .map(|&dimension| {
            let current = stride;
            stride *= dimension;
            current
        })
        .collect()
}
