// Adapted from cc4s contraction/contraction.cxx and shared/util.cxx.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! Raw exhaustive NS mapping variants. Candidate preflight and topology-axis
//! canonicalization are separate source stages.

use crate::mapping::{Distribution, Mapping, Topology};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Rejected {
    NonAscii { operand: usize },
    RankMismatch { operand: usize },
    RepeatedLabel { operand: usize, label: u8 },
    LengthMismatch {
        label: u8,
        expected: usize,
        actual: usize,
    },
    VariantCountOverflow,
    VariantOutOfRange { variant: usize, count: usize },
}

#[derive(Clone, Debug)]
pub struct Variant {
    pub distributions: [Distribution; 3],
    pub two_dimensional: usize,
    pub replicated_labels: Vec<usize>,
    pub orientations: Vec<u8>,
    pub ab_labels: Vec<usize>,
    pub ac_labels: Vec<usize>,
    pub bc_labels: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct VariantSpace {
    shapes: [Vec<usize>; 3],
    indices: [Vec<usize>; 3],
    topology: Topology,
    inverse: Vec<[Option<usize>; 3]>,
    ab: Vec<usize>,
    ac: Vec<usize>,
    bc: Vec<usize>,
    count: usize,
}

fn choose(n: usize, k: usize) -> Result<usize, Rejected> {
    if k > n {
        return Ok(0);
    }
    let k = k.min(n - k);
    let mut value = 1usize;
    for i in 0..k {
        value = value
            .checked_mul(n - i)
            .ok_or(Rejected::VariantCountOverflow)?
            / (i + 1);
    }
    Ok(value)
}

fn checked_product(values: &[usize]) -> Result<usize, Rejected> {
    values.iter().try_fold(1usize, |product, &value| {
        product
            .checked_mul(value)
            .ok_or(Rejected::VariantCountOverflow)
    })
}

// shared/util.cxx packed_size for the SH,...,SH,NS table used by get_choice.
fn packed_size(order: usize, lengths: &[i64], hollow: &[bool]) -> i64 {
    if order == 0 {
        return 1;
    }
    let mut k = 1i64;
    let mut partial = 1i64;
    let mut size = 1i64;
    let mut maximum = lengths[0];
    for i in 0..order {
        partial = partial * maximum / k;
        k += 1;
        // Both SH and the terminating NS differ from source SY.
        maximum -= 1;
        if !hollow[i] {
            size *= partial;
            k = 1;
            partial = 1;
            if i + 1 < order {
                maximum = lengths[i + 1];
            }
        }
    }
    size * partial
}

// Direct get_choice/calc_idx_arr port. In particular choice zero for k>1 is
// all zeroes, matching the pinned source's "FIXME add 1?" behavior.
fn get_choice(n: usize, k: usize, choice: usize) -> Vec<usize> {
    if k == 0 {
        return Vec::new();
    }
    if k == 1 {
        return vec![choice];
    }
    let mut lengths = vec![n as i64; k];
    let mut hollow = vec![true; k];
    hollow[k - 1] = false;
    let mut coordinates = vec![0usize; k];
    let mut remainder = choice as i64;
    for dimension in (0..k).rev() {
        if remainder == 0 {
            break;
        }
        if dimension == 0 || !hollow[dimension - 1] {
            let stride = packed_size(dimension, &lengths, &hollow);
            coordinates[dimension] = (remainder / stride) as usize;
            remainder -= coordinates[dimension] as i64 * stride;
        } else {
            let mut group = 2usize;
            let mut factorial = 2usize;
            while dimension >= group && hollow[dimension - group] {
                group += 1;
                factorial *= group;
            }
            let stride = packed_size(dimension - group + 1, &lengths, &hollow);
            let scaled = remainder as f64 * factorial as f64 / stride as f64;
            let mut maximum = scaled.powf(1. / group as f64) as i64 + group as i64 + 1;
            let mut prefix;
            loop {
                for length in &mut lengths[dimension - group + 1..=dimension] {
                    *length = maximum;
                }
                prefix = packed_size(dimension + 1, &lengths, &hollow);
                if prefix <= remainder || maximum == 0 {
                    break;
                }
                maximum -= 1;
            }
            if prefix == 0 {
                maximum = 0;
            }
            coordinates[dimension] = maximum as usize;
            remainder -= prefix;
        }
    }
    assert_eq!(remainder, 0);
    coordinates
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

fn lcm(a: usize, b: usize) -> usize {
    a / gcd(a, b) * b
}

impl VariantSpace {
    pub fn new(
        shapes: [&[usize]; 3],
        indices: [&str; 3],
        topology: Topology,
    ) -> Result<Self, Rejected> {
        let mut labels = Vec::new();
        let mut lengths = Vec::new();
        let mut normalized: [Vec<usize>; 3] = std::array::from_fn(|_| Vec::new());
        for operand in 0..3 {
            if !indices[operand].is_ascii() {
                return Err(Rejected::NonAscii { operand });
            }
            if indices[operand].len() != shapes[operand].len() {
                return Err(Rejected::RankMismatch { operand });
            }
            for (axis, label) in indices[operand].bytes().enumerate() {
                if indices[operand].as_bytes()[..axis].contains(&label) {
                    return Err(Rejected::RepeatedLabel { operand, label });
                }
                let id = if let Some(id) = labels.iter().position(|&old| old == label) {
                    if lengths[id] != shapes[operand][axis] {
                        return Err(Rejected::LengthMismatch {
                            label,
                            expected: lengths[id],
                            actual: shapes[operand][axis],
                        });
                    }
                    id
                } else {
                    labels.push(label);
                    lengths.push(shapes[operand][axis]);
                    labels.len() - 1
                };
                normalized[operand].push(id);
            }
        }

        let mut inverse = vec![[None; 3]; labels.len()];
        for operand in 0..3 {
            for (axis, &label) in normalized[operand].iter().enumerate() {
                inverse[label][operand] = Some(axis);
            }
        }
        let ab: Vec<_> = inverse
            .iter()
            .enumerate()
            .filter_map(|(label, entry)| {
                (entry[0].is_some() && entry[1].is_some() && entry[2].is_none())
                    .then_some(label)
            })
            .collect();
        let ac: Vec<_> = inverse
            .iter()
            .enumerate()
            .filter_map(|(label, entry)| {
                (entry[0].is_some() && entry[1].is_none() && entry[2].is_some())
                    .then_some(label)
            })
            .collect();
        let bc: Vec<_> = inverse
            .iter()
            .enumerate()
            .filter_map(|(label, entry)| {
                (entry[0].is_none() && entry[1].is_some() && entry[2].is_some())
                    .then_some(label)
            })
            .collect();
        let maximum_2d = (topology.dimensions.len() / 2)
            .min(ab.len())
            .min(ac.len())
            .min(bc.len());
        let mut count = 0usize;
        for two_dimensional in 0..=maximum_2d {
            let remaining = topology.dimensions.len() - 2 * two_dimensional;
            if remaining <= inverse.len() {
                let factors = [
                    3usize
                        .checked_pow(two_dimensional as u32)
                        .ok_or(Rejected::VariantCountOverflow)?,
                    choose(inverse.len(), remaining)?,
                    choose(ab.len(), two_dimensional)?,
                    choose(ac.len(), two_dimensional)?,
                    choose(bc.len(), two_dimensional)?,
                ];
                count = count
                    .checked_add(checked_product(&factors)?)
                    .ok_or(Rejected::VariantCountOverflow)?;
            }
        }
        Ok(Self {
            shapes: shapes.map(|shape| shape.to_vec()),
            indices: normalized,
            topology,
            inverse,
            ab,
            ac,
            bc,
            count,
        })
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn normalized_indices(&self) -> &[Vec<usize>; 3] {
        &self.indices
    }

    fn bucket_size(&self, two_dimensional: usize) -> Result<usize, Rejected> {
        let remaining = self.topology.dimensions.len() - 2 * two_dimensional;
        checked_product(&[
            3usize
                .checked_pow(two_dimensional as u32)
                .ok_or(Rejected::VariantCountOverflow)?,
            choose(self.inverse.len(), remaining)?,
            choose(self.ab.len(), two_dimensional)?,
            choose(self.ac.len(), two_dimensional)?,
            choose(self.bc.len(), two_dimensional)?,
        ])
    }

    pub fn decode(&self, variant: usize) -> Result<Variant, Rejected> {
        if variant >= self.count {
            return Err(Rejected::VariantOutOfRange {
                variant,
                count: self.count,
            });
        }
        let maximum_2d = (self.topology.dimensions.len() / 2)
            .min(self.ab.len())
            .min(self.ac.len())
            .min(self.bc.len());
        let mut offset = 0usize;
        let mut selected = None;
        for two_dimensional in 0..=maximum_2d {
            let remaining = self.topology.dimensions.len() - 2 * two_dimensional;
            if remaining > self.inverse.len() {
                continue;
            }
            let size = self.bucket_size(two_dimensional)?;
            if variant < offset + size {
                selected = Some((two_dimensional, variant - offset));
                break;
            }
            offset += size;
        }
        let (two_dimensional, mut value) = selected.unwrap();
        let remaining = self.topology.dimensions.len() - 2 * two_dimensional;
        let replicated_count = choose(self.inverse.len(), remaining)?;
        let replicated_labels = get_choice(self.inverse.len(), remaining, value % replicated_count);
        value /= replicated_count;

        let mut orientations = Vec::with_capacity(two_dimensional);
        for _ in 0..two_dimensional {
            orientations.push((value % 3) as u8);
            value /= 3;
        }
        let ab_count = choose(self.ab.len(), two_dimensional)?;
        let ab_labels = get_choice(self.ab.len(), two_dimensional, value % ab_count)
            .into_iter()
            .map(|choice| self.ab[choice])
            .collect::<Vec<_>>();
        value /= ab_count;
        let ac_count = choose(self.ac.len(), two_dimensional)?;
        let ac_labels = get_choice(self.ac.len(), two_dimensional, value % ac_count)
            .into_iter()
            .map(|choice| self.ac[choice])
            .collect::<Vec<_>>();
        value /= ac_count;
        let bc_count = choose(self.bc.len(), two_dimensional)?;
        let bc_labels = get_choice(self.bc.len(), two_dimensional, value % bc_count)
            .into_iter()
            .map(|choice| self.bc[choice])
            .collect::<Vec<_>>();

        let mut maps: [Vec<Mapping>; 3] =
            std::array::from_fn(|operand| vec![Mapping::Unmapped; self.shapes[operand].len()]);
        for (axis, &label) in (2 * two_dimensional..self.topology.dimensions.len())
            .zip(&replicated_labels)
        {
            for operand in 0..3 {
                if let Some(dimension) = self.inverse[label][operand] {
                    maps[operand][dimension].augment_physical(&self.topology, axis);
                }
            }
        }
        for i in 0..two_dimensional {
            let first = 2 * i;
            let second = first + 1;
            let ab = self.inverse[ab_labels[i]];
            match orientations[i] {
                0 => {
                    maps[0][ab[0].unwrap()].augment_physical(&self.topology, first);
                    maps[1][ab[1].unwrap()].augment_physical(&self.topology, second);
                }
                1 | 2 => {
                    maps[0][ab[0].unwrap()].augment_physical(&self.topology, first);
                    maps[1][ab[1].unwrap()].augment_physical(&self.topology, first);
                }
                _ => unreachable!(),
            }

            let ac = self.inverse[ac_labels[i]];
            match orientations[i] {
                // Pinned source cases 0 and 2 are identical; case 1 is empty.
                0 | 2 => {
                    maps[0][ac[0].unwrap()].augment_physical(&self.topology, second);
                    maps[2][ac[2].unwrap()].augment_physical(&self.topology, second);
                }
                1 => {}
                _ => unreachable!(),
            }

            let bc = self.inverse[bc_labels[i]];
            match orientations[i] {
                0 => {
                    maps[1][bc[1].unwrap()].augment_physical(&self.topology, first);
                    maps[2][bc[2].unwrap()].augment_physical(&self.topology, first);
                }
                1 => {
                    maps[1][bc[1].unwrap()].augment_physical(&self.topology, second);
                    maps[2][bc[2].unwrap()].augment_physical(&self.topology, second);
                }
                2 => {
                    maps[1][bc[1].unwrap()].augment_physical(&self.topology, second);
                    maps[2][bc[2].unwrap()].augment_physical(&self.topology, first);
                }
                _ => unreachable!(),
            }
        }

        let mut changed = true;
        while changed {
            changed = false;
            for label in 0..self.inverse.len() {
                let phase = (0..3)
                    .filter_map(|operand| {
                        self.inverse[label][operand]
                            .map(|dimension| maps[operand][dimension].phase())
                    })
                    .fold(1, lcm);
                for operand in 0..3 {
                    if let Some(dimension) = self.inverse[label][operand] {
                        if maps[operand][dimension].phase() != phase {
                            maps[operand][dimension].augment_virtual(phase);
                            changed = true;
                        }
                    }
                }
            }
        }

        let distributions = std::array::from_fn(|operand| {
            Distribution::new(
                self.shapes[operand].clone(),
                self.topology.clone(),
                std::mem::take(&mut maps[operand]),
            )
        });
        Ok(Variant {
            distributions,
            two_dimensional,
            replicated_labels,
            orientations,
            ab_labels,
            ac_labels,
            bc_labels,
        })
    }
}
