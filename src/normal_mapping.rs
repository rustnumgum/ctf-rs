// Adapted from cc4s contraction/contraction.cxx map_to_topology and its
// category helpers. Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! Source-order normal mapping for nonsymmetric, unique-label contractions.

use crate::{
    map_tensor,
    mapping::{Distribution, Mapping, Topology},
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProblemError {
    NonAscii { operand: usize },
    RankMismatch { operand: usize },
    RepeatedLabel { operand: usize, label: u8 },
    LengthMismatch {
        label: u8,
        expected: usize,
        actual: usize,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Negative {
    WeighPhysical,
    ExtraPhysical,
    Assign(map_tensor::Rejected),
}

impl From<map_tensor::Rejected> for Negative {
    fn from(value: map_tensor::Rejected) -> Self {
        Self::Assign(value)
    }
}

#[derive(Clone, Debug)]
pub struct Problem {
    shapes: [Vec<usize>; 3],
    indices: [Vec<usize>; 3],
    inverse: Vec<[Option<usize>; 3]>,
}

fn permutation(order: usize) -> [usize; 3] {
    match order {
        0 => [0, 1, 2],
        1 => [0, 2, 1],
        2 => [2, 0, 1],
        3 => [1, 2, 0],
        4 => [1, 0, 2],
        5 => [2, 1, 0],
        _ => panic!("normal mapping permutation must be in 0..6"),
    }
}

fn virtual_one() -> Mapping {
    Mapping::Virtual {
        copies: 1,
        child: Box::new(Mapping::Unmapped),
    }
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

fn head_factor(mapping: &Mapping) -> usize {
    match mapping {
        Mapping::Unmapped => 1,
        Mapping::Physical { processes, .. } => *processes,
        Mapping::Virtual { copies, .. } => *copies,
    }
}

fn mark_leading_physical(mapping: &Mapping, used: &mut [bool]) {
    if let Mapping::Physical { axis, child, .. } = mapping {
        used[*axis] = true;
        mark_leading_physical(child, used);
    }
}

fn free_axes(topology: &Topology, first: &[Mapping], second: &[Mapping]) -> Vec<usize> {
    let mut used = vec![false; topology.dimensions.len()];
    for mapping in first.iter().chain(second) {
        mark_leading_physical(mapping, &mut used);
    }
    used.iter()
        .enumerate()
        .filter_map(|(axis, &used)| (!used).then_some(axis))
        .collect()
}

fn padded_shapes(shapes: &[Vec<usize>; 3], maps: &[Vec<Mapping>; 3]) -> [Vec<usize>; 3] {
    std::array::from_fn(|operand| {
        shapes[operand]
            .iter()
            .zip(&maps[operand])
            .map(|(&length, mapping)| length.div_ceil(mapping.phase()) * mapping.phase())
            .collect()
    })
}

fn map_weigh(
    labels: &[usize],
    inverse: &[[Option<usize>; 3]],
    operands: [usize; 3],
    padded: &[Vec<usize>; 3],
    topology: &Topology,
    maps: &mut [Vec<Mapping>; 3],
) -> Result<(), Negative> {
    for &label in labels {
        for &operand in &operands {
            if matches!(maps[operand][inverse[label][operand].unwrap()], Mapping::Physical { .. }) {
                return Err(Negative::WeighPhysical);
            }
        }
    }
    let axes = free_axes(topology, &maps[operands[0]], &maps[operands[1]]);
    let mut common = Vec::with_capacity(labels.len());
    let mut lengths = Vec::with_capacity(labels.len());
    for &label in labels {
        let copies = operands.iter().fold(1, |phase, &operand| {
            lcm(
                phase,
                head_factor(&maps[operand][inverse[label][operand].unwrap()]),
            )
        });
        common.push(Mapping::Virtual {
            copies,
            child: Box::new(Mapping::Unmapped),
        });
        lengths.push(padded[operands[0]][inverse[label][operands[0]].unwrap()]);
    }
    map_tensor::assign(
        &lengths,
        topology,
        &axes,
        &vec![false; labels.len() * labels.len()],
        &mut vec![false; labels.len()],
        &mut common,
        false,
    )?;
    for (position, &label) in labels.iter().enumerate() {
        for &operand in &operands {
            maps[operand][inverse[label][operand].unwrap()] = common[position].clone();
        }
    }
    Ok(())
}

fn map_contracted(
    labels: &[usize],
    inverse: &[[Option<usize>; 3]],
    operands: [usize; 3],
    padded: &[Vec<usize>; 3],
    topology: &Topology,
    maps: &mut [Vec<Mapping>; 3],
) -> Result<(), Negative> {
    let axes = free_axes(topology, &maps[operands[0]], &maps[operands[1]]);
    let mut paired = Vec::with_capacity(2 * labels.len());
    let mut lengths = Vec::with_capacity(2 * labels.len());
    let mut table = vec![false; 4 * labels.len() * labels.len()];
    for (position, &label) in labels.iter().enumerate() {
        let left = inverse[label][operands[0]].unwrap();
        let right = inverse[label][operands[1]].unwrap();
        paired.push(maps[operands[0]][left].clone());
        paired.push(maps[operands[1]][right].clone());
        lengths.extend([padded[operands[0]][left]; 2]);
        let order = 2 * labels.len();
        table[(2 * position) * order + 2 * position + 1] = true;
        table[(2 * position + 1) * order + 2 * position] = true;
    }
    map_tensor::assign(
        &lengths,
        topology,
        &axes,
        &table,
        &mut vec![false; 2 * labels.len()],
        &mut paired,
        false,
    )?;
    for (position, &label) in labels.iter().enumerate() {
        maps[operands[0]][inverse[label][operands[0]].unwrap()] = paired[2 * position].clone();
        maps[operands[1]][inverse[label][operands[1]].unwrap()] =
            paired[2 * position + 1].clone();
    }
    Ok(())
}

fn map_remaining(
    operand: usize,
    padded: &[Vec<usize>; 3],
    topology: &Topology,
    maps: &mut [Vec<Mapping>; 3],
    fill: bool,
) -> Result<(), map_tensor::Rejected> {
    let axes = free_axes(topology, &maps[operand], &[]);
    let mut restricted: Vec<bool> = maps[operand]
        .iter()
        .map(|mapping| !matches!(mapping, Mapping::Unmapped))
        .collect();
    map_tensor::assign(
        &padded[operand],
        topology,
        &axes,
        &vec![false; maps[operand].len() * maps[operand].len()],
        &mut restricted,
        &mut maps[operand],
        fill,
    )
}

fn map_noncontracted(
    labels: &[usize],
    inverse: &[[Option<usize>; 3]],
    operands: [usize; 3],
    padded: &[Vec<usize>; 3],
    shapes: &[Vec<usize>; 3],
    topology: &Topology,
    maps: &mut [Vec<Mapping>; 3],
) -> Result<(), Negative> {
    if let Err(error) = map_remaining(operands[0], padded, topology, maps, true) {
        if shapes.iter().any(|shape| !shape.is_empty()) {
            return Err(error.into());
        }
    }
    for &label in labels {
        let output = inverse[label][operands[2]].unwrap();
        if let Some(input) = inverse[label][operands[0]] {
            let mapping = maps[operands[0]][input].clone();
            maps[operands[2]][output] = mapping;
        }
        if let Some(input) = inverse[label][operands[1]] {
            let mapping = maps[operands[1]][input].clone();
            maps[operands[2]][output] = mapping;
        }
    }
    map_remaining(operands[2], padded, topology, maps, false)?;
    for &label in labels {
        let output = inverse[label][operands[2]].unwrap();
        if let Some(input) = inverse[label][operands[0]] {
            let mapping = maps[operands[2]][output].clone();
            maps[operands[0]][input] = mapping;
        }
        if let Some(input) = inverse[label][operands[1]] {
            let mapping = maps[operands[2]][output].clone();
            maps[operands[1]][input] = mapping;
        }
    }
    Ok(())
}

fn map_extra(
    labels: &[usize],
    inverse: &[[Option<usize>; 3]],
    maps: &mut [Vec<Mapping>; 3],
) -> Result<(), Negative> {
    for &label in labels {
        let (operand, dimension) = (0..3)
            .find_map(|operand| inverse[label][operand].map(|dimension| (operand, dimension)))
            .unwrap();
        if matches!(maps[operand][dimension], Mapping::Physical { .. }) {
            return Err(Negative::ExtraPhysical);
        }
        if matches!(maps[operand][dimension], Mapping::Unmapped) {
            maps[operand][dimension] = virtual_one();
        }
    }
    Ok(())
}

impl Problem {
    pub fn new(shapes: [&[usize]; 3], indices: [&str; 3]) -> Result<Self, ProblemError> {
        let mut labels = Vec::new();
        let mut lengths = Vec::new();
        let mut normalized: [Vec<usize>; 3] = std::array::from_fn(|_| Vec::new());
        for operand in 0..3 {
            if !indices[operand].is_ascii() {
                return Err(ProblemError::NonAscii { operand });
            }
            if indices[operand].len() != shapes[operand].len() {
                return Err(ProblemError::RankMismatch { operand });
            }
            for (axis, label) in indices[operand].bytes().enumerate() {
                if indices[operand].as_bytes()[..axis].contains(&label) {
                    return Err(ProblemError::RepeatedLabel { operand, label });
                }
                let id = if let Some(id) = labels.iter().position(|&old| old == label) {
                    if lengths[id] != shapes[operand][axis] {
                        return Err(ProblemError::LengthMismatch {
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
        Ok(Self {
            shapes: shapes.map(|shape| shape.to_vec()),
            indices: normalized,
            inverse,
        })
    }

    /// Return raw candidate maps, including layouts rejected by source preflight.
    /// Call mapping_preflight::check before using these as tensor distributions.
    pub fn map_to_topology(
        &self,
        topology: &Topology,
        permutation_order: usize,
        initial_maps: [Option<&[Mapping]>; 3],
    ) -> Result<[Distribution; 3], Negative> {
        let operands = permutation(permutation_order);
        let mut maps: [Vec<Mapping>; 3] = std::array::from_fn(|operand| {
            if let Some(initial) = initial_maps[operand] {
                assert_eq!(initial.len(), self.shapes[operand].len());
                initial.to_vec()
            } else {
                vec![Mapping::Unmapped; self.shapes[operand].len()]
            }
        });
        let padded = padded_shapes(&self.shapes, &maps);
        let mut weigh = Vec::new();
        let mut contracted = Vec::new();
        let mut noncontracted = Vec::new();
        let mut extra = Vec::new();
        for label in 0..self.inverse.len() {
            let present = operands.map(|operand| self.inverse[label][operand].is_some());
            if present == [true, true, true] {
                weigh.push(label);
            } else if present[0] && present[1] {
                contracted.push(label);
            } else if present[2] && (present[0] || present[1]) {
                noncontracted.push(label);
            } else {
                extra.push(label);
            }
        }

        map_weigh(
            &weigh,
            &self.inverse,
            operands,
            &padded,
            topology,
            &mut maps,
        )?;
        map_contracted(
            &contracted,
            &self.inverse,
            operands,
            &padded,
            topology,
            &mut maps,
        )?;
        map_extra(&extra, &self.inverse, &mut maps)?;
        map_noncontracted(
            &noncontracted,
            &self.inverse,
            operands,
            &padded,
            &self.shapes,
            topology,
            &mut maps,
        )?;
        for operand in 0..3 {
            let order = maps[operand].len();
            map_tensor::coordinate_symmetry(
                &mut maps[operand],
                &vec![false; order * order],
            )?;
        }
        map_contracted(
            &contracted,
            &self.inverse,
            operands,
            &padded,
            topology,
            &mut maps,
        )?;
        map_noncontracted(
            &noncontracted,
            &self.inverse,
            operands,
            &padded,
            &self.shapes,
            topology,
            &mut maps,
        )?;
        for operand in 0..3 {
            let order = maps[operand].len();
            map_tensor::coordinate_symmetry(
                &mut maps[operand],
                &vec![false; order * order],
            )?;
        }

        Ok(std::array::from_fn(|operand| {
            Distribution {
                shape: self.shapes[operand].clone(),
                topology: topology.clone(),
                mappings: std::mem::take(&mut maps[operand]),
            }
        }))
    }

    pub fn normalized_indices(&self) -> &[Vec<usize>; 3] {
        &self.indices
    }
}
