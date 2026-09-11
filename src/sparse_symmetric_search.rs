// Adapted from cc4s CTF summation/summation.cxx::map_sum_indices and map.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! Source-order automatic mapping for a compressed sparse symmetric input and
//! a dense output. The selected layouts are the layouts consumed by execution.

use crate::{
    context::Context,
    cost::Models,
    map_tensor,
    mapping::{Distribution, Mapping, Topology},
    redist_cost,
    symmetric_distribution::SymmetricDistribution,
    symmetry::Symmetry,
};

#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub memory_limit: u64,
}

#[derive(Clone, Debug)]
pub struct Selected {
    pub source_id: usize,
    pub seconds: f64,
    pub memory_bytes: u64,
    pub input: SymmetricDistribution,
    pub output: Distribution,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    NonAscii { operand: usize },
    RankMismatch { operand: usize },
    RepeatedLabel { operand: usize, label: u8 },
    MissingInputLabel { label: u8 },
    LengthMismatch { label: u8, expected: usize, actual: usize },
}

struct Problem {
    input_indices: Vec<usize>,
    output_indices: Vec<usize>,
    dimensions: Vec<usize>,
}

impl Problem {
    fn new(
        input: &SymmetricDistribution,
        output: &Distribution,
        indices: [&str; 2],
    ) -> Result<Self, Error> {
        let shapes = [input.distribution().shape.as_slice(), output.shape.as_slice()];
        let mut labels = Vec::new();
        let mut dimensions = Vec::new();
        let mut normalized: [Vec<usize>; 2] = std::array::from_fn(|_| Vec::new());
        for operand in 0..2 {
            if !indices[operand].is_ascii() {
                return Err(Error::NonAscii { operand });
            }
            if indices[operand].len() != shapes[operand].len() {
                return Err(Error::RankMismatch { operand });
            }
            for (axis, label) in indices[operand].bytes().enumerate() {
                if indices[operand].as_bytes()[..axis].contains(&label) {
                    return Err(Error::RepeatedLabel { operand, label });
                }
                let id = if let Some(id) = labels.iter().position(|&old| old == label) {
                    if dimensions[id] != shapes[operand][axis] {
                        return Err(Error::LengthMismatch {
                            label,
                            expected: dimensions[id],
                            actual: shapes[operand][axis],
                        });
                    }
                    id
                } else {
                    labels.push(label);
                    dimensions.push(shapes[operand][axis]);
                    labels.len() - 1
                };
                normalized[operand].push(id);
            }
        }
        for (&label, &id) in indices[1].as_bytes().iter().zip(&normalized[1]) {
            if !normalized[0].contains(&id) {
                return Err(Error::MissingInputLabel { label });
            }
        }
        Ok(Self {
            input_indices: std::mem::take(&mut normalized[0]),
            output_indices: std::mem::take(&mut normalized[1]),
            dimensions,
        })
    }

    fn symmetry_table(indices: &[usize], links: &[Symmetry], order: usize) -> Vec<bool> {
        let mut table = vec![false; order * order];
        for axis in 0..links.len().saturating_sub(1) {
            if links[axis] != Symmetry::NS {
                let left = indices[axis];
                let right = indices[axis + 1];
                table[left * order + right] = true;
                table[right * order + left] = true;
            }
        }
        table
    }

    fn operand_table(indices: &[usize], links: &[Symmetry]) -> Vec<bool> {
        let mut table = vec![false; indices.len() * indices.len()];
        for axis in 0..links.len().saturating_sub(1) {
            if links[axis] != Symmetry::NS {
                table[axis * indices.len() + axis + 1] = true;
                table[(axis + 1) * indices.len() + axis] = true;
            }
        }
        table
    }

    fn map_candidate(
        &self,
        input: &SymmetricDistribution,
        output: &Distribution,
        topology: &Topology,
        seed: usize,
    ) -> Result<(SymmetricDistribution, Distribution), map_tensor::Rejected> {
        let mut maps: [Vec<Mapping>; 2] = [
            vec![Mapping::Unmapped; self.input_indices.len()],
            vec![Mapping::Unmapped; self.output_indices.len()],
        ];
        let common_lengths: Vec<_> = self.output_indices.iter()
            .map(|&label| self.dimensions[label]).collect();
        let union_table = Self::symmetry_table(
            &self.input_indices,
            input.links(),
            self.dimensions.len(),
        );
        let mut common_table = vec![false; self.output_indices.len().pow(2)];
        for (left, &left_label) in self.output_indices.iter().enumerate() {
            for (right, &right_label) in self.output_indices.iter().enumerate() {
                common_table[left * self.output_indices.len() + right] =
                    union_table[left_label * self.dimensions.len() + right_label];
            }
        }
        let mut common_maps = vec![Mapping::Unmapped; self.output_indices.len()];
        map_tensor::assign(
            &common_lengths,
            topology,
            &(0..topology.dimensions.len()).collect::<Vec<_>>(),
            &common_table,
            &mut vec![false; self.output_indices.len()],
            &mut common_maps,
            false,
        )?;
        for (output_axis, (&label, mapping)) in self.output_indices.iter()
            .zip(&common_maps).enumerate() {
            let input_axis = self.input_indices.iter().position(|&candidate| candidate == label)
                .unwrap();
            maps[0][input_axis] = mapping.clone();
            maps[1][output_axis] = mapping.clone();
        }

        let input_table = Self::operand_table(&self.input_indices, input.links());
        let output_links = vec![Symmetry::NS; self.output_indices.len()];
        let output_table = Self::operand_table(&self.output_indices, &output_links);
        let tables = [&input_table, &output_table];
        let shapes = [input.distribution().shape.as_slice(), output.shape.as_slice()];
        assign_remaining(shapes[seed], topology, tables[seed], &mut maps[seed], true)?;

        let other = 1 - seed;
        for (output_axis, &label) in self.output_indices.iter().enumerate() {
            let input_axis = self.input_indices.iter().position(|&candidate| candidate == label)
                .unwrap();
            if seed == 0 {
                maps[1][output_axis] = maps[0][input_axis].clone();
            } else {
                maps[0][input_axis] = maps[1][output_axis].clone();
            }
        }
        assign_remaining(shapes[other], topology, tables[other], &mut maps[other], false)?;
        map_tensor::coordinate_symmetry(&mut maps[0], &input_table)?;

        // The dense destination of the required custom accumulation must own
        // every process-grid dimension, matching check_sum_mapping.
        let output_axes = used_axes(&maps[1], topology.dimensions.len());
        if output_axes.iter().zip(&topology.dimensions)
            .any(|(&used, &processes)| processes > 1 && !used)
        {
            return Err(map_tensor::Rejected::NoAssignableDimension);
        }
        let mapped_input = SymmetricDistribution::new(
            Distribution::new(input.distribution().shape.clone(), topology.clone(), maps[0].clone()),
            input.links().to_vec(),
        );
        let mapped_output = Distribution::new(output.shape.clone(), topology.clone(), maps[1].clone());
        Ok((mapped_input, mapped_output))
    }
}

fn mark_axes(mapping: &Mapping, used: &mut [bool]) {
    match mapping {
        Mapping::Unmapped => {}
        Mapping::Physical { axis, child, .. } => {
            used[*axis] = true;
            mark_axes(child, used);
        }
        Mapping::Virtual { child, .. } => mark_axes(child, used),
    }
}

fn used_axes(mappings: &[Mapping], order: usize) -> Vec<bool> {
    let mut used = vec![false; order];
    for mapping in mappings { mark_axes(mapping, &mut used); }
    used
}

fn physical_npe(distribution: &Distribution) -> usize {
    used_axes(&distribution.mappings, distribution.topology.dimensions.len())
        .iter()
        .zip(&distribution.topology.dimensions)
        .filter_map(|(&used, &processes)| used.then_some(processes))
        .product()
}

fn assign_remaining(
    shape: &[usize],
    topology: &Topology,
    table: &[bool],
    mappings: &mut [Mapping],
    fill: bool,
) -> Result<(), map_tensor::Rejected> {
    let mut used = vec![false; topology.dimensions.len()];
    for mapping in mappings.iter() { mark_axes(mapping, &mut used); }
    let axes: Vec<_> = used.iter().enumerate()
        .filter_map(|(axis, &used)| (!used).then_some(axis)).collect();
    let mut restricted: Vec<_> = mappings.iter()
        .map(|mapping| !matches!(mapping, Mapping::Unmapped)).collect();
    map_tensor::assign(shape, topology, &axes, table, &mut restricted, mappings, fill)
}

fn same_input(old: &SymmetricDistribution, new: &SymmetricDistribution) -> bool {
    old.links() == new.links()
        && redist_cost::same_mapping(old.distribution(), new.distribution())
}

#[derive(Clone)]
struct Candidate {
    source_id: usize,
    score: f64,
    seconds: f64,
    memory_bytes: u64,
}

#[allow(clippy::too_many_arguments)]
fn candidate(
    source_id: usize,
    old_input: &SymmetricDistribution,
    old_output: &Distribution,
    input: SymmetricDistribution,
    output: Distribution,
    models: &Models,
    nonzeros: u64,
    fraction: f64,
    input_element_bytes: usize,
    input_pair_bytes: usize,
    output_element_bytes: usize,
    memory_limit: u64,
) -> Option<Candidate> {
    let sparse_entries = input.local_len() as f64 * fraction;
    let input_resident = sparse_entries * input_pair_bytes as f64;
    let output_resident = output.local_len() as f64 * output_element_bytes as f64;
    let resident = input_resident + output_resident;
    let log_processes = (output.topology.size() as f64).log2().max(1.);
    // summation::map ranks candidates in packed element counts, independently
    // of the byte-based detailed memory and model estimates below.
    let mut score = (input.local_len() + output.local_len()) as f64;
    let (input_seconds, input_temporary) = if same_input(old_input, &input) {
        (0., 0usize)
    } else {
        let npe = physical_npe(input.distribution());
        score += 25. * npe.min(2) as f64 * nonzeros as f64 / npe as f64
            * log_processes;
        let entries = old_input.local_len().max(input.local_len()) as f64 * fraction;
        let bytes = (entries * input_element_bytes as f64) as usize;
        (
            models.get("spredist_mdl").estimate(&[
                1.,
                (output.topology.size() as f64).log2(),
                bytes as f64 * (output.topology.size() as f64).log2(),
            ]),
            (2. * entries * input_pair_bytes as f64) as usize,
        )
    };
    let output_redistribution = redist_cost::dense(
        old_output,
        &output,
        output_element_bytes,
        models,
    );
    if !redist_cost::same_mapping(old_output, &output) {
        score += if redist_cost::can_block_reshuffle(old_output, &output) {
            output.local_len() as f64 * log_processes
        } else {
            // Execution restores the caller-owned output, matching is_home B.
            10. * output.local_len() as f64 * log_processes
        };
    }
    let memory_bytes = resident as usize
        + input_temporary.max(output_redistribution.temporary_bytes);
    if memory_bytes as u64 >= memory_limit { return None; }
    Some(Candidate {
        source_id,
        score,
        seconds: input_seconds + output_redistribution.seconds,
        memory_bytes: memory_bytes as u64,
    })
}

#[derive(Clone, Copy)]
struct Winner {
    source_id: usize,
    seconds: f64,
    memory_bytes: u64,
}

fn select_global(context: &Context<'_>, local: Option<Candidate>) -> Option<Winner> {
    let score = local.as_ref().map_or(f64::MAX, |candidate| candidate.score);
    let memory = local.as_ref().map_or(-1, |candidate| {
        candidate.memory_bytes.try_into().unwrap()
    });
    let (scores, memories) = context.inner.gather_plan_cost(score, memory);
    let mut winner = [-1i32];
    if context.rank() == 0 {
        let mut best = f64::MAX;
        for rank in 0..context.size() {
            if memories[rank] >= 0 && scores[rank] < best {
                best = scores[rank];
                winner[0] = rank as i32;
            }
        }
    }
    context.broadcast(0, &mut winner);
    if winner[0] < 0 { return None; }
    let root = winner[0] as usize;
    let mut words = if context.rank() == root {
        let selected = local.as_ref().unwrap();
        [selected.source_id as u64, selected.seconds.to_bits(), selected.memory_bytes]
    } else {
        [0; 3]
    };
    context.broadcast(root, &mut words);
    let source_id = words[0] as usize;
    Some(Winner {
        source_id,
        seconds: f64::from_bits(words[1]),
        memory_bytes: words[2],
    })
}

/// Search the pinned two-pass summation mappings. `nonzeros` is the global
/// canonical stored-entry count, excluding physical replicas.
#[allow(clippy::too_many_arguments)]
pub fn search(
    context: &Context<'_>,
    old_input: &SymmetricDistribution,
    old_output: &Distribution,
    indices: [&str; 2],
    catalog: &[Topology],
    models: &Models,
    nonzeros: u64,
    input_element_bytes: usize,
    input_pair_bytes: usize,
    output_element_bytes: usize,
    options: Options,
) -> Result<Option<Selected>, Error> {
    assert_eq!(old_input.distribution().topology.size(), context.size());
    assert_eq!(old_output.topology.size(), context.size());
    assert!(catalog.iter().all(|topology| topology.size() == context.size()));
    let problem = Problem::new(old_input, old_output, indices)?;
    let packed_global = crate::symmetry::Layout::new(
        old_input.distribution().shape.clone(),
        old_input.links().to_vec(),
    ).len();
    let fraction = 1.0f64.min(nonzeros as f64 / packed_global as f64);
    let mut local = None;
    let mut best = f64::MAX;
    for (template, topology) in catalog.iter().enumerate() {
        for seed in 0..2 {
            let source_id = 2 * template + seed;
            if source_id % context.size() != context.rank() { continue; }
            let Ok((input, output)) = problem.map_candidate(
                old_input,
                old_output,
                topology,
                seed,
            ) else { continue };
            let Some(candidate) = candidate(
                source_id,
                old_input,
                old_output,
                input,
                output,
                models,
                nonzeros,
                fraction,
                input_element_bytes,
                input_pair_bytes,
                output_element_bytes,
                options.memory_limit,
            ) else { continue };
            if candidate.score < best {
                best = candidate.score;
                local = Some(candidate);
            }
        }
    }
    let Some(global) = select_global(context, local) else { return Ok(None) };
    let template = global.source_id / 2;
    let seed = global.source_id % 2;
    let (input, output) = problem.map_candidate(
        old_input,
        old_output,
        &catalog[template],
        seed,
    ).unwrap();
    Ok(Some(Selected {
        source_id: global.source_id,
        seconds: global.seconds,
        memory_bytes: global.memory_bytes,
        input,
        output,
    }))
}
