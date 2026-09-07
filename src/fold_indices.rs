// Adapted from cc4s contraction/contraction.cxx::get_fold_indices/can_fold
// (lines 405-584). Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! Fold-index eligibility and selection only. Partial/symmetric folded tensor
//! layouts and execution are separate stages.

use crate::{folding::Operand, symmetry::Symmetry};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    NonAscii { operand: Operand },
    RankMismatch {
        operand: Operand,
        indices: usize,
        links: usize,
    },
    InvalidLinks { operand: Operand, axis: usize },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Eligibility {
    Eligible,
    DenseCustom,
    RepeatedLabel {
        operand: Operand,
        axis: usize,
        label: char,
    },
    SparseNotFullyFolded,
    SparseWeighIndex,
    NoFoldableLabels,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Selection {
    pub eligibility: Eligibility,
    /// Original conv_idx-style IDs: A, then previously unseen B, then C.
    pub labels: Vec<usize>,
}

fn validate(indices: [&str; 3], links: [&[Symmetry]; 3]) -> Result<(), Error> {
    let operands = [Operand::A, Operand::B, Operand::C];
    for operand in 0..3 {
        if !indices[operand].is_ascii() {
            return Err(Error::NonAscii {
                operand: operands[operand],
            });
        }
        if indices[operand].len() != links[operand].len() {
            return Err(Error::RankMismatch {
                operand: operands[operand],
                indices: indices[operand].len(),
                links: links[operand].len(),
            });
        }
        if !links[operand].is_empty() && *links[operand].last().unwrap() != Symmetry::NS {
            return Err(Error::InvalidLinks {
                operand: operands[operand],
                axis: links[operand].len() - 1,
            });
        }
        for axis in 0..links[operand].len().saturating_sub(1) {
            if links[operand][axis] != Symmetry::NS
                && links[operand][axis + 1] != Symmetry::NS
                && links[operand][axis] != links[operand][axis + 1]
            {
                return Err(Error::InvalidLinks {
                    operand: operands[operand],
                    axis: axis + 1,
                });
            }
        }
    }
    Ok(())
}

fn normalized(indices: [&str; 3]) -> ([Vec<usize>; 3], Vec<[Option<usize>; 3]>) {
    let mut labels = Vec::new();
    let ids: [Vec<usize>; 3] = std::array::from_fn(|operand| {
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
    });
    let mut inverse = vec![[None; 3]; labels.len()];
    for operand in 0..3 {
        for (axis, &id) in ids[operand].iter().enumerate() {
            inverse[id][operand] = Some(axis);
        }
    }
    (ids, inverse)
}

fn count(entry: &[Option<usize>; 3]) -> usize {
    entry.iter().filter(|position| position.is_some()).count()
}

fn relative_matches(
    current_position: Option<usize>,
    base_position: Option<usize>,
    current_label: usize,
    base_label: usize,
) -> bool {
    match (current_position, base_position) {
        (Some(current), Some(base)) => {
            current as isize - base as isize == current_label as isize - base_label as isize
        }
        (None, None) => true,
        _ => false,
    }
}

fn symmetry_matches(
    left: Symmetry,
    right_position: Option<usize>,
    right_links: &[Symmetry],
) -> bool {
    right_position.is_none_or(|position| left == right_links[position])
}

fn group_end(links: &[Symmetry], start: usize) -> usize {
    let mut end = start;
    loop {
        end += 1;
        if links[end - 1] == Symmetry::NS {
            return end;
        }
    }
}

fn get_fold_indices(
    ids: &[Vec<usize>; 3],
    inverse: &[[Option<usize>; 3]],
    links: [&[Symmetry]; 3],
) -> Vec<usize> {
    let mut selected = vec![true; inverse.len()];

    // Literal source traversal order A, C, B. Every dimension starts a scan
    // through the remainder of its symmetry group, not only group heads.
    for start in 0..ids[0].len() {
        let base_label = ids[0][start];
        let base = inverse[base_label];
        let base_count = count(&base);
        let end = group_end(links[0], start);
        let mut broken = false;
        for current_a in start..end {
            let current_label = ids[0][current_a];
            let current = inverse[current_label];
            let current_count = count(&current);
            if base_count == 2 {
                if current_count != 2
                    || !relative_matches(current[1], base[1], current_label, base_label)
                    || !relative_matches(current[2], base[2], current_label, base_label)
                    || !symmetry_matches(links[0][current_a], current[1], links[1])
                    || !symmetry_matches(links[0][current_a], current[2], links[2])
                {
                    broken = true;
                }
            } else if base_count != 3 {
                if current_count != 3
                    || links[0][current_a] != links[1][current[1].unwrap()]
                    || links[0][current_a] != links[2][current[2].unwrap()]
                {
                    broken = true;
                }
            } else if current_count != 3
                || !relative_matches(current[1], base[1], current_label, base_label)
                || !relative_matches(current[2], base[2], current_label, base_label)
                || !relative_matches(current[0], base[0], current_label, base_label)
                || links[0][current_a] != links[1][current[1].unwrap()]
                || links[1][current[1].unwrap()] != links[2][current[2].unwrap()]
                || links[0][current_a] != links[2][current[2].unwrap()]
            {
                broken = true;
            }
        }
        if broken {
            for &label in &ids[0][start..end] {
                selected[label] = false;
            }
        }
    }

    for start in 0..ids[2].len() {
        let base_label = ids[2][start];
        let base = inverse[base_label];
        let end = group_end(links[2], start);
        let mut broken = false;
        for current_c in start..end {
            let current_label = ids[2][current_c];
            let current = inverse[current_label];
            if count(&current) == 1
                || !relative_matches(current[0], base[0], current_label, base_label)
                || !relative_matches(current[1], base[1], current_label, base_label)
                || !symmetry_matches(links[2][current_c], current[0], links[0])
                || !symmetry_matches(links[2][current_c], current[1], links[1])
            {
                broken = true;
            }
        }
        if broken {
            for &label in &ids[2][start..end] {
                selected[label] = false;
            }
        }
    }

    for start in 0..ids[1].len() {
        let base_label = ids[1][start];
        let base = inverse[base_label];
        let end = group_end(links[1], start);
        let mut broken = false;
        for current_b in start..end {
            let current_label = ids[1][current_b];
            let current = inverse[current_label];
            if count(&current) == 1
                || !relative_matches(current[2], base[2], current_label, base_label)
                || !relative_matches(current[0], base[0], current_label, base_label)
                || !symmetry_matches(links[1][current_b], current[2], links[2])
                || !symmetry_matches(links[1][current_b], current[0], links[0])
            {
                broken = true;
            }
        }
        if broken {
            for &label in &ids[1][start..end] {
                selected[label] = false;
            }
        }
    }

    selected
        .into_iter()
        .enumerate()
        .filter_map(|(label, keep)| keep.then_some(label))
        .collect()
}

/// Port of can_fold and get_fold_indices. Ineligible is an ordinary source
/// decision; `Error` is reserved for descriptions on which the source assumes
/// statically valid tensor metadata.
pub fn select(
    indices: [&str; 3],
    links: [&[Symmetry]; 3],
    sparse: bool,
    custom: bool,
) -> Result<Selection, Error> {
    validate(indices, links)?;
    if !sparse && custom {
        return Ok(Selection {
            eligibility: Eligibility::DenseCustom,
            labels: Vec::new(),
        });
    }
    let operands = [Operand::A, Operand::B, Operand::C];
    for operand in 0..3 {
        let mut seen = [false; 256];
        for (axis, label) in indices[operand].bytes().enumerate() {
            if seen[label as usize] {
                return Ok(Selection {
                    eligibility: Eligibility::RepeatedLabel {
                        operand: operands[operand],
                        axis,
                        label: label as char,
                    },
                    labels: Vec::new(),
                });
            }
            seen[label as usize] = true;
        }
    }

    let (ids, inverse) = normalized(indices);
    let labels = get_fold_indices(&ids, &inverse, links);
    if sparse {
        let total_order: usize = indices.iter().map(|value| value.len()).sum();
        if total_order % 2 == 1 || total_order / 2 < labels.len() {
            return Ok(Selection {
                eligibility: Eligibility::SparseNotFullyFolded,
                labels,
            });
        }
        if inverse.iter().any(|entry| entry.iter().all(Option::is_some)) {
            return Ok(Selection {
                eligibility: Eligibility::SparseWeighIndex,
                labels,
            });
        }
    }
    Ok(Selection {
        eligibility: if labels.is_empty() {
            Eligibility::NoFoldableLabels
        } else {
            Eligibility::Eligible
        },
        labels,
    })
}
