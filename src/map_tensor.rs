// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
// Adapted from src/mapping/mapping.cxx at the pinned cc4s commit.
use crate::mapping::{Mapping, Topology};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rejected {
    PhaseLimit,
    SymmetryIterations,
    NoAssignableDimension,
}

fn lcm(a: usize, b: usize) -> Result<usize, Rejected> {
    let (mut x, mut y) = (a, b);
    while y != 0 {
        (x, y) = (y, x % y);
    }
    let value = (a / x).checked_mul(b).ok_or(Rejected::PhaseLimit)?;
    if value >= 16384 {
        Err(Rejected::PhaseLimit)
    } else {
        Ok(value)
    }
}

/// Synchronize phase along edges of the upstream order-by-order symmetry table.
pub fn coordinate_symmetry(maps: &mut [Mapping], table: &[bool]) -> Result<(), Rejected> {
    let n = maps.len();
    assert_eq!(table.len(), n * n);
    let mut adjusted = true;
    for _ in 0..20 {
        if !adjusted {
            return Ok(());
        }
        adjusted = false;
        for i in 0..n {
            if matches!(maps[i], Mapping::Unmapped) {
                continue;
            }
            let phase = maps[i].phase();
            for j in 0..n {
                if i == j || !table[i * n + j] {
                    continue;
                }
                let sym_phase = maps[j].phase();
                if sym_phase == phase {
                    continue;
                }
                adjusted = true;
                let common = lcm(phase, sym_phase)?;
                if matches!(maps[j], Mapping::Unmapped) || sym_phase != common {
                    maps[j].augment_virtual(common);
                }
                if common > phase {
                    // Upstream keeps the phase captured at the start of this i
                    // iteration while multiplying the terminal virtual factor.
                    let total = maps[i]
                        .phase()
                        .checked_mul(common / phase)
                        .ok_or(Rejected::PhaseLimit)?;
                    maps[i].augment_virtual(total);
                }
            }
        }
    }
    if adjusted {
        Err(Rejected::SymmetryIterations)
    } else {
        Ok(())
    }
}

fn append_physical(map: &mut Mapping, axis: usize, processes: usize) -> Result<(), Rejected> {
    match map {
        Mapping::Physical { child, .. } => append_physical(child, axis, processes),
        Mapping::Unmapped => {
            *map = Mapping::Physical {
                axis,
                processes,
                child: Box::new(Mapping::Unmapped),
            };
            Ok(())
        }
        Mapping::Virtual { copies, child } => {
            assert!(matches!(**child, Mapping::Unmapped));
            let virtual_copies = if *copies == processes {
                1
            } else {
                lcm(*copies, processes)? / processes
            };
            let child = if virtual_copies == 1 {
                Mapping::Unmapped
            } else {
                Mapping::Virtual {
                    copies: virtual_copies,
                    child: Box::new(Mapping::Unmapped),
                }
            };
            *map = Mapping::Physical {
                axis,
                processes,
                child: Box::new(child),
            };
            Ok(())
        }
    }
}

/// Assign the supplied physical axes in order. As upstream, rejection may leave
/// partially assigned maps; callers evaluating candidates own their map copies.
pub fn assign(
    shape: &[usize],
    topology: &Topology,
    axes: &[usize],
    table: &[bool],
    restricted: &mut [bool],
    maps: &mut [Mapping],
    fill: bool,
) -> Result<(), Rejected> {
    assert_eq!(shape.len(), maps.len());
    assert_eq!(restricted.len(), maps.len());
    coordinate_symmetry(maps, table)?;
    for &axis in axes {
        let processes = topology.dimensions[axis];
        let mut selected = None;
        let mut longest = 0;
        for j in 0..shape.len() {
            let length = shape[j] / maps[j].physical_phase();
            if restricted[j] || (selected.is_some() && length <= longest) {
                continue;
            }
            let compatible = match &maps[j] {
                Mapping::Physical { axis: previous, .. } => {
                    fill && axis.checked_sub(1) == Some(*previous)
                }
                _ => true,
            };
            if compatible {
                selected = Some(j);
                longest = length;
            }
        }
        let Some(j) = selected else {
            if fill {
                return Err(Rejected::NoAssignableDimension);
            }
            break;
        };
        append_physical(&mut maps[j], axis, processes)?;
        if !fill {
            restricted[j] = true;
        }
        coordinate_symmetry(maps, table)?;
    }
    for map in maps {
        if matches!(map, Mapping::Unmapped) {
            map.augment_virtual(1);
        }
    }
    Ok(())
}
