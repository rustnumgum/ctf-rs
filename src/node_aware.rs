// Adapted from cc4s CTF src/mapping/node_aware_dist.cxx, written by Andreas Irmler.
// Upstream header: Copyright (c) 2022, Edgar Solomonik. See LICENSE.
//! Node-grid candidate enumeration; this is not the contraction communication layer.

fn factors(mut n: usize) -> Vec<usize> {
    assert!(n > 0);
    // Retain upstream's explicit unit factor, which participates in set differences.
    if n < 4 { return vec![n]; }
    let mut result = Vec::new();
    let mut d = 2;
    while n > 1 {
        while n%d == 0 { result.push(d); n /= d; }
        d += 1;
    }
    result
}

/// Sorted multiset subtraction, matching std::set_difference (multiplicity matters).
fn difference(left: &[usize], right: &[usize]) -> Vec<usize> {
    let mut result = Vec::new();
    let mut j = 0;
    for &value in left {
        while j < right.len() && right[j] < value { j += 1; }
        if j < right.len() && right[j] == value { j += 1; }
        else { result.push(value); }
    }
    result
}

#[derive(Clone)]
struct Tree { order: usize, settled: Vec<Vec<usize>>, open: Vec<Vec<usize>> }

/// Return all valid inter-node grids in the pinned source's branch order.
/// Dimension i divides process_grid[i], and each grid's product equals nodes.
pub fn inter_node_grids(process_grid: &[usize], nodes: usize) -> Vec<Vec<usize>> {
    assert!(nodes > 0 && process_grid.iter().all(|&n|n > 0));
    let ranks: usize = process_grid.iter().product();
    assert_eq!(ranks%nodes,0);
    let node_factors = factors(nodes);
    let rank_factors = factors(ranks);
    let mut assigned = Vec::new();
    let mut settled = Vec::new();
    let mut open = Vec::new();
    for &dimension in process_grid {
        let grid_factors = factors(dimension);
        let others = difference(&rank_factors,&grid_factors);
        let mut forced = difference(&node_factors,&others);
        assigned.extend_from_slice(&forced);
        open.push(difference(&grid_factors,&forced));
        if forced.is_empty() { forced.push(1); }
        settled.push(forced);
    }
    assigned.sort_unstable();
    let mut pending = difference(&node_factors,&assigned);
    let mut tree = vec![Tree { order: 0,settled,open }];
    while let Some(factor) = pending.pop() {
        let order = tree.last().unwrap().order;
        let begin = tree.iter().position(|b|b.order == order).unwrap();
        let end = tree.len();
        for index in begin..end {
            for axis in 0..process_grid.len() {
                if let Some(position) = tree[index].open[axis].iter().position(|&f|f == factor) {
                    let mut candidate = tree[index].clone();
                    candidate.order += 1;
                    candidate.settled[axis].push(factor);
                    candidate.settled[axis].sort_unstable();
                    candidate.open[axis].remove(position);
                    if !tree[end..].iter().any(|b|b.settled == candidate.settled) { tree.push(candidate); }
                }
            }
        }
    }
    let order = tree.last().unwrap().order;
    tree.into_iter().filter(|b|b.order == order).map(|b|b.settled.iter().map(|s|s.iter().product()).collect()).collect()
}
