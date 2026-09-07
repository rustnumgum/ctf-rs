//! Cost-balanced ready-task partitioning adapted from `interface/schedule.cxx`.

/// Select a contiguous descending-cost window of ready tasks and assign every
/// parent rank to one selected task by the midpoint of its equal-cost block.
///
/// The source scheduler uses the longest window whose cheapest task is at
/// least one processor's share of the cumulative cost. Stable sorting keeps
/// input order for equal costs; all cost arithmetic remains `f64`.
pub(crate) fn partition(
    ready: &[usize],
    costs: &[f64],
    size: usize,
    max_partitions: Option<usize>,
) -> Vec<(usize, Vec<usize>)> {
    assert_eq!(ready.len(), costs.len());
    assert!(size > 0);
    assert!(costs.iter().all(|&cost| cost.is_finite() && cost > 0.0));
    if ready.is_empty() {
        return Vec::new();
    }

    let max_partitions = max_partitions.unwrap_or(size);
    assert!(max_partitions > 0);
    let max_colors = size.min(ready.len()).min(max_partitions);
    assert!(max_colors > 0);

    let mut ordered: Vec<_> = ready
        .iter()
        .copied()
        .zip(costs.iter().copied())
        .collect();
    // `sort_by` is stable, so equal-cost tasks retain their input order.
    ordered.sort_by(|left, right| right.1.partial_cmp(&left.1).unwrap());

    let mut best_start = 0;
    let mut best_len = 0;
    let mut best_cost = 0.0;
    for start in 0..ordered.len() {
        let mut sum_cost = 0.0;
        let mut min_cost = 0.0;
        let mut length = 0;
        for index in start..ordered.len() {
            let this_cost = ordered[index].1;
            if min_cost == 0.0 || this_cost < min_cost {
                min_cost = this_cost;
            }
            if min_cost < (this_cost + sum_cost) / size as f64 {
                break;
            }
            length = index - start + 1;
            sum_cost += this_cost;
            if length >= max_colors {
                break;
            }
        }
        if length > best_len {
            best_start = start;
            best_len = length;
            best_cost = sum_cost;
        }
    }

    assert!(best_len > 0);
    let selected = &ordered[best_start..best_start + best_len];
    assert!(best_cost.is_finite() && best_cost > 0.0);
    let processor_block = best_cost / size as f64;
    let mut assignments: Vec<Vec<usize>> = (0..best_len).map(|_| Vec::new()).collect();
    for rank in 0..size {
        let mut sample = processor_block * rank as f64 + processor_block / 2.0;
        let mut task = None;
        for (index, &(_, cost)) in selected.iter().enumerate() {
            if sample < cost {
                task = Some(index);
                break;
            }
            sample -= cost;
        }
        let task = task.expect("cost midpoint must fall inside a selected task");
        assignments[task].push(rank);
    }
    assert!(assignments.iter().all(|ranks| !ranks.is_empty()));

    selected
        .iter()
        .zip(assignments)
        .map(|(&(task, _), ranks)| (task, ranks))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::partition;

    #[test]
    fn longest_window_skips_expensive_outlier() {
        let result = partition(&[0, 1, 2, 3], &[100.0, 10.0, 9.0, 8.0], 4, None);
        assert_eq!(result.iter().map(|(task, _)| *task).collect::<Vec<_>>(), [1, 2, 3]);
        assert_eq!(result.iter().map(|(_, ranks)| ranks.len()).collect::<Vec<_>>(), [1, 2, 1]);
    }

    #[test]
    fn subsecond_costs_remain_partitionable() {
        let result = partition(&[7, 8], &[0.0005, 0.0005], 2, None);
        assert_eq!(result, vec![(7, vec![0]), (8, vec![1])]);
    }

    #[test]
    fn partition_cap_limits_selected_window() {
        let result = partition(&[0, 1, 2, 3], &[4.0, 4.0, 4.0, 4.0], 4, Some(2));
        assert_eq!(result.iter().map(|(task, _)| *task).collect::<Vec<_>>(), [0, 1]);
        assert_eq!(result.iter().map(|(_, ranks)| ranks.len()).collect::<Vec<_>>(), [2, 2]);
    }

    #[test]
    fn size_one_keeps_first_sorted_task() {
        assert_eq!(partition(&[4, 5, 6], &[3.0, 2.0, 1.0], 1, None), vec![(4, vec![0])]);
    }

    #[test]
    fn tied_costs_keep_input_order() {
        let result = partition(&[9, 4, 7], &[2.0, 2.0, 2.0], 3, None);
        assert_eq!(result, vec![(9, vec![0]), (4, vec![1]), (7, vec![2])]);
    }
}
