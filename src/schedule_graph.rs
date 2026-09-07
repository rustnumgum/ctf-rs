// Dependency construction follows cc4s CTF interface/schedule.cxx
// add_operation_typed and schedule_op_successors.
use std::collections::{BTreeMap, BTreeSet};

#[derive(Default)]
pub(crate) struct Graph {
    successors: Vec<BTreeSet<usize>>,
    dependency_counts: Vec<usize>,
    latest_write: BTreeMap<usize, usize>,
    readers: BTreeMap<usize, BTreeSet<usize>>,
}

impl Graph {
    /// Add one operation and return its stable insertion-order identifier.
    pub(crate) fn add(&mut self, inputs: &[usize], output: usize) -> usize {
        let id = self.successors.len();
        self.successors.push(BTreeSet::new());
        self.dependency_counts.push(0);

        let mut dependencies = BTreeSet::new();
        let inputs: BTreeSet<_> = inputs.iter().copied().collect();
        // Record reads before resetting the output's reader set. If this is an
        // in-place update, the new operation is then excluded from its own WAR.
        for input in inputs {
            if let Some(&writer) = self.latest_write.get(&input) {
                dependencies.insert(writer);
            }
            self.readers.entry(input).or_default().insert(id);
        }

        if let Some(&writer) = self.latest_write.get(&output) {
            dependencies.insert(writer);
        }
        if let Some(readers) = self.readers.get(&output) {
            dependencies.extend(readers.iter().copied().filter(|&reader| reader != id));
        }

        for dependency in dependencies {
            if self.successors[dependency].insert(id) {
                self.dependency_counts[id] += 1;
            }
        }
        self.latest_write.insert(output, id);
        self.readers.insert(output, BTreeSet::new());
        id
    }

    pub(crate) fn start(&self) -> Run {
        let ready = self
            .dependency_counts
            .iter()
            .enumerate()
            .filter_map(|(id, &count)| (count == 0).then_some(id))
            .collect();
        Run {
            ready,
            dependency_left: self.dependency_counts.clone(),
        }
    }
}

pub(crate) struct Run {
    pub(crate) ready: Vec<usize>,
    dependency_left: Vec<usize>,
}

impl Run {
    pub(crate) fn finish(&mut self, graph: &Graph, ids: &[usize]) {
        let completed: BTreeSet<_> = ids.iter().copied().collect();
        assert_eq!(completed.len(), ids.len());
        assert!(completed.iter().all(|id| self.ready.contains(id)));
        self.ready.retain(|id| !completed.contains(id));

        for &id in ids {
            for &successor in &graph.successors[id] {
                let left = &mut self.dependency_left[successor];
                assert!(*left > 0);
                *left -= 1;
                if *left == 0 {
                    self.ready.push(successor);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Graph;

    #[test]
    fn read_write_overwrite_dependencies() {
        let mut graph = Graph::default();
        let read_first = graph.add(&[0], 1);
        let read_second = graph.add(&[0], 2);
        let overwrite = graph.add(&[], 0);
        let overwrite_again = graph.add(&[], 0);
        let update = graph.add(&[0], 0);
        let consume = graph.add(&[0], 3);

        assert_eq!(graph.dependency_counts, vec![0, 0, 2, 1, 1, 1]);
        assert_eq!(
            graph.successors[read_first].iter().copied().collect::<Vec<_>>(),
            vec![overwrite]
        );
        assert_eq!(
            graph.successors[read_second]
                .iter()
                .copied()
                .collect::<Vec<_>>(),
            vec![overwrite]
        );
        assert_eq!(
            graph.successors[overwrite]
                .iter()
                .copied()
                .collect::<Vec<_>>(),
            vec![overwrite_again]
        );
        assert_eq!(
            graph.successors[overwrite_again]
                .iter()
                .copied()
                .collect::<Vec<_>>(),
            vec![update]
        );
        assert_eq!(
            graph.successors[update].iter().copied().collect::<Vec<_>>(),
            vec![consume]
        );
    }

    #[test]
    fn start_can_be_replayed() {
        let mut graph = Graph::default();
        graph.add(&[0], 1);
        graph.add(&[0], 2);
        graph.add(&[], 0);
        graph.add(&[0], 3);

        for _ in 0..2 {
            let mut run = graph.start();
            assert_eq!(run.ready, vec![0, 1]);
            run.finish(&graph, &[1]);
            assert_eq!(run.ready, vec![0]);
            run.finish(&graph, &[0]);
            assert_eq!(run.ready, vec![2]);
            run.finish(&graph, &[2]);
            assert_eq!(run.ready, vec![3]);
            run.finish(&graph, &[3]);
            assert!(run.ready.is_empty());
        }
    }
}
