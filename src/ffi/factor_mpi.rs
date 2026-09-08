//! Block collectives used by the distributed row-system solver.
use super::Comm;
use ::mpi::{collective::SystemOperation, traits::*};

impl Comm<'_> {
    pub(crate) fn reduce_scatter_f64(&self, input: &[f64], count: usize) -> Vec<f64> {
        assert_eq!(input.len(), count * self.size());
        let mut input_storage = input.to_vec();
        if input_storage.is_empty() {
            input_storage.push(0.0);
        }
        let mut output = vec![0.0; count.max(1)];
        self.communicator().reduce_scatter_block_into(
            &input_storage[..input.len()],
            &mut output[..count],
            SystemOperation::sum(),
        );
        output.truncate(count);
        output
    }

    pub(crate) fn scatter_f64(&self, root: usize, input: &[f64], count: usize) -> Vec<f64> {
        assert!(root < self.size());
        if self.rank() == root {
            assert_eq!(input.len(), count * self.size());
        }
        let mut output = vec![0.0; count.max(1)];
        let root_process = self.communicator().process_at_rank(root as i32);
        if self.rank() == root {
            root_process.scatter_into_root(input, &mut output[..count]);
        } else {
            root_process.scatter_into(&mut output[..count]);
        }
        output.truncate(count);
        output
    }

    pub(crate) fn gather_f64(&self, root: usize, input: &[f64]) -> Vec<f64> {
        assert!(root < self.size());
        let length = if self.rank() == root {
            input.len() * self.size()
        } else {
            0
        };
        let mut output = vec![0.0; length.max(1)];
        let root_process = self.communicator().process_at_rank(root as i32);
        if self.rank() == root {
            root_process.gather_into_root(input, &mut output[..length]);
        } else {
            root_process.gather_into(input);
        }
        output.truncate(length);
        output
    }
}
