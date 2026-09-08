//! Root-only variable-byte gather for final disjoint sparse-row assembly.
use super::Comm;
use ::mpi::{datatype::PartitionMut, traits::*};

impl Comm<'_> {
    pub(crate) fn gather_bytes(&self, root: usize, bytes: &[u8]) -> Option<Vec<Vec<u8>>> {
        assert!(root < self.size());
        let is_root = self.rank() == root;
        let count: i32 = bytes.len().try_into().unwrap();
        let mut counts = vec![0i32; if is_root { self.size() } else { 0 }];
        let root_process = self.communicator().process_at_rank(root as i32);
        if is_root {
            root_process.gather_into_root(&count, &mut counts[..]);
        } else {
            root_process.gather_into(&count);
        }

        let mut total = 0i32;
        let displacements: Vec<_> = counts
            .iter()
            .map(|&count| {
                let offset = total;
                total = total.checked_add(count).unwrap();
                offset
            })
            .collect();
        let length: usize = total.try_into().unwrap();
        let mut send = bytes.to_vec();
        if send.is_empty() {
            send.push(0);
        }
        let mut received = vec![0u8; length.max(1)];
        if is_root {
            let mut received_partition =
                PartitionMut::new(&mut received[..length], &counts[..], &displacements[..]);
            root_process
                .gather_varcount_into_root(&send[..count as usize], &mut received_partition);
            Some(
                counts
                    .into_iter()
                    .zip(displacements)
                    .map(|(count, offset)| {
                        received[offset as usize..(offset + count) as usize].to_vec()
                    })
                    .collect(),
            )
        } else {
            root_process.gather_varcount_into(&send[..count as usize]);
            None
        }
    }
}
