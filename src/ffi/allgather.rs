// Adapted from cc4s CTF tensor/untyped_tensor.cxx all-rank extraction.
// Upstream provenance and retained license: docs/provenance.md and LICENSE.
//! Variable-length byte all-gather on an existing communicator.

use super::Comm;
use ::mpi::{datatype::PartitionMut, traits::CommunicatorCollectives};

impl Comm<'_> {
    pub(crate) fn all_gather_bytes(&self, bytes: &[u8]) -> Vec<u8> {
        let count: i32 = bytes.len().try_into().unwrap();
        let mut counts = vec![0i32; self.size()];
        self.communicator().all_gather_into(&count, &mut counts[..]);

        let mut total = 0i32;
        let displacements: Vec<i32> = counts
            .iter()
            .map(|&count| {
                assert!(count >= 0);
                let displacement = total;
                total = total.checked_add(count).unwrap();
                displacement
            })
            .collect();
        let actual_total: usize = total.try_into().unwrap();
        // Keep distinct valid allocations when either side has a zero count;
        // some MPI implementations reject aliased/dangling empty-Vec pointers.
        let mut send = bytes.to_vec();
        if send.is_empty() {
            send.push(0);
        }
        let mut received = vec![0u8; actual_total.max(1)];
        let mut received_partition = PartitionMut::new(
            &mut received[..actual_total],
            &counts[..],
            &displacements[..],
        );
        self.communicator()
            .all_gather_varcount_into(&send[..count as usize], &mut received_partition);
        received.truncate(actual_total);
        received
    }
}
