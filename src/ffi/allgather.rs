// Adapted from cc4s CTF tensor/untyped_tensor.cxx all-rank extraction.
// Upstream provenance and retained license: docs/provenance.md and LICENSE.
//! Variable-length byte all-gather on an existing communicator.

use super::{check, Comm};
use mpi_sys as sys;

impl Comm {
    pub(crate) fn all_gather_bytes(&self, bytes: &[u8]) -> Vec<u8> {
        let count: i32 = bytes.len().try_into().unwrap();
        let mut counts = vec![0i32; self.size()];
        unsafe {
            check(sys::MPI_Allgather(
                (&count as *const i32).cast(),
                1,
                sys::RSMPI_INT32_T,
                counts.as_mut_ptr().cast(),
                1,
                sys::RSMPI_INT32_T,
                self.raw,
            ));
        }

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
        unsafe {
            check(sys::MPI_Allgatherv(
                send.as_ptr().cast(),
                count,
                sys::RSMPI_UINT8_T,
                received.as_mut_ptr().cast(),
                counts.as_ptr(),
                displacements.as_ptr(),
                sys::RSMPI_UINT8_T,
                self.raw,
            ));
        }
        received.truncate(actual_total);
        received
    }
}
