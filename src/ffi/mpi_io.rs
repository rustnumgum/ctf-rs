// Adapted from cc4s CTF interface/graph_io_aux.cxx read_data_mpiio and
// write_data_mpiio. Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! Communicator-scoped MPI-IO for newline-terminated sparse text records.

use super::{Comm, check};
use ::mpi::{collective::SystemOperation, ffi as sys, traits::CommunicatorCollectives};
use std::{ffi::CString, path::Path};

const OVERLAP: sys::MPI_Offset = 300;

fn mpi_path(path: &Path) -> CString {
    CString::new(path.to_str().unwrap()).unwrap()
}

impl Comm<'_> {
    /// Collectively reads the complete newline-terminated records assigned to
    /// this rank by the source's fixed 300-byte overlap partition.
    pub(crate) fn read_sparse_text(&self, path: &Path) -> Vec<u8> {
        let path = mpi_path(path);
        // SAFETY: `self.raw()` is a live communicator handle owned by
        // `self`; `path` is a valid NUL-terminated CString kept alive for
        // the whole block; `RSMPI_FILE_NULL` is a plain handle value,
        // overwritten by MPI_File_open before use. `file_size`/`status`'s
        // zero-initialized bit patterns are valid for their all-integer
        // #[repr(C)] layouts and are fully written before being read.
        // `bytes` is allocated to exactly `requested` elements and that
        // same `requested` count (cast with `try_into`, so it cannot
        // silently truncate) is passed to MPI_File_read_at_all, so the
        // write stays in bounds; `actual`, read back via MPI_Get_count
        // only after that call returns, is asserted `>= 0` before being
        // used to truncate `bytes`. `file` is closed exactly once, on
        // every return path (empty file included).
        unsafe {
            let mut file = sys::RSMPI_FILE_NULL;
            check(sys::MPI_File_open(
                self.raw(),
                path.as_ptr(),
                sys::MPI_MODE_RDONLY as i32,
                sys::RSMPI_INFO_NULL,
                &mut file,
            ));

            let mut file_size: sys::MPI_Offset = 0;
            check(sys::MPI_File_get_size(file, &mut file_size));
            assert!(file_size >= 0);
            if file_size == 0 {
                check(sys::MPI_File_close(&mut file));
                return Vec::new();
            }

            let rank: sys::MPI_Offset = self.rank().try_into().unwrap();
            let size: sys::MPI_Offset = self.size().try_into().unwrap();
            let block = file_size / size;
            let start = rank * block;
            let owned_end = start + block;
            let read_end = if rank + 1 == size {
                file_size
            } else {
                (owned_end + OVERLAP).min(file_size)
            };
            let requested: usize = (read_end - start).try_into().unwrap();
            let mut bytes = vec![0u8; requested];
            let mut status: sys::MPI_Status = std::mem::zeroed();
            check(sys::MPI_File_read_at_all(
                file,
                start,
                bytes.as_mut_ptr().cast(),
                requested.try_into().unwrap(),
                sys::RSMPI_UINT8_T,
                &mut status,
            ));
            let mut actual = 0;
            check(sys::MPI_Get_count(&status, sys::RSMPI_UINT8_T, &mut actual));
            assert!(actual >= 0);
            bytes.truncate(actual as usize);
            check(sys::MPI_File_close(&mut file));

            let mut first = 0;
            let mut last = bytes.len();
            let mut invalid = None;
            if rank != 0 {
                match bytes.iter().position(|&byte| byte == b'\n') {
                    Some(position) => first = position + 1,
                    None => invalid = Some("sparse text record exceeds the 300-byte overlap"),
                }
            }
            if rank + 1 != size {
                let boundary: usize = block.try_into().unwrap();
                match bytes
                    .get(boundary..)
                    .and_then(|tail| tail.iter().position(|&byte| byte == b'\n'))
                {
                    Some(position) => last = boundary + position + 1,
                    None => invalid = Some("sparse text record exceeds the 300-byte overlap"),
                }
            } else if bytes.last() != Some(&b'\n') {
                invalid = Some("sparse text file must end with a newline");
            }
            if let Some(message) = invalid {
                panic!("{message}");
            }
            assert!(first <= last);
            bytes[first..last].to_vec()
        }
    }

    /// Collectively replaces `path` with the concatenation of each rank's
    /// already newline-terminated sparse text records, in communicator order.
    pub(crate) fn write_sparse_text(&self, path: &Path, bytes: &[u8]) {
        let path = mpi_path(path);
        // SAFETY: `self.raw()` is a live communicator handle owned by
        // `self`; `path` is a valid NUL-terminated CString kept alive for
        // the whole block; `RSMPI_FILE_NULL` is a plain handle value,
        // overwritten by each MPI_File_open before use, and `file` is
        // reused (reopened) only after the first MPI_File_close returns,
        // so no stale handle is ever passed to a collective call. `bytes`
        // (`bytes.len()` elements, cast with `try_into`) is read only by
        // MPI_File_write_at_all and stays valid for the whole block since
        // it is borrowed for the function's duration. `status`'s
        // zero-initialized bit pattern is valid and fully overwritten
        // before any field would be read. `offset` comes from a collective
        // `scan_into` so every rank writes a distinct, non-overlapping
        // byte range. `file` is closed exactly once on the return path.
        unsafe {
            let mut file = sys::RSMPI_FILE_NULL;
            check(sys::MPI_File_open(
                self.raw(),
                path.as_ptr(),
                (sys::MPI_MODE_WRONLY | sys::MPI_MODE_CREATE | sys::MPI_MODE_DELETE_ON_CLOSE)
                    as i32,
                sys::RSMPI_INFO_NULL,
                &mut file,
            ));
            check(sys::MPI_File_close(&mut file));
            check(sys::MPI_File_open(
                self.raw(),
                path.as_ptr(),
                (sys::MPI_MODE_WRONLY | sys::MPI_MODE_CREATE) as i32,
                sys::RSMPI_INFO_NULL,
                &mut file,
            ));

            let length: i64 = bytes.len().try_into().unwrap();
            let mut inclusive = 0i64;
            self.communicator()
                .scan_into(&length, &mut inclusive, SystemOperation::sum());
            let offset: sys::MPI_Offset = inclusive - length;
            let mut status: sys::MPI_Status = std::mem::zeroed();
            check(sys::MPI_File_write_at_all(
                file,
                offset,
                bytes.as_ptr().cast(),
                bytes.len().try_into().unwrap(),
                sys::RSMPI_UINT8_T,
                &mut status,
            ));
            check(sys::MPI_File_close(&mut file));
        }
    }
}
