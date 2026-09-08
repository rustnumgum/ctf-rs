//! Communicator-scoped independent-offset binary MPI-IO. The operation order
//! follows CTF tensor/untyped_tensor.cxx's binary read/write path.

use super::{Comm, check};
use ::mpi::ffi as sys;
use std::{ffi::CString, path::Path};

fn mpi_path(path: &Path) -> CString {
    CString::new(path.to_str().unwrap()).unwrap()
}

impl Comm<'_> {
    /// Collectively opens `path` without truncation, independently writes this
    /// rank's byte range at `offset`, then collectively closes the file.
    pub(crate) fn write_binary_at(&self, path: &Path, offset: u64, bytes: &[u8]) {
        let path = mpi_path(path);
        let offset: sys::MPI_Offset = offset.try_into().unwrap();
        let count: i32 = bytes.len().try_into().unwrap();
        let empty = [0u8];
        let storage = if bytes.is_empty() { &empty[..] } else { bytes };
        unsafe {
            let mut file = sys::RSMPI_FILE_NULL;
            check(sys::MPI_File_open(
                self.raw(),
                path.as_ptr(),
                (sys::MPI_MODE_WRONLY | sys::MPI_MODE_CREATE) as i32,
                sys::RSMPI_INFO_NULL,
                &mut file,
            ));
            let mut status: sys::MPI_Status = std::mem::zeroed();
            check(sys::MPI_File_write_at(
                file,
                offset,
                storage.as_ptr().cast(),
                count,
                sys::RSMPI_UINT8_T,
                &mut status,
            ));
            check(sys::MPI_File_close(&mut file));
        }
    }

    /// Collectively opens `path`, independently reads exactly `length` bytes
    /// at `offset`, then collectively closes the file.
    pub(crate) fn read_binary_at(&self, path: &Path, offset: u64, length: usize) -> Vec<u8> {
        let path = mpi_path(path);
        let offset: sys::MPI_Offset = offset.try_into().unwrap();
        let count: i32 = length.try_into().unwrap();
        let mut bytes = vec![0u8; length.max(1)];
        let actual;
        unsafe {
            let mut file = sys::RSMPI_FILE_NULL;
            check(sys::MPI_File_open(
                self.raw(),
                path.as_ptr(),
                sys::MPI_MODE_RDONLY as i32,
                sys::RSMPI_INFO_NULL,
                &mut file,
            ));
            let mut status: sys::MPI_Status = std::mem::zeroed();
            check(sys::MPI_File_read_at(
                file,
                offset,
                bytes.as_mut_ptr().cast(),
                count,
                sys::RSMPI_UINT8_T,
                &mut status,
            ));
            let mut read = 0;
            check(sys::MPI_Get_count(&status, sys::RSMPI_UINT8_T, &mut read));
            actual = read;
            check(sys::MPI_File_close(&mut file));
        }
        assert_eq!(
            actual, count,
            "binary MPI-IO read ended before requested length"
        );
        bytes.truncate(length);
        bytes
    }
}
