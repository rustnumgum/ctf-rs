//! CPU/MPI tensor operations ported from cc4s CTF.
//!
//! Communicators are explicitly closed; tensor destruction never communicates.
//! Native MPI and numerical-library calls are confined to the internal `ffi` module.
#![deny(unsafe_op_in_unsafe_fn)]

pub mod algebra;
pub mod context;
pub mod mapping;
pub mod tensor;
pub mod linalg;
mod ffi;
