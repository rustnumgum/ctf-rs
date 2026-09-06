//! CPU/MPI tensor operations ported from cc4s CTF.
//!
//! Communicators are explicitly closed; tensor destruction never communicates.
//! Native MPI and numerical-library calls are confined to the internal `ffi` module.
#![deny(unsafe_op_in_unsafe_fn)]

pub mod algebra;
pub mod context;
pub mod mapping;
pub mod map_tensor;
pub mod topology_candidates;
pub mod node_aware;
pub mod tensor;
mod diagonal;
pub mod summation;
pub mod contraction;
pub mod ctr_2d;
pub mod linalg;
pub mod sparse_formats;
pub mod planning;
pub mod model;
pub mod cost;
pub mod initial_models;
pub mod selector;
pub mod plan_cost;
pub mod symmetry;
pub mod sym_indices;
mod ffi;
