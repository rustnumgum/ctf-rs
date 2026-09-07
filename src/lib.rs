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
pub mod sparse;
pub mod planning;
pub mod model;
pub mod multilinear;
pub mod cost;
pub mod initial_models;
pub mod selector;
pub mod plan_cost;
pub mod symmetry;
pub mod sym_indices;
pub mod sym_permutations;
pub mod folding;
#[cfg(feature = "native-scalapack")]
pub mod matrix;
mod ffi;
pub mod symmetric_distribution;
pub mod symmetric_tensor;
pub mod symmetric_sum;
pub mod symmetric_sum_comm;
pub mod symmetric_contraction;
pub mod symmetric_contraction_comm;
pub mod scalar_conversion;
pub mod sparse_sequential;
pub mod sparse_function_kernel;
mod dense_function;
mod dense_transform;
pub mod grid_plan_cost;
pub mod redist_cost;
pub mod mapping_preflight;
pub mod mapping_variants;
pub mod normal_mapping;
pub mod normal_search;
pub mod mapped_cost;
pub mod dense_search;
mod dense_execution;
pub mod fold_indices;
pub mod fold_layout;
pub mod partial_fold;
pub mod partial_fold_kernel;
pub mod folded_cost;
