// Adapted from cc4s CTF contraction/spctr_tsr.cxx::seq_tsr_spctr at
// f69cbb46e23bc2f39cda5722ce096f56301dab4f.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! CPU sparse leaf estimates. Source heuristic traffic is not allocation size.
use crate::cost::Models;

#[derive(Clone, Copy, Debug)]
pub enum Folded {
    CooDense,
    CsrDense,
    CsrSparseDense,
    CsrSparse,
    CcsrDense,
}
#[derive(Clone, Debug)]
pub enum Kernel {
    /// Extent of each distinct union label, in source inverse-index order.
    General { extents: Vec<usize> },
    Folded { kind: Folded, m: usize, n: usize, k: usize },
}
#[derive(Clone, Debug)]
pub struct Local {
    pub kernel: Kernel,
    pub custom: bool,
    /// sy_packed_size for each operand. Folded axes have already been replaced
    /// by unit extents, as in the source leaf constructor.
    pub packed_elements: [usize; 3],
    pub element_bytes: [usize; 3],
    pub sparse: [bool; 3],
    pub nnz_fraction: [f64; 3],
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Estimate {
    pub flops: f64,
    pub memory_traffic_bytes: usize,
    pub seconds: f64,
}
impl Local {
    /// Source CPU k0..k5 models, without excluded CUDA/offload models. The
    /// source leaf has zero extra working-memory footprint; traffic below is
    /// instead the model's sparse 10x/30x access heuristic.
    pub fn estimate(&self, models: &Models) -> Estimate {
        let (id, mut flops, multipliers) = match &self.kernel {
            Kernel::General { extents } => {
                let mut flops = 2.0;
                for &extent in extents { flops *= extent as f64; }
                (0, flops, [1; 3])
            }
            Kernel::Folded { kind, m, n, k } => {
                let id = match kind {
                    Folded::CooDense => 1, Folded::CsrDense => 2,
                    Folded::CsrSparseDense => 3, Folded::CsrSparse => 4,
                    Folded::CcsrDense => 5,
                };
                (id, 2.0 * *m as f64 * *n as f64 * *k as f64,
                    [m*k, n*k, m*n])
            }
        };
        for operand in 0..2 {
            if self.sparse[operand] { flops *= self.nnz_fraction[operand] * 3.0; }
        }
        assert!(flops >= 0.0);
        let mut memory_traffic_bytes = 0;
        for operand in 0..3 {
            let mut bytes = self.packed_elements[operand] * self.element_bytes[operand] * multipliers[operand];
            if self.sparse[operand] {
                // The source uint64 *= double truncates each operand before
                // the three traffic terms are added.
                bytes = (bytes as f64 * (self.nnz_fraction[operand] *
                    if operand == 2 { 30.0 } else { 10.0 })) as usize;
            }
            memory_traffic_bytes += bytes;
        }
        let name = if self.custom { format!("seq_tsr_spctr_cst_k{id}") }
            else { format!("seq_tsr_spctr_k{id}") };
        let seconds = models.get(&name).estimate(&[1.0, memory_traffic_bytes as f64, flops]);
        Estimate { flops, memory_traffic_bytes, seconds }
    }
}
