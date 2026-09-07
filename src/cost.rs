// Adapted from cc4s interface/common.cxx, contraction/ctr_tsr.cxx and
// redistribution/nosym_transp.cxx. Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! Named CPU model bank and source cost formulas. Estimates are seconds. Loading
//! or fitting coefficients is explicit; estimates never time operations themselves.
use crate::model::LinearModel;
use std::io::{self, BufRead, Write};

pub struct Models {
    models: Vec<LinearModel>,
}
#[derive(Clone, Copy)]
pub enum Communication {
    Broadcast,
    Reduce { custom: bool },
    AllReduce { custom: bool },
    AllToAll,
    AllToAllV,
}
impl Models {
    pub fn upstream(history_size: usize) -> Self {
        Self {
            models: crate::initial_models::CPU
                .iter()
                .map(|(name, coeff)| LinearModel::new(*name, coeff.to_vec(), history_size))
                .collect(),
        }
    }
    pub fn iter(&self) -> impl Iterator<Item = &LinearModel> {
        self.models.iter()
    }
    pub fn get(&self, name: &str) -> &LinearModel {
        self.models
            .iter()
            .find(|m| m.name() == name)
            .expect("unknown CPU model")
    }
    pub fn get_mut(&mut self, name: &str) -> &mut LinearModel {
        self.models
            .iter_mut()
            .find(|m| m.name() == name)
            .expect("unknown CPU model")
    }
    /// Source named whitespace-separated coefficient records. Unknown names
    /// (e.g. excluded CUDA models) are ignored; missing/malformed CPU records
    /// return errors rather than substituting zero or silently retaining seeds.
    pub fn load(&mut self, reader: impl BufRead) -> io::Result<()> {
        let mut records = std::collections::HashMap::new();
        for line in reader.lines() {
            let line = line?;
            let mut fields = line.split_whitespace();
            if let Some(name) = fields.next() {
                records
                    .entry(name.to_owned())
                    .or_insert_with(|| fields.map(str::to_owned).collect::<Vec<_>>());
            }
        }
        let mut coefficients = Vec::new();
        for model in &self.models {
            let values = records.get(model.name()).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("missing model {}", model.name()),
                )
            })?;
            if values.len() != model.coefficients().len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "coefficient count mismatch",
                ));
            }
            coefficients.push(
                values
                    .iter()
                    .map(|s| {
                        s.parse::<f64>()
                            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
                    })
                    .collect::<io::Result<Vec<_>>>()?,
            );
        }
        for (model, values) in self.models.iter_mut().zip(coefficients) {
            model.set_coefficients(&values);
        }
        Ok(())
    }
    /// Source coefficient output precision: scientific with four fractional digits.
    /// File opening and root selection belong to the caller, not a global world.
    pub fn write(&self, mut writer: impl Write) -> io::Result<()> {
        for model in &self.models {
            write!(writer, "{}", model.name())?;
            for value in model.coefficients() {
                write!(writer, " {value:.4e}")?;
            }
            writeln!(writer)?;
        }
        Ok(())
    }
    /// msg_size means bytes, except AllToAll where it is bytes per peer chunk.
    pub fn communication(&self, operation: Communication, ranks: usize, msg_size: usize) -> f64 {
        assert!(ranks > 0);
        let log = (ranks as f64).log2();
        let bytes = msg_size as f64;
        let (name, volume) = match operation {
            Communication::Broadcast => ("bcast_mdl", bytes),
            Communication::Reduce { custom: false } => ("red_mdl", bytes * log),
            Communication::Reduce { custom: true } => ("red_mdl_cst", bytes * log),
            Communication::AllReduce { custom: false } => ("allred_mdl", bytes * log),
            Communication::AllReduce { custom: true } => ("allred_mdl_cst", bytes * log),
            Communication::AllToAll => ("alltoall_mdl", log * ranks as f64 * bytes),
            Communication::AllToAllV => ("alltoallv_mdl", log * bytes),
        };
        self.get(name).estimate(&[1., log, volume])
    }
    pub fn local_contraction(
        &self,
        custom: bool,
        folded: bool,
        memory_traffic: f64,
        flops: f64,
    ) -> f64 {
        let name = match (custom, folded) {
            (true, false) => "seq_tsr_ctr_mdl_cst",
            (true, true) => "seq_tsr_ctr_mdl_cst_inr",
            (false, false) => "seq_tsr_ctr_mdl_ref",
            (false, true) => "seq_tsr_ctr_mdl_inr",
        };
        self.get(name).estimate(&[1., memory_traffic, flops])
    }
    pub fn transpose(&self, shape: &[usize], permutation: &[usize]) -> f64 {
        assert_eq!(shape.len(), permutation.len());
        let mut sorted = permutation.to_vec();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..shape.len()).collect::<Vec<_>>());
        if shape.is_empty() {
            return 0.;
        }
        let contiguous: usize = shape
            .iter()
            .enumerate()
            .take_while(|(i, _)| permutation[*i] == *i)
            .map(|(_, n)| n)
            .product();
        let total: usize = shape.iter().product();
        if contiguous == total {
            return 0.;
        }
        let name = if contiguous < 4 {
            "non_contig_transp_mdl"
        } else if contiguous <= 64 {
            "shrt_contig_transp_mdl"
        } else {
            "long_contig_transp_mdl"
        };
        self.get(name).estimate(&[1., total as f64])
    }
}
