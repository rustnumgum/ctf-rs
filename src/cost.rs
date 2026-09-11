// Adapted from cc4s interface/common.cxx, contraction/ctr_tsr.cxx and
// redistribution/nosym_transp.cxx. Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! Named CPU model bank and source cost formulas. Estimates are seconds. Loading
//! or fitting coefficients is explicit; estimates never time operations themselves.
use crate::{
    context::Context,
    model::{LinearModel, source_coefficient},
};
use std::{
    collections::{HashMap, HashSet},
    fs::{self, File, OpenOptions},
    io::{self, BufRead, BufReader, BufWriter, Write},
    path::Path,
};

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
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut LinearModel> {
        self.models.iter_mut()
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
        let mut records = HashMap::new();
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
                write!(writer, " {}", source_coefficient(*value))?;
            }
            writeln!(writer)?;
        }
        Ok(())
    }
    /// Load every registered CPU model from an explicit source-format file.
    pub fn load_all_models(&mut self, file_name: impl AsRef<Path>) -> io::Result<()> {
        self.load(BufReader::new(File::open(file_name)?))
    }
    /// Replace registered records and append missing ones while retaining records
    /// for models outside this CPU registry, as the source writer does.
    pub fn write_all_models(&self, file_name: impl AsRef<Path>) -> io::Result<()> {
        let file_name = file_name.as_ref();
        let existing = match fs::read_to_string(file_name) {
            Ok(contents) => contents,
            Err(error) if error.kind() == io::ErrorKind::NotFound => String::new(),
            Err(error) => return Err(error),
        };
        let mut replacements = HashMap::new();
        for model in &self.models {
            let mut record = model.name().to_owned();
            for value in model.coefficients() {
                record.push(' ');
                record.push_str(&source_coefficient(*value));
            }
            replacements.insert(model.name(), record);
        }
        let mut found = HashSet::new();
        let mut output = BufWriter::new(File::create(file_name)?);
        for line in existing.lines() {
            let name = line.split_once(' ').map_or(line, |(name, _)| name);
            if let Some(record) = replacements.get(name) {
                writeln!(output, "{record}")?;
                found.insert(name);
            } else {
                writeln!(output, "{line}")?;
            }
        }
        for model in &self.models {
            if !found.contains(model.name()) {
                writeln!(output, "{}", replacements[model.name()])?;
            }
        }
        output.flush()
    }
    /// Append coefficients once and retained observations in rank order to one
    /// file per registered model. The path must already exist.
    pub fn dump_all_models(&self, context: &Context<'_>, path: impl AsRef<Path>) -> io::Result<()> {
        let path = path.as_ref();
        let mut result = Ok(());
        for rank in 0..context.size() {
            if rank == context.rank() && result.is_ok() {
                result = self.dump_rank(path, rank == 0);
            }
            context.barrier();
        }
        result
    }
    fn dump_rank(&self, path: &Path, include_coefficients: bool) -> io::Result<()> {
        for model in &self.models {
            let mut output = BufWriter::new(
                OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path.join(model.name()))?,
            );
            if include_coefficients {
                for value in model.coefficients() {
                    write!(output, "{value} ")?;
                }
                writeln!(output)?;
            }
            for (seconds, parameters) in model.retained_observations() {
                write!(output, "{seconds} ")?;
                for value in parameters {
                    write!(output, "{value} ")?;
                }
                writeln!(output)?;
            }
            output.flush()?;
        }
        Ok(())
    }
    /// Print source declarations followed by diagnostics for observed models.
    pub fn print_all_models(&self, mut output: impl Write) -> io::Result<()> {
        for model in &self.models {
            write!(output, "double {}_init[] = {{", model.name())?;
            for (index, value) in model.coefficients().iter().enumerate() {
                if index > 0 {
                    write!(output, ", ")?;
                }
                write!(output, "{}", source_coefficient(*value))?;
            }
            writeln!(output, "}};")?;
        }
        for model in &self.models {
            let diagnostics = model.diagnostics();
            if diagnostics.observations > 0 {
                writeln!(
                    output,
                    "{} is_tuned = {} is_active = 1 ({}) avg_tot_time = {:.6} avg_over_time = {:.6} avg_under_time = {:.6}",
                    model.name(),
                    usize::from(diagnostics.tuned),
                    diagnostics.observations,
                    diagnostics.average_total_time,
                    diagnostics.average_over_time,
                    diagnostics.average_under_time,
                )?;
            }
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
