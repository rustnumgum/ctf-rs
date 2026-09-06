//! Block collectives used by the distributed row-system solver.
use super::{Comm, check, sys};

impl Comm {
    pub(crate) fn reduce_scatter_f64(&self, input: &[f64], count: usize) -> Vec<f64> {
        assert_eq!(input.len(), count * self.size());
        let counts = vec![i32::try_from(count).unwrap(); self.size()];
        let mut output = vec![0.; count.max(1)];
        unsafe {
            check(sys::MPI_Reduce_scatter(input.as_ptr().cast(), output.as_mut_ptr().cast(),
                counts.as_ptr(), sys::RSMPI_DOUBLE, sys::RSMPI_SUM, self.raw));
        }
        output.truncate(count);
        output
    }

    pub(crate) fn scatter_f64(&self, root: usize, input: &[f64], count: usize) -> Vec<f64> {
        if self.rank() == root { assert_eq!(input.len(), count * self.size()); }
        let mut output = vec![0.; count.max(1)];
        let count_i32 = i32::try_from(count).unwrap();
        unsafe {
            check(sys::MPI_Scatter(input.as_ptr().cast(), count_i32, sys::RSMPI_DOUBLE,
                output.as_mut_ptr().cast(), count_i32, sys::RSMPI_DOUBLE,
                i32::try_from(root).unwrap(), self.raw));
        }
        output.truncate(count);
        output
    }

    pub(crate) fn gather_f64(&self, root: usize, input: &[f64]) -> Vec<f64> {
        let count = i32::try_from(input.len()).unwrap();
        let length = if self.rank() == root { input.len() * self.size() } else { 0 };
        let mut output = vec![0.; length.max(1)];
        unsafe {
            check(sys::MPI_Gather(input.as_ptr().cast(), count, sys::RSMPI_DOUBLE,
                output.as_mut_ptr().cast(), count, sys::RSMPI_DOUBLE,
                i32::try_from(root).unwrap(), self.raw));
        }
        output.truncate(length);
        output
    }
}
