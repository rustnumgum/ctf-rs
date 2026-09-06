//! Local numerical-kernel boundary. Tensor algorithms depend on this interface,
//! not BLAS/LAPACK symbols. A future faer implementation uses the same contracts;
//! distributed ScaLAPACK operations are deliberately not part of this interface.

#[derive(Clone, Copy, Debug)]
pub enum Transpose { No, Yes }

pub struct Gemm<'a> {
    pub trans_a: Transpose,
    pub trans_b: Transpose,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub alpha: f64,
    pub a: &'a [f64],
    pub lda: usize,
    pub b: &'a [f64],
    pub ldb: usize,
    pub beta: f64,
    pub c: &'a mut [f64],
    pub ldc: usize,
}

/// A narrow compile-time interface, not a runtime backend registry. Layout is
/// column-major, and decomposition outputs retain LAPACK's mathematical contract
/// (not its workspace representation). All failures are returned as native info.
pub trait LocalKernels {
    fn gemm(args: Gemm<'_>);
    fn cholesky(n: usize, a: &mut [f64], lower: bool) -> Result<(), i32>;
    fn qr(m: usize, n: usize, a: &[f64]) -> Result<(Vec<f64>, Vec<f64>), i32>;
    fn svd(m: usize, n: usize, a: &[f64]) -> Result<Svd, i32>;
    fn eigh(n: usize, a: &[f64]) -> Result<(Vec<f64>, Vec<f64>), i32>;
    /// Tall QR reduction to (R, first n entries of Q^T b), without forming Q.
    fn qr_reduce(m: usize,n: usize,a: &[f64],b: &[f64]) -> Result<(Vec<f64>,Vec<f64>),i32>;
    /// Tall rank-revealing minimum-norm least squares (LAPACK machine threshold).
    fn least_squares(m: usize,n: usize,a: &[f64],b: &[f64]) -> Result<Vec<f64>,i32>;
}

pub struct Svd { pub u: Vec<f64>, pub values: Vec<f64>, pub vt: Vec<f64> }

#[cfg(feature = "native-linalg")]
pub struct Native;
#[cfg(feature = "native-linalg")]
impl LocalKernels for Native {
    fn gemm(args: Gemm<'_>) { crate::ffi::linalg::gemm(args); }
    fn cholesky(n: usize, a: &mut [f64], lower: bool) -> Result<(), i32> {
        crate::ffi::linalg::cholesky(n, a, lower)
    }
    fn qr(m: usize, n: usize, a: &[f64]) -> Result<(Vec<f64>, Vec<f64>), i32> {
        crate::ffi::linalg::qr(m, n, a)
    }
    fn svd(m: usize, n: usize, a: &[f64]) -> Result<Svd, i32> {
        crate::ffi::linalg::svd(m, n, a)
    }
    fn eigh(n: usize, a: &[f64]) -> Result<(Vec<f64>, Vec<f64>), i32> {
        crate::ffi::linalg::eigh(n, a)
    }
    fn qr_reduce(m:usize,n:usize,a:&[f64],b:&[f64])->Result<(Vec<f64>,Vec<f64>),i32> {
        crate::ffi::linalg::qr_reduce(m,n,a,b)
    }
    fn least_squares(m:usize,n:usize,a:&[f64],b:&[f64])->Result<Vec<f64>,i32> {
        crate::ffi::linalg::least_squares(m,n,a,b)
    }
}
