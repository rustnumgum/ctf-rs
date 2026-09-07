//! Local numerical-kernel boundary. Tensor algorithms depend on this interface,
//! not BLAS/LAPACK symbols. A future faer implementation uses the same contracts;
//! distributed ScaLAPACK operations are deliberately not part of this interface.

#[derive(Clone, Copy, Debug)]
pub enum Transpose {
    No,
    Yes,
}

pub struct Gemm<'a, T> {
    pub trans_a: Transpose,
    pub trans_b: Transpose,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub alpha: T,
    pub a: &'a [T],
    pub lda: usize,
    pub b: &'a [T],
    pub ldb: usize,
    pub beta: T,
    pub c: &'a mut [T],
    pub ldc: usize,
}

pub trait GemmKernel<T> {
    fn gemm(args: Gemm<'_, T>);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Uplo {
    Lower,
    Upper,
}

pub struct Syr<'a, T> {
    pub uplo: Uplo,
    pub n: usize,
    pub alpha: T,
    pub x: &'a [T],
    pub incx: i32,
    pub a: &'a mut [T],
    pub lda: usize,
}

/// Typed BLAS symmetric rank-one update. Complex implementations use plain
/// transpose (`csyr`/`zsyr`), never conjugating Hermitian updates.
pub trait SyrKernel<T> {
    fn syr(args: Syr<'_, T>);
}

/// A narrow compile-time interface, not a runtime backend registry. Layout is
/// column-major, and decomposition outputs retain LAPACK's mathematical contract
/// (not its workspace representation). All failures are returned as native info.
pub trait LocalKernels: GemmKernel<f64> {
    fn cholesky(n: usize, a: &mut [f64], lower: bool) -> Result<(), i32>;
    fn qr(m: usize, n: usize, a: &[f64]) -> Result<(Vec<f64>, Vec<f64>), i32>;
    fn svd(m: usize, n: usize, a: &[f64]) -> Result<Svd, i32>;
    fn eigh(n: usize, a: &[f64]) -> Result<(Vec<f64>, Vec<f64>), i32>;
    /// Tall QR reduction to (R, first n entries of Q^T b), without forming Q.
    fn qr_reduce(m: usize, n: usize, a: &[f64], b: &[f64]) -> Result<(Vec<f64>, Vec<f64>), i32>;
    /// Tall rank-revealing minimum-norm least squares (LAPACK machine threshold).
    fn least_squares(m: usize, n: usize, a: &[f64], b: &[f64]) -> Result<Vec<f64>, i32>;
}

pub struct Svd {
    pub u: Vec<f64>,
    pub values: Vec<f64>,
    pub vt: Vec<f64>,
}

#[cfg(feature = "native-linalg")]
pub struct Native;
#[cfg(feature = "native-linalg")]
impl GemmKernel<f32> for Native {
    fn gemm(args: Gemm<'_, f32>) {
        crate::ffi::linalg::gemm_f32(args);
    }
}
#[cfg(feature = "native-linalg")]
impl GemmKernel<f64> for Native {
    fn gemm(args: Gemm<'_, f64>) {
        crate::ffi::linalg::gemm(args);
    }
}
#[cfg(feature = "native-linalg")]
impl GemmKernel<crate::algebra::Complex<f32>> for Native {
    fn gemm(args: Gemm<'_, crate::algebra::Complex<f32>>) {
        crate::ffi::linalg::gemm_c32(args);
    }
}
#[cfg(feature = "native-linalg")]
impl GemmKernel<crate::algebra::Complex<f64>> for Native {
    fn gemm(args: Gemm<'_, crate::algebra::Complex<f64>>) {
        crate::ffi::linalg::gemm_c64(args);
    }
}
#[cfg(feature = "native-linalg")]
impl SyrKernel<f32> for Native {
    fn syr(args: Syr<'_, f32>) {
        crate::ffi::linalg::syr_f32(args);
    }
}
#[cfg(feature = "native-linalg")]
impl SyrKernel<f64> for Native {
    fn syr(args: Syr<'_, f64>) {
        crate::ffi::linalg::syr(args);
    }
}
#[cfg(feature = "native-linalg")]
impl SyrKernel<crate::algebra::Complex<f32>> for Native {
    fn syr(args: Syr<'_, crate::algebra::Complex<f32>>) {
        crate::ffi::linalg::syr_c32(args);
    }
}
#[cfg(feature = "native-linalg")]
impl SyrKernel<crate::algebra::Complex<f64>> for Native {
    fn syr(args: Syr<'_, crate::algebra::Complex<f64>>) {
        crate::ffi::linalg::syr_c64(args);
    }
}
#[cfg(feature = "native-linalg")]
impl LocalKernels for Native {
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
    fn qr_reduce(m: usize, n: usize, a: &[f64], b: &[f64]) -> Result<(Vec<f64>, Vec<f64>), i32> {
        crate::ffi::linalg::qr_reduce(m, n, a, b)
    }
    fn least_squares(m: usize, n: usize, a: &[f64], b: &[f64]) -> Result<Vec<f64>, i32> {
        crate::ffi::linalg::least_squares(m, n, a, b)
    }
}
