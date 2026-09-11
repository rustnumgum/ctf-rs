// Independently written Rust bindings to the standard BLAS/LAPACK interfaces.
//! Native BLAS/LAPACK LP64 entry points; no CTF compatibility shim.
use crate::{
    algebra::Complex,
    linalg::{Gemm, Syr, Svd, Transpose, Uplo},
};
use std::ffi::c_char;

// SAFETY: these signatures mirror the reference LAPACK/BLAS Fortran ABI
// (trailing underscore, every scalar and array passed by pointer, column
// major layout, no varargs); each call site below argues pointer validity
// and buffer length against the caller's asserted matrix shape.
#[cfg_attr(any(target_os = "windows", target_os = "macos"), link(name = "openblas"))]
#[cfg_attr(not(any(target_os = "windows", target_os = "macos")), link(name = "blas"))]
unsafe extern "C" {
    fn dormqr_(
        side: *const c_char,
        trans: *const c_char,
        m: *const i32,
        n: *const i32,
        k: *const i32,
        a: *const f64,
        lda: *const i32,
        tau: *const f64,
        c: *mut f64,
        ldc: *const i32,
        work: *mut f64,
        lwork: *const i32,
        info: *mut i32,
    );
    fn dgelsd_(
        m: *const i32,
        n: *const i32,
        nrhs: *const i32,
        a: *mut f64,
        lda: *const i32,
        b: *mut f64,
        ldb: *const i32,
        s: *mut f64,
        rcond: *const f64,
        rank: *mut i32,
        work: *mut f64,
        lwork: *const i32,
        iwork: *mut i32,
        info: *mut i32,
    );
    fn dgemm_(
        ta: *const c_char,
        tb: *const c_char,
        m: *const i32,
        n: *const i32,
        k: *const i32,
        alpha: *const f64,
        a: *const f64,
        lda: *const i32,
        b: *const f64,
        ldb: *const i32,
        beta: *const f64,
        c: *mut f64,
        ldc: *const i32,
    );
    fn sgemm_(
        ta: *const c_char,
        tb: *const c_char,
        m: *const i32,
        n: *const i32,
        k: *const i32,
        alpha: *const f32,
        a: *const f32,
        lda: *const i32,
        b: *const f32,
        ldb: *const i32,
        beta: *const f32,
        c: *mut f32,
        ldc: *const i32,
    );
    fn cgemm_(
        ta: *const c_char,
        tb: *const c_char,
        m: *const i32,
        n: *const i32,
        k: *const i32,
        alpha: *const Complex<f32>,
        a: *const Complex<f32>,
        lda: *const i32,
        b: *const Complex<f32>,
        ldb: *const i32,
        beta: *const Complex<f32>,
        c: *mut Complex<f32>,
        ldc: *const i32,
    );
    fn zgemm_(
        ta: *const c_char,
        tb: *const c_char,
        m: *const i32,
        n: *const i32,
        k: *const i32,
        alpha: *const Complex<f64>,
        a: *const Complex<f64>,
        lda: *const i32,
        b: *const Complex<f64>,
        ldb: *const i32,
        beta: *const Complex<f64>,
        c: *mut Complex<f64>,
        ldc: *const i32,
    );
    fn ssyr_(
        uplo: *const c_char,
        n: *const i32,
        alpha: *const f32,
        x: *const f32,
        incx: *const i32,
        a: *mut f32,
        lda: *const i32,
    );
    fn dsyr_(
        uplo: *const c_char,
        n: *const i32,
        alpha: *const f64,
        x: *const f64,
        incx: *const i32,
        a: *mut f64,
        lda: *const i32,
    );
    fn csyr_(
        uplo: *const c_char,
        n: *const i32,
        alpha: *const Complex<f32>,
        x: *const Complex<f32>,
        incx: *const i32,
        a: *mut Complex<f32>,
        lda: *const i32,
    );
    fn zsyr_(
        uplo: *const c_char,
        n: *const i32,
        alpha: *const Complex<f64>,
        x: *const Complex<f64>,
        incx: *const i32,
        a: *mut Complex<f64>,
        lda: *const i32,
    );
}
// SAFETY: same Fortran-ABI argument as the BLAS block above; call sites
// below argue buffer length and leading dimension per call.
#[cfg_attr(any(target_os = "windows", target_os = "macos"), link(name = "openblas"))]
#[cfg_attr(not(any(target_os = "windows", target_os = "macos")), link(name = "lapack"))]
unsafe extern "C" {
    fn dpotrf_(uplo: *const c_char, n: *const i32, a: *mut f64, lda: *const i32, info: *mut i32);
    fn dgeqrf_(
        m: *const i32,
        n: *const i32,
        a: *mut f64,
        lda: *const i32,
        tau: *mut f64,
        work: *mut f64,
        lwork: *const i32,
        info: *mut i32,
    );
    fn dorgqr_(
        m: *const i32,
        n: *const i32,
        k: *const i32,
        a: *mut f64,
        lda: *const i32,
        tau: *const f64,
        work: *mut f64,
        lwork: *const i32,
        info: *mut i32,
    );
    fn dgesvd_(
        ju: *const c_char,
        jvt: *const c_char,
        m: *const i32,
        n: *const i32,
        a: *mut f64,
        lda: *const i32,
        s: *mut f64,
        u: *mut f64,
        ldu: *const i32,
        vt: *mut f64,
        ldvt: *const i32,
        work: *mut f64,
        lwork: *const i32,
        info: *mut i32,
    );
    fn dsyev_(
        job: *const c_char,
        uplo: *const c_char,
        n: *const i32,
        a: *mut f64,
        lda: *const i32,
        values: *mut f64,
        work: *mut f64,
        lwork: *const i32,
        info: *mut i32,
    );
}
fn int(n: usize) -> i32 {
    n.try_into().unwrap()
}
fn gemm_flops(m: usize, n: usize, k: usize) -> i64 {
    2 * i64::try_from(m).unwrap() * i64::try_from(n).unwrap() * i64::try_from(k).unwrap()
}
pub(crate) fn qr_reduce(
    m: usize,
    n: usize,
    a: &[f64],
    b: &[f64],
) -> Result<(Vec<f64>, Vec<f64>), i32> {
    assert!(m >= n && n > 0);
    assert_eq!(a.len(), m * n);
    assert_eq!(b.len(), m);
    let (mi, ni) = (int(m), int(n));
    let mut a = a.to_vec();
    let mut b = b.to_vec();
    let mut tau = vec![0.; n];
    let mut query = [0.];
    let mut info = 0;
    // SAFETY: `a` has exactly `m * n` elements (asserted above) matching
    // leading dimension `mi = m` and `ni = n` columns; `tau` has `n`
    // elements as dgeqrf_ requires. `lwork = -1` is LAPACK's workspace
    // query mode: it only writes the optimal size into `query` (length 1)
    // and does not otherwise read or write `work`. `info` is a valid,
    // uniquely-owned out-pointer.
    unsafe {
        dgeqrf_(
            &mi,
            &ni,
            a.as_mut_ptr(),
            &mi,
            tau.as_mut_ptr(),
            query.as_mut_ptr(),
            &-1,
            &mut info,
        );
    }
    result(info)?;
    let mut work = vec![0.; query[0] as usize];
    // SAFETY: same `a`/`tau`/leading-dimension argument as the query call
    // above; `work` was just allocated with `query[0]` elements, the size
    // dgeqrf_ itself reported as sufficient for `lwork`.
    unsafe {
        dgeqrf_(
            &mi,
            &ni,
            a.as_mut_ptr(),
            &mi,
            tau.as_mut_ptr(),
            work.as_mut_ptr(),
            &int(work.len()),
            &mut info,
        );
    }
    result(info)?;
    let mut r = vec![0.; n * n];
    for j in 0..n {
        for i in 0..=j {
            r[i + j * n] = a[i + j * m];
        }
    }
    // SAFETY: `a`/`tau` describe the Householder reflectors dgeqrf_ just
    // produced above, with the same leading dimension; `b` has `m`
    // elements (asserted above), matching the single right-hand-side
    // column (`&1`) with leading dimension `mi`. `lwork = -1` is again a
    // workspace query that only writes into `query` (length 1).
    unsafe {
        dormqr_(
            &(b'L' as c_char),
            &(b'T' as c_char),
            &mi,
            &1,
            &ni,
            a.as_ptr(),
            &mi,
            tau.as_ptr(),
            b.as_mut_ptr(),
            &mi,
            query.as_mut_ptr(),
            &-1,
            &mut info,
        );
    }
    result(info)?;
    work.resize(query[0] as usize, 0.);
    // SAFETY: same `a`/`tau`/`b` argument as the query call above; `work`
    // was just resized to `query[0]` elements, the size dormqr_ reported
    // as sufficient for `lwork`.
    unsafe {
        dormqr_(
            &(b'L' as c_char),
            &(b'T' as c_char),
            &mi,
            &1,
            &ni,
            a.as_ptr(),
            &mi,
            tau.as_ptr(),
            b.as_mut_ptr(),
            &mi,
            work.as_mut_ptr(),
            &int(work.len()),
            &mut info,
        );
    }
    result(info)?;
    b.truncate(n);
    Ok((r, b))
}
pub(crate) fn least_squares(m: usize, n: usize, a: &[f64], b: &[f64]) -> Result<Vec<f64>, i32> {
    assert!(m >= n && n > 0);
    assert_eq!(a.len(), m * n);
    assert_eq!(b.len(), m);
    let (mi, ni) = (int(m), int(n));
    let mut a = a.to_vec();
    let mut b = b.to_vec();
    let mut s = vec![0.; n];
    let mut query = [0.];
    let mut iquery = [0];
    let mut rank = 0;
    let mut info = 0;
    // SAFETY: `a` has exactly `m * n` elements and `b` exactly `m`
    // (asserted above), matching leading dimension `mi = m`, `ni = n`
    // columns, and one right-hand-side column (`&1`); `s` has `n` elements
    // as dgelsd_ requires. `lwork = -1` is LAPACK's workspace query mode:
    // it only writes the optimal real and integer workspace sizes into
    // `query`/`iquery` (each length 1) without otherwise touching them.
    unsafe {
        dgelsd_(
            &mi,
            &ni,
            &1,
            a.as_mut_ptr(),
            &mi,
            b.as_mut_ptr(),
            &mi,
            s.as_mut_ptr(),
            &-1.,
            &mut rank,
            query.as_mut_ptr(),
            &-1,
            iquery.as_mut_ptr(),
            &mut info,
        );
    }
    result(info)?;
    let mut work = vec![0.; query[0] as usize];
    let mut iwork = vec![0; iquery[0] as usize];
    // SAFETY: same `a`/`b`/`s` argument as the query call above; `work`
    // and `iwork` were just allocated with `query[0]`/`iquery[0]`
    // elements, the sizes dgelsd_ itself reported as sufficient.
    unsafe {
        dgelsd_(
            &mi,
            &ni,
            &1,
            a.as_mut_ptr(),
            &mi,
            b.as_mut_ptr(),
            &mi,
            s.as_mut_ptr(),
            &-1.,
            &mut rank,
            work.as_mut_ptr(),
            &int(work.len()),
            iwork.as_mut_ptr(),
            &mut info,
        );
    }
    result(info)?;
    b.truncate(n);
    Ok(b)
}
fn result(info: i32) -> Result<(), i32> {
    if info == 0 { Ok(()) } else { Err(info) }
}
fn trans(t: Transpose) -> c_char {
    match t {
        Transpose::No => b'N' as c_char,
        Transpose::Yes => b'T' as c_char,
    }
}
fn matrix_len(rows: usize, cols: usize, ld: usize) -> usize {
    assert!(ld >= rows.max(1));
    if rows == 0 || cols == 0 {
        0
    } else {
        (cols - 1) * ld + rows
    }
}
fn validate_gemm<T>(g: &Gemm<'_, T>) {
    let (ar, ac) = match g.trans_a {
        Transpose::No => (g.m, g.k),
        Transpose::Yes => (g.k, g.m),
    };
    let (br, bc) = match g.trans_b {
        Transpose::No => (g.k, g.n),
        Transpose::Yes => (g.n, g.k),
    };
    assert!(g.a.len() >= matrix_len(ar, ac, g.lda));
    assert!(g.b.len() >= matrix_len(br, bc, g.ldb));
    assert!(g.c.len() >= matrix_len(g.m, g.n, g.ldc));
}
pub(crate) fn gemm(g: Gemm<'_, f64>) {
    validate_gemm(&g);
    crate::flop_counter::add_computed_flops(gemm_flops(g.m,g.n,g.k));
    // SAFETY: validate_gemm asserted `g.a`/`g.b`/`g.c` each have at least
    // `matrix_len(rows, cols, ld)` elements for the transpose-adjusted
    // shape and leading dimension passed below, so dgemm_ never reads or
    // writes past their bounds; `g` (and its borrowed slices) outlives
    // this synchronous call.
    unsafe {
        dgemm_(
            &trans(g.trans_a),
            &trans(g.trans_b),
            &int(g.m),
            &int(g.n),
            &int(g.k),
            &g.alpha,
            g.a.as_ptr(),
            &int(g.lda),
            g.b.as_ptr(),
            &int(g.ldb),
            &g.beta,
            g.c.as_mut_ptr(),
            &int(g.ldc),
        );
    }
}
macro_rules! gemm {
    ($function:ident, $native:ident, $scalar:ty) => {
        pub(crate) fn $function(g: Gemm<'_, $scalar>) {
            validate_gemm(&g);
            crate::flop_counter::add_computed_flops(gemm_flops(g.m,g.n,g.k));
            // SAFETY: same argument as the f64 `gemm` above: validate_gemm
            // asserted every buffer is at least as long as its
            // transpose-adjusted shape and leading dimension require.
            unsafe {
                $native(
                    &trans(g.trans_a),
                    &trans(g.trans_b),
                    &int(g.m),
                    &int(g.n),
                    &int(g.k),
                    &g.alpha,
                    g.a.as_ptr(),
                    &int(g.lda),
                    g.b.as_ptr(),
                    &int(g.ldb),
                    &g.beta,
                    g.c.as_mut_ptr(),
                    &int(g.ldc),
                );
            }
        }
    };
}
gemm!(gemm_f32, sgemm_, f32);
gemm!(gemm_c32, cgemm_, Complex<f32>);
gemm!(gemm_c64, zgemm_, Complex<f64>);
fn validate_syr<T>(s:&Syr<'_,T>){assert!(s.incx>0);let x=if s.n==0{0}else{1+(s.n-1)*s.incx as usize};assert!(s.x.len()>=x);assert!(s.a.len()>=matrix_len(s.n,s.n,s.lda));}
fn uplo(value:Uplo)->c_char{match value{Uplo::Lower=>b'L' as c_char,Uplo::Upper=>b'U' as c_char}}
// SAFETY: applies to the call inside the expansion below; validate_syr
// asserted `s.x` has at least `1 + (n-1)*incx` elements and `s.a` at
// least `matrix_len(n, n, lda)`, matching the count/stride/leading-
// dimension arguments passed to the native `*syr_` call.
macro_rules! syr {($function:ident,$native:ident,$scalar:ty)=>{pub(crate) fn $function(s:Syr<'_,$scalar>){validate_syr(&s);unsafe{$native(&uplo(s.uplo),&int(s.n),&s.alpha,s.x.as_ptr(),&s.incx,s.a.as_mut_ptr(),&int(s.lda));}}};}
syr!(syr_f32,ssyr_,f32);syr!(syr,dsyr_,f64);syr!(syr_c32,csyr_,Complex<f32>);syr!(syr_c64,zsyr_,Complex<f64>);
pub(crate) fn cholesky(n: usize, a: &mut [f64], lower: bool) -> Result<(), i32> {
    assert_eq!(a.len(), n * n);
    let mut info = 0;
    // SAFETY: `a` has exactly `n * n` elements (asserted above), matching
    // leading dimension `n.max(1)` for an `n`-by-`n` matrix; `info` is a
    // valid, uniquely-owned out-pointer.
    unsafe {
        dpotrf_(
            &(if lower { b'L' } else { b'U' } as c_char),
            &int(n),
            a.as_mut_ptr(),
            &int(n.max(1)),
            &mut info,
        );
    }
    result(info)
}
pub(crate) fn qr(m: usize, n: usize, a: &[f64]) -> Result<(Vec<f64>, Vec<f64>), i32> {
    assert_eq!(a.len(), m * n);
    let k = m.min(n);
    let mi = int(m);
    let ni = int(n);
    let ki = int(k);
    let lda = int(m.max(1));
    let mut q = a.to_vec();
    let mut tau = vec![0.; k];
    let mut query = [0.];
    let mut info = 0;
    // SAFETY: `q` was cloned from `a`, `m * n` elements (asserted above),
    // matching leading dimension `lda = m.max(1)`; `tau` has `k = min(m,
    // n)` elements as dgeqrf_ requires. `lwork = -1` is LAPACK's workspace
    // query mode: it only writes the optimal size into `query` (length 1).
    unsafe {
        dgeqrf_(
            &mi,
            &ni,
            q.as_mut_ptr(),
            &lda,
            tau.as_mut_ptr(),
            query.as_mut_ptr(),
            &-1,
            &mut info,
        );
    }
    result(info)?;
    let lwork = query[0] as i32;
    let mut work = vec![0.; lwork as usize];
    // SAFETY: same `q`/`tau` argument as the query call above; `work` was
    // just allocated with `lwork` elements, the size dgeqrf_ itself
    // reported as sufficient.
    unsafe {
        dgeqrf_(
            &mi,
            &ni,
            q.as_mut_ptr(),
            &lda,
            tau.as_mut_ptr(),
            work.as_mut_ptr(),
            &lwork,
            &mut info,
        );
    }
    result(info)?;
    let mut r = vec![0.; k * n];
    for j in 0..n {
        for i in 0..k.min(j + 1) {
            r[i + j * k] = q[i + j * m];
        }
    }
    q.truncate(m * k);
    // SAFETY: `q` still has `m * k` elements after the truncate above,
    // matching leading dimension `lda` and `ki = k` reflectors/columns;
    // `tau` has the `k` elements dgeqrf_ wrote above. `lwork = -1` is
    // again a workspace query, writing only into `query` (length 1).
    unsafe {
        dorgqr_(
            &mi,
            &ki,
            &ki,
            q.as_mut_ptr(),
            &lda,
            tau.as_ptr(),
            query.as_mut_ptr(),
            &-1,
            &mut info,
        );
    }
    result(info)?;
    let lwork = query[0] as i32;
    work.resize(lwork as usize, 0.);
    // SAFETY: same `q`/`tau` argument as the query call above; `work` was
    // just resized to `lwork` elements, the size dorgqr_ reported as
    // sufficient.
    unsafe {
        dorgqr_(
            &mi,
            &ki,
            &ki,
            q.as_mut_ptr(),
            &lda,
            tau.as_ptr(),
            work.as_mut_ptr(),
            &lwork,
            &mut info,
        );
    }
    result(info)?;
    Ok((q, r))
}
pub(crate) fn svd(m: usize, n: usize, a: &[f64]) -> Result<Svd, i32> {
    assert_eq!(a.len(), m * n);
    let k = m.min(n);
    let mi = int(m);
    let ni = int(n);
    let lda = int(m.max(1));
    let ldvt = int(k.max(1));
    let mut a = a.to_vec();
    let mut u = vec![0.; m * k];
    let mut values = vec![0.; k];
    let mut vt = vec![0.; k * n];
    let mut query = [0.];
    let mut info = 0;
    let job = b'S' as c_char;
    // SAFETY: `a` has `m * n` elements (asserted above) with leading
    // dimension `lda = m.max(1)`; `u` has `m * k` and `vt` has `k * n`
    // elements (`job = 'S'` writes the reduced `k = min(m, n)` singular
    // vectors), with leading dimensions `lda`/`ldvt` matching their
    // allocated row counts; `values` has the `k` elements dgesvd_ needs
    // for the singular values. `lwork = -1` is a workspace query, writing
    // only into `query` (length 1).
    unsafe {
        dgesvd_(
            &job,
            &job,
            &mi,
            &ni,
            a.as_mut_ptr(),
            &lda,
            values.as_mut_ptr(),
            u.as_mut_ptr(),
            &lda,
            vt.as_mut_ptr(),
            &ldvt,
            query.as_mut_ptr(),
            &-1,
            &mut info,
        );
    }
    result(info)?;
    let lwork = query[0] as i32;
    let mut work = vec![0.; lwork as usize];
    // SAFETY: same argument as the query call above; `work` was just
    // allocated with `lwork` elements, the size dgesvd_ reported as
    // sufficient.
    unsafe {
        dgesvd_(
            &job,
            &job,
            &mi,
            &ni,
            a.as_mut_ptr(),
            &lda,
            values.as_mut_ptr(),
            u.as_mut_ptr(),
            &lda,
            vt.as_mut_ptr(),
            &ldvt,
            work.as_mut_ptr(),
            &lwork,
            &mut info,
        );
    }
    result(info)?;
    Ok(Svd { u, values, vt })
}
pub(crate) fn eigh(n: usize, a: &[f64]) -> Result<(Vec<f64>, Vec<f64>), i32> {
    assert_eq!(a.len(), n * n);
    let mut vectors = a.to_vec();
    let mut values = vec![0.; n];
    let mut info = 0;
    let mut query = [0.];
    let ni = int(n);
    let lda = int(n.max(1));
    let job = b'V' as c_char;
    let uplo = b'U' as c_char;
    // SAFETY: `vectors` was cloned from `a`, `n * n` elements (asserted
    // above), matching leading dimension `lda = n.max(1)`; `values` has
    // the `n` elements dsyev_ needs for the eigenvalues. `lwork = -1` is
    // a workspace query, writing only into `query` (length 1).
    unsafe {
        dsyev_(
            &job,
            &uplo,
            &ni,
            vectors.as_mut_ptr(),
            &lda,
            values.as_mut_ptr(),
            query.as_mut_ptr(),
            &-1,
            &mut info,
        );
    }
    result(info)?;
    let lwork = query[0] as i32;
    let mut work = vec![0.; lwork as usize];
    // SAFETY: same `vectors`/`values` argument as the query call above;
    // `work` was just allocated with `lwork` elements, the size dsyev_
    // reported as sufficient.
    unsafe {
        dsyev_(
            &job,
            &uplo,
            &ni,
            vectors.as_mut_ptr(),
            &lda,
            values.as_mut_ptr(),
            work.as_mut_ptr(),
            &lwork,
            &mut info,
        );
    }
    result(info)?;
    Ok((vectors, values))
}
