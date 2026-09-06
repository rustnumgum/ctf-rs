// Independently written Rust bindings to the standard BLAS/LAPACK interfaces.
//! Native BLAS/LAPACK LP64 entry points; no CTF compatibility shim.
use crate::linalg::{Gemm, Svd, Transpose};
use std::ffi::c_char;

#[link(name = "blas")]
unsafe extern "C" {
    fn dormqr_(side:*const c_char,trans:*const c_char,m:*const i32,n:*const i32,k:*const i32,
        a:*const f64,lda:*const i32,tau:*const f64,c:*mut f64,ldc:*const i32,work:*mut f64,lwork:*const i32,info:*mut i32);
    fn dgelsd_(m:*const i32,n:*const i32,nrhs:*const i32,a:*mut f64,lda:*const i32,
        b:*mut f64,ldb:*const i32,s:*mut f64,rcond:*const f64,rank:*mut i32,work:*mut f64,lwork:*const i32,iwork:*mut i32,info:*mut i32);
    fn dgemm_(ta: *const c_char, tb: *const c_char, m: *const i32, n: *const i32, k: *const i32,
        alpha: *const f64, a: *const f64, lda: *const i32, b: *const f64, ldb: *const i32,
        beta: *const f64, c: *mut f64, ldc: *const i32);
}
#[link(name = "lapack")]
unsafe extern "C" {
    fn dpotrf_(uplo: *const c_char, n: *const i32, a: *mut f64, lda: *const i32, info: *mut i32);
    fn dgeqrf_(m: *const i32, n: *const i32, a: *mut f64, lda: *const i32, tau: *mut f64,
        work: *mut f64, lwork: *const i32, info: *mut i32);
    fn dorgqr_(m: *const i32, n: *const i32, k: *const i32, a: *mut f64, lda: *const i32,
        tau: *const f64, work: *mut f64, lwork: *const i32, info: *mut i32);
    fn dgesvd_(ju: *const c_char, jvt: *const c_char, m: *const i32, n: *const i32,
        a: *mut f64, lda: *const i32, s: *mut f64, u: *mut f64, ldu: *const i32,
        vt: *mut f64, ldvt: *const i32, work: *mut f64, lwork: *const i32, info: *mut i32);
    fn dsyev_(job: *const c_char, uplo: *const c_char, n: *const i32, a: *mut f64, lda: *const i32,
        values: *mut f64, work: *mut f64, lwork: *const i32, info: *mut i32);
}
fn int(n: usize) -> i32 { n.try_into().unwrap() }
pub(crate) fn qr_reduce(m:usize,n:usize,a:&[f64],b:&[f64])->Result<(Vec<f64>,Vec<f64>),i32> {
    assert!(m>=n && n>0);assert_eq!(a.len(),m*n);assert_eq!(b.len(),m);
    let (mi,ni)=(int(m),int(n));let mut a=a.to_vec();let mut b=b.to_vec();
    let mut tau=vec![0.;n];let mut query=[0.];let mut info=0;
    unsafe { dgeqrf_(&mi,&ni,a.as_mut_ptr(),&mi,tau.as_mut_ptr(),query.as_mut_ptr(),&-1,&mut info); }
    result(info)?;let mut work=vec![0.;query[0] as usize];
    unsafe { dgeqrf_(&mi,&ni,a.as_mut_ptr(),&mi,tau.as_mut_ptr(),work.as_mut_ptr(),&int(work.len()),&mut info); }
    result(info)?;
    let mut r=vec![0.;n*n];for j in 0..n {for i in 0..=j {r[i+j*n]=a[i+j*m];}}
    unsafe { dormqr_(&(b'L' as c_char),&(b'T' as c_char),&mi,&1,&ni,a.as_ptr(),&mi,tau.as_ptr(),b.as_mut_ptr(),&mi,query.as_mut_ptr(),&-1,&mut info); }
    result(info)?;work.resize(query[0] as usize,0.);
    unsafe { dormqr_(&(b'L' as c_char),&(b'T' as c_char),&mi,&1,&ni,a.as_ptr(),&mi,tau.as_ptr(),b.as_mut_ptr(),&mi,work.as_mut_ptr(),&int(work.len()),&mut info); }
    result(info)?;b.truncate(n);Ok((r,b))
}
pub(crate) fn least_squares(m:usize,n:usize,a:&[f64],b:&[f64])->Result<Vec<f64>,i32> {
    assert!(m>=n && n>0);assert_eq!(a.len(),m*n);assert_eq!(b.len(),m);
    let (mi,ni)=(int(m),int(n));let mut a=a.to_vec();let mut b=b.to_vec();let mut s=vec![0.;n];
    let mut query=[0.];let mut iquery=[0];let mut rank=0;let mut info=0;
    unsafe { dgelsd_(&mi,&ni,&1,a.as_mut_ptr(),&mi,b.as_mut_ptr(),&mi,s.as_mut_ptr(),&-1.,&mut rank,query.as_mut_ptr(),&-1,iquery.as_mut_ptr(),&mut info); }
    result(info)?;let mut work=vec![0.;query[0] as usize];let mut iwork=vec![0;iquery[0] as usize];
    unsafe { dgelsd_(&mi,&ni,&1,a.as_mut_ptr(),&mi,b.as_mut_ptr(),&mi,s.as_mut_ptr(),&-1.,&mut rank,work.as_mut_ptr(),&int(work.len()),iwork.as_mut_ptr(),&mut info); }
    result(info)?;b.truncate(n);Ok(b)
}
fn result(info: i32) -> Result<(), i32> { if info == 0 { Ok(()) } else { Err(info) } }
fn trans(t: Transpose) -> c_char { match t { Transpose::No => b'N' as c_char, Transpose::Yes => b'T' as c_char } }
fn matrix_len(rows: usize, cols: usize, ld: usize) -> usize {
    assert!(ld >= rows.max(1));
    if rows == 0 || cols == 0 { 0 } else { (cols-1)*ld+rows }
}
pub(crate) fn gemm(g: Gemm<'_>) {
    let (ar, ac) = match g.trans_a { Transpose::No => (g.m,g.k), Transpose::Yes => (g.k,g.m) };
    let (br, bc) = match g.trans_b { Transpose::No => (g.k,g.n), Transpose::Yes => (g.n,g.k) };
    assert!(g.a.len() >= matrix_len(ar, ac, g.lda));
    assert!(g.b.len() >= matrix_len(br, bc, g.ldb));
    assert!(g.c.len() >= matrix_len(g.m, g.n, g.ldc));
    unsafe { dgemm_(&trans(g.trans_a), &trans(g.trans_b), &int(g.m), &int(g.n), &int(g.k),
        &g.alpha, g.a.as_ptr(), &int(g.lda), g.b.as_ptr(), &int(g.ldb), &g.beta, g.c.as_mut_ptr(), &int(g.ldc)); }
}
pub(crate) fn cholesky(n: usize, a: &mut [f64], lower: bool) -> Result<(), i32> {
    assert_eq!(a.len(), n*n);
    let mut info = 0;
    unsafe { dpotrf_(&(if lower { b'L' } else { b'U' } as c_char), &int(n), a.as_mut_ptr(), &int(n.max(1)), &mut info); }
    result(info)
}
pub(crate) fn qr(m: usize, n: usize, a: &[f64]) -> Result<(Vec<f64>, Vec<f64>), i32> {
    assert_eq!(a.len(), m*n);
    let k = m.min(n); let mi = int(m); let ni = int(n); let ki = int(k); let lda = int(m.max(1));
    let mut q = a.to_vec(); let mut tau = vec![0.; k]; let mut query = [0.]; let mut info = 0;
    unsafe { dgeqrf_(&mi, &ni, q.as_mut_ptr(), &lda, tau.as_mut_ptr(), query.as_mut_ptr(), &-1, &mut info); }
    result(info)?;
    let lwork = query[0] as i32; let mut work = vec![0.; lwork as usize];
    unsafe { dgeqrf_(&mi, &ni, q.as_mut_ptr(), &lda, tau.as_mut_ptr(), work.as_mut_ptr(), &lwork, &mut info); }
    result(info)?;
    let mut r = vec![0.; k*n];
    for j in 0..n { for i in 0..k.min(j+1) { r[i+j*k] = q[i+j*m]; } }
    q.truncate(m*k);
    unsafe { dorgqr_(&mi, &ki, &ki, q.as_mut_ptr(), &lda, tau.as_ptr(), query.as_mut_ptr(), &-1, &mut info); }
    result(info)?;
    let lwork = query[0] as i32; work.resize(lwork as usize, 0.);
    unsafe { dorgqr_(&mi, &ki, &ki, q.as_mut_ptr(), &lda, tau.as_ptr(), work.as_mut_ptr(), &lwork, &mut info); }
    result(info)?; Ok((q, r))
}
pub(crate) fn svd(m: usize, n: usize, a: &[f64]) -> Result<Svd, i32> {
    assert_eq!(a.len(), m*n);
    let k = m.min(n); let mi = int(m); let ni = int(n); let lda = int(m.max(1)); let ldvt = int(k.max(1));
    let mut a = a.to_vec(); let mut u = vec![0.; m*k]; let mut values = vec![0.; k]; let mut vt = vec![0.; k*n];
    let mut query = [0.]; let mut info = 0; let job = b'S' as c_char;
    unsafe { dgesvd_(&job, &job, &mi, &ni, a.as_mut_ptr(), &lda, values.as_mut_ptr(), u.as_mut_ptr(), &lda,
        vt.as_mut_ptr(), &ldvt, query.as_mut_ptr(), &-1, &mut info); }
    result(info)?; let lwork = query[0] as i32; let mut work = vec![0.; lwork as usize];
    unsafe { dgesvd_(&job, &job, &mi, &ni, a.as_mut_ptr(), &lda, values.as_mut_ptr(), u.as_mut_ptr(), &lda,
        vt.as_mut_ptr(), &ldvt, work.as_mut_ptr(), &lwork, &mut info); }
    result(info)?; Ok(Svd { u, values, vt })
}
pub(crate) fn eigh(n: usize, a: &[f64]) -> Result<(Vec<f64>, Vec<f64>), i32> {
    assert_eq!(a.len(), n*n);
    let mut vectors = a.to_vec(); let mut values = vec![0.; n]; let mut info = 0; let mut query = [0.];
    let ni = int(n); let lda = int(n.max(1)); let job = b'V' as c_char; let uplo = b'U' as c_char;
    unsafe { dsyev_(&job, &uplo, &ni, vectors.as_mut_ptr(), &lda, values.as_mut_ptr(), query.as_mut_ptr(), &-1, &mut info); }
    result(info)?; let lwork = query[0] as i32; let mut work = vec![0.; lwork as usize];
    unsafe { dsyev_(&job, &uplo, &ni, vectors.as_mut_ptr(), &lda, values.as_mut_ptr(), work.as_mut_ptr(), &lwork, &mut info); }
    result(info)?; Ok((vectors, values))
}
