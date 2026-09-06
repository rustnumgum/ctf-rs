use std::ffi::c_char;

#[link(name = "blas")]
unsafe extern "C" {
    fn dsyr_(
        uplo: *const c_char,
        n: *const i32,
        alpha: *const f64,
        x: *const f64,
        incx: *const i32,
        a: *mut f64,
        lda: *const i32,
    );
}

#[link(name = "lapack")]
unsafe extern "C" {
    fn dposv_(
        uplo: *const c_char,
        n: *const i32,
        nrhs: *const i32,
        a: *mut f64,
        lda: *const i32,
        b: *mut f64,
        ldb: *const i32,
        info: *mut i32,
    );
}

pub(crate) fn syr(n: usize, alpha: f64, x: &[f64], a: &mut [f64]) {
    assert_eq!(x.len(), n);
    assert_eq!(a.len(), n * n);
    let ni = i32::try_from(n).unwrap();
    let uplo = b'L' as c_char;
    let incx = 1;
    unsafe {
        dsyr_(
            &uplo,
            &ni,
            &alpha,
            x.as_ptr(),
            &incx,
            a.as_mut_ptr(),
            &ni,
        );
    }
}

pub(crate) fn posv(n: usize, a: &mut [f64], b: &mut [f64]) -> Result<(), i32> {
    assert_eq!(a.len(), n * n);
    assert_eq!(b.len(), n);
    let ni = i32::try_from(n).unwrap();
    let nrhs = 1;
    let uplo = b'L' as c_char;
    let mut info = 0;
    unsafe {
        dposv_(
            &uplo,
            &ni,
            &nrhs,
            a.as_mut_ptr(),
            &ni,
            b.as_mut_ptr(),
            &ni,
            &mut info,
        );
    }
    if info == 0 { Ok(()) } else { Err(info) }
}
