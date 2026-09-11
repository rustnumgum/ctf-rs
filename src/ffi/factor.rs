use std::ffi::c_char;

// SAFETY: signature mirrors the reference BLAS Fortran ABI (trailing
// underscore, every scalar and array passed by pointer, column major
// layout); the call site below argues buffer length and leading dimension
// against the caller's asserted shape.
#[cfg_attr(any(target_os = "windows", target_os = "macos"), link(name = "openblas"))]
#[cfg_attr(not(any(target_os = "windows", target_os = "macos")), link(name = "blas"))]
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

// SAFETY: signature mirrors the reference LAPACK Fortran ABI; the call
// site below argues buffer length and leading dimension against the
// caller's asserted shape.
#[cfg_attr(any(target_os = "windows", target_os = "macos"), link(name = "openblas"))]
#[cfg_attr(not(any(target_os = "windows", target_os = "macos")), link(name = "lapack"))]
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
    // SAFETY: `x` has exactly `n` elements and `a` exactly `n * n`
    // (asserted above), matching count/stride `incx = 1` and leading
    // dimension `ni = n` for an `n`-by-`n` matrix.
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
    // SAFETY: `a` has exactly `n * n` elements and `b` exactly `n`
    // (asserted above), matching leading dimensions `ni = n` for the
    // `n`-by-`n` system and one right-hand-side column (`nrhs = 1`);
    // `info` is a valid, uniquely-owned out-pointer.
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
