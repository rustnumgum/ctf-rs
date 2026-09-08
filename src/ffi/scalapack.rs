//! Independently written bindings to the ScaLAPACK/BLACS LP64 interfaces.
//!
//! The operation selection and one-based whole-matrix calls follow the pinned
//! CTF reference at f69cbb46e23bc2f39cda5722ce096f56301dab4f
//! (`src/interface/matrix.cxx` and `src/shared/lapack_symbs.cxx`).
use crate::algebra::Complex;
use ::mpi::{ffi as sys, topology::Communicator, traits::AsRaw};
use std::{ffi::c_char, marker::PhantomData, rc::Rc};

#[cfg_attr(target_os = "windows", link(name = "scalapack"))]
#[cfg_attr(not(target_os = "windows"), link(name = "scalapack-openmpi"))]
unsafe extern "C" {
    fn Csys2blacs_handle(comm: sys::MPI_Comm) -> i32;
    fn Cfree_blacs_system_handle(context: i32);
    fn Cblacs_gridinit(context: *mut i32, order: *const c_char, rows: i32, cols: i32);
    fn Cblacs_gridinfo(context: i32, rows: *mut i32, cols: *mut i32, row: *mut i32, col: *mut i32);
    fn Cblacs_gridexit(context: i32);

    fn descinit_(
        desc: *mut i32,
        m: *const i32,
        n: *const i32,
        block_rows: *const i32,
        block_cols: *const i32,
        source_row: *const i32,
        source_col: *const i32,
        context: *const i32,
        lld: *const i32,
        info: *mut i32,
    );
    fn pdpotrf_(
        uplo: *const c_char,
        n: *const i32,
        a: *mut f64,
        ia: *const i32,
        ja: *const i32,
        desc: *const i32,
        info: *mut i32,
    );
    fn pdposv_(
        uplo: *const c_char,
        n: *const i32,
        nrhs: *const i32,
        a: *mut f64,
        ia: *const i32,
        ja: *const i32,
        desc_a: *const i32,
        b: *mut f64,
        ib: *const i32,
        jb: *const i32,
        desc_b: *const i32,
        info: *mut i32,
    );
    fn pdsyevx_(
        job_z: *const c_char,
        range: *const c_char,
        uplo: *const c_char,
        n: *const i32,
        a: *mut f64,
        ia: *const i32,
        ja: *const i32,
        desc_a: *const i32,
        vl: *const f64,
        vu: *const f64,
        il: *const i32,
        iu: *const i32,
        abstol: *const f64,
        m: *mut i32,
        nz: *mut i32,
        w: *mut f64,
        orfac: *const f64,
        z: *mut f64,
        iz: *const i32,
        jz: *const i32,
        desc_z: *const i32,
        work: *mut f64,
        lwork: *const i32,
        iwork: *mut i32,
        liwork: *const i32,
        ifail: *mut i32,
        iclustr: *mut i32,
        gap: *mut f64,
        info: *mut i32,
    );
    fn pdgeqrf_(
        m: *const i32,
        n: *const i32,
        a: *mut f64,
        ia: *const i32,
        ja: *const i32,
        desc: *const i32,
        tau: *mut f64,
        work: *mut f64,
        lwork: *const i32,
        info: *mut i32,
    );
    fn pdorgqr_(
        m: *const i32,
        n: *const i32,
        k: *const i32,
        a: *mut f64,
        ia: *const i32,
        ja: *const i32,
        desc: *const i32,
        tau: *const f64,
        work: *mut f64,
        lwork: *const i32,
        info: *mut i32,
    );
    fn pdgesvd_(
        job_u: *const c_char,
        job_vt: *const c_char,
        m: *const i32,
        n: *const i32,
        a: *mut f64,
        ia: *const i32,
        ja: *const i32,
        desc_a: *const i32,
        s: *mut f64,
        u: *mut f64,
        iu: *const i32,
        ju: *const i32,
        desc_u: *const i32,
        vt: *mut f64,
        ivt: *const i32,
        jvt: *const i32,
        desc_vt: *const i32,
        work: *mut f64,
        lwork: *const i32,
        info: *mut i32,
    );
    fn pdtrsm_(
        side: *const c_char,
        uplo: *const c_char,
        transpose: *const c_char,
        diagonal: *const c_char,
        m: *const i32,
        n: *const i32,
        alpha: *const f64,
        factor: *const f64,
        ia: *const i32,
        ja: *const i32,
        desc_factor: *const i32,
        b: *mut f64,
        ib: *const i32,
        jb: *const i32,
        desc_b: *const i32,
    );
}

pub(crate) struct Grid {
    context: i32,
    system_context: i32,
    rows: usize,
    cols: usize,
    row: usize,
    col: usize,
    _single_thread: PhantomData<Rc<()>>,
}

macro_rules! typed_spd_family {
    (
        $scalar:ty,
        $potrf:ident,
        $posv:ident,
        $trsm:ident,
        $cholesky:ident,
        $solve_spd:ident,
        $solve_tri:ident,
        $one:expr
    ) => {
        unsafe extern "C" {
            fn $potrf(
                uplo: *const c_char,
                n: *const i32,
                a: *mut $scalar,
                ia: *const i32,
                ja: *const i32,
                desc: *const i32,
                info: *mut i32,
            );
            fn $posv(
                uplo: *const c_char,
                n: *const i32,
                nrhs: *const i32,
                a: *mut $scalar,
                ia: *const i32,
                ja: *const i32,
                desc_a: *const i32,
                b: *mut $scalar,
                ib: *const i32,
                jb: *const i32,
                desc_b: *const i32,
                info: *mut i32,
            );
            fn $trsm(
                side: *const c_char,
                uplo: *const c_char,
                transpose: *const c_char,
                diagonal: *const c_char,
                m: *const i32,
                n: *const i32,
                alpha: *const $scalar,
                factor: *const $scalar,
                ia: *const i32,
                ja: *const i32,
                desc_factor: *const i32,
                b: *mut $scalar,
                ib: *const i32,
                jb: *const i32,
                desc_b: *const i32,
            );
        }

        impl Grid {
            pub(crate) fn $cholesky(
                &self,
                n: usize,
                a: &mut [$scalar],
                desc: &[i32; 9],
                lower: bool,
            ) -> Result<(), i32> {
                self.validate_matrix(desc, a.len());
                let n = int(n);
                assert!(n <= desc[2] && n <= desc[3]);
                let uplo = if lower { b'L' } else { b'U' } as c_char;
                let one = 1;
                let mut info = 0;
                unsafe {
                    $potrf(
                        &uplo,
                        &n,
                        a.as_mut_ptr(),
                        &one,
                        &one,
                        desc.as_ptr(),
                        &mut info,
                    );
                }
                result(info)
            }

            pub(crate) fn $solve_spd(
                &self,
                n: usize,
                nrhs: usize,
                a: &mut [$scalar],
                desc_a: &[i32; 9],
                b: &mut [$scalar],
                desc_b: &[i32; 9],
            ) -> Result<(), i32> {
                self.validate_matrix(desc_a, a.len());
                self.validate_matrix(desc_b, b.len());
                let (n, nrhs) = (int(n), int(nrhs));
                assert!(n <= desc_a[2] && n <= desc_a[3]);
                assert!(n <= desc_b[2] && nrhs <= desc_b[3]);
                let uplo = b'L' as c_char;
                let one = 1;
                let mut info = 0;
                unsafe {
                    $posv(
                        &uplo,
                        &n,
                        &nrhs,
                        a.as_mut_ptr(),
                        &one,
                        &one,
                        desc_a.as_ptr(),
                        b.as_mut_ptr(),
                        &one,
                        &one,
                        desc_b.as_ptr(),
                        &mut info,
                    );
                }
                result(info)
            }

            #[allow(clippy::too_many_arguments)]
            pub(crate) fn $solve_tri(
                &self,
                m: usize,
                n: usize,
                factor: &[$scalar],
                desc_factor: &[i32; 9],
                b: &mut [$scalar],
                desc_b: &[i32; 9],
                lower: bool,
                from_left: bool,
                transpose: bool,
            ) {
                self.validate_matrix(desc_factor, factor.len());
                self.validate_matrix(desc_b, b.len());
                let factor_order = if from_left { m } else { n };
                assert!(int(factor_order) <= desc_factor[2] && int(factor_order) <= desc_factor[3]);
                assert!(int(m) <= desc_b[2] && int(n) <= desc_b[3]);
                let side = if from_left { b'L' } else { b'R' } as c_char;
                let uplo = if lower { b'L' } else { b'U' } as c_char;
                let transpose = if transpose { b'T' } else { b'N' } as c_char;
                let diagonal = b'N' as c_char;
                let (m, n) = (int(m), int(n));
                let one = 1;
                let alpha: $scalar = $one;
                unsafe {
                    $trsm(
                        &side,
                        &uplo,
                        &transpose,
                        &diagonal,
                        &m,
                        &n,
                        &alpha,
                        factor.as_ptr(),
                        &one,
                        &one,
                        desc_factor.as_ptr(),
                        b.as_mut_ptr(),
                        &one,
                        &one,
                        desc_b.as_ptr(),
                    );
                }
            }
        }
    };
}

typed_spd_family!(
    f32,
    pspotrf_,
    psposv_,
    pstrsm_,
    cholesky_f32,
    solve_spd_f32,
    solve_tri_f32,
    1.0
);
typed_spd_family!(
    Complex<f32>,
    pcpotrf_,
    pcposv_,
    pctrsm_,
    cholesky_c32,
    solve_spd_c32,
    solve_tri_c32,
    Complex::new(1.0, 0.0)
);
typed_spd_family!(
    Complex<f64>,
    pzpotrf_,
    pzposv_,
    pztrsm_,
    cholesky_c64,
    solve_spd_c64,
    solve_tri_c64,
    Complex::new(1.0, 0.0)
);

macro_rules! typed_qr_family {
    (
        $scalar:ty,
        $geqrf:ident,
        $generate_q:ident,
        $qr:ident,
        $zero:expr,
        $query_len:expr
    ) => {
        unsafe extern "C" {
            fn $geqrf(
                m: *const i32,
                n: *const i32,
                a: *mut $scalar,
                ia: *const i32,
                ja: *const i32,
                desc: *const i32,
                tau: *mut $scalar,
                work: *mut $scalar,
                lwork: *const i32,
                info: *mut i32,
            );
            fn $generate_q(
                m: *const i32,
                n: *const i32,
                k: *const i32,
                a: *mut $scalar,
                ia: *const i32,
                ja: *const i32,
                desc: *const i32,
                tau: *const $scalar,
                work: *mut $scalar,
                lwork: *const i32,
                info: *mut i32,
            );
        }

        impl Grid {
            pub(crate) fn $qr(
                &self,
                m: usize,
                n: usize,
                a: &[$scalar],
                desc: &[i32; 9],
            ) -> Result<(Vec<$scalar>, Vec<$scalar>), i32> {
                self.validate_matrix(desc, a.len());
                let (m, n) = (int(m), int(n));
                assert!(m <= desc[2] && n <= desc[3]);

                let k = m.min(n);
                let tau_len = numroc(
                    k as usize,
                    desc[5] as usize,
                    self.col,
                    desc[7] as usize,
                    self.cols,
                );
                let mut tau = vec![$zero; tau_len];
                let mut q = a.to_vec();
                let one = 1;
                let query = -1;
                let mut work_query = [$zero];
                let mut info = 0;
                unsafe {
                    $geqrf(
                        &m,
                        &n,
                        q.as_mut_ptr(),
                        &one,
                        &one,
                        desc.as_ptr(),
                        tau.as_mut_ptr(),
                        work_query.as_mut_ptr(),
                        &query,
                        &mut info,
                    );
                }
                result(info)?;

                let lwork = int(($query_len)(work_query[0]));
                let mut work = vec![$zero; lwork as usize];
                unsafe {
                    $geqrf(
                        &m,
                        &n,
                        q.as_mut_ptr(),
                        &one,
                        &one,
                        desc.as_ptr(),
                        tau.as_mut_ptr(),
                        work.as_mut_ptr(),
                        &lwork,
                        &mut info,
                    );
                }
                result(info)?;

                let r = q.clone();
                work_query[0] = $zero;
                unsafe {
                    $generate_q(
                        &m,
                        &k,
                        &k,
                        q.as_mut_ptr(),
                        &one,
                        &one,
                        desc.as_ptr(),
                        tau.as_ptr(),
                        work_query.as_mut_ptr(),
                        &query,
                        &mut info,
                    );
                }
                result(info)?;

                let lwork = int(($query_len)(work_query[0]));
                let mut work = vec![$zero; lwork as usize];
                unsafe {
                    $generate_q(
                        &m,
                        &k,
                        &k,
                        q.as_mut_ptr(),
                        &one,
                        &one,
                        desc.as_ptr(),
                        tau.as_ptr(),
                        work.as_mut_ptr(),
                        &lwork,
                        &mut info,
                    );
                }
                result(info).map(|()| (q, r))
            }
        }
    };
}

typed_qr_family!(f32, psgeqrf_, psorgqr_, qr_f32, 0.0, |value: f32| value
    as usize);
typed_qr_family!(
    Complex<f32>,
    pcgeqrf_,
    pcungqr_,
    qr_c32,
    Complex::new(0.0, 0.0),
    |value: Complex<f32>| value.re as usize
);
typed_qr_family!(
    Complex<f64>,
    pzgeqrf_,
    pzungqr_,
    qr_c64,
    Complex::new(0.0, 0.0),
    |value: Complex<f64>| value.re as usize
);

macro_rules! real_svd_family {
    ($scalar:ty, $gesvd:ident, $svd:ident, $zero:expr, $query_len:expr) => {
        unsafe extern "C" {
            fn $gesvd(
                job_u: *const c_char,
                job_vt: *const c_char,
                m: *const i32,
                n: *const i32,
                a: *mut $scalar,
                ia: *const i32,
                ja: *const i32,
                desc_a: *const i32,
                s: *mut $scalar,
                u: *mut $scalar,
                iu: *const i32,
                ju: *const i32,
                desc_u: *const i32,
                vt: *mut $scalar,
                ivt: *const i32,
                jvt: *const i32,
                desc_vt: *const i32,
                work: *mut $scalar,
                lwork: *const i32,
                info: *mut i32,
            );
        }

        impl Grid {
            #[allow(clippy::too_many_arguments)]
            pub(crate) fn $svd(
                &self,
                m: usize,
                n: usize,
                a: &[$scalar],
                desc_a: &[i32; 9],
                u: &mut [$scalar],
                desc_u: &[i32; 9],
                vt: &mut [$scalar],
                desc_vt: &[i32; 9],
            ) -> Result<Vec<$scalar>, i32> {
                self.validate_matrix(desc_a, a.len());
                self.validate_matrix(desc_u, u.len());
                self.validate_matrix(desc_vt, vt.len());
                let (m, n) = (int(m), int(n));
                let k = m.min(n);
                assert!(m <= desc_a[2] && n <= desc_a[3]);
                assert!(m <= desc_u[2] && k <= desc_u[3]);
                assert!(k <= desc_vt[2] && n <= desc_vt[3]);

                let mut a = a.to_vec();
                let mut s = vec![$zero; k as usize];
                let vectors = b'V' as c_char;
                let one = 1;
                let query = -1;
                let mut work_query = [$zero];
                let mut info = 0;
                unsafe {
                    $gesvd(
                        &vectors,
                        &vectors,
                        &m,
                        &n,
                        a.as_mut_ptr(),
                        &one,
                        &one,
                        desc_a.as_ptr(),
                        s.as_mut_ptr(),
                        u.as_mut_ptr(),
                        &one,
                        &one,
                        desc_u.as_ptr(),
                        vt.as_mut_ptr(),
                        &one,
                        &one,
                        desc_vt.as_ptr(),
                        work_query.as_mut_ptr(),
                        &query,
                        &mut info,
                    );
                }
                result(info)?;

                let lwork = int(($query_len)(work_query[0]));
                let mut work = vec![$zero; lwork as usize];
                unsafe {
                    $gesvd(
                        &vectors,
                        &vectors,
                        &m,
                        &n,
                        a.as_mut_ptr(),
                        &one,
                        &one,
                        desc_a.as_ptr(),
                        s.as_mut_ptr(),
                        u.as_mut_ptr(),
                        &one,
                        &one,
                        desc_u.as_ptr(),
                        vt.as_mut_ptr(),
                        &one,
                        &one,
                        desc_vt.as_ptr(),
                        work.as_mut_ptr(),
                        &lwork,
                        &mut info,
                    );
                }
                result(info).map(|()| s)
            }
        }
    };
}

macro_rules! complex_svd_family {
    (
        $scalar:ty,
        $real:ty,
        $gesvd:ident,
        $svd:ident,
        $scalar_zero:expr,
        $real_zero:expr
    ) => {
        unsafe extern "C" {
            fn $gesvd(
                job_u: *const c_char,
                job_vt: *const c_char,
                m: *const i32,
                n: *const i32,
                a: *mut $scalar,
                ia: *const i32,
                ja: *const i32,
                desc_a: *const i32,
                s: *mut $real,
                u: *mut $scalar,
                iu: *const i32,
                ju: *const i32,
                desc_u: *const i32,
                vt: *mut $scalar,
                ivt: *const i32,
                jvt: *const i32,
                desc_vt: *const i32,
                work: *mut $scalar,
                lwork: *const i32,
                rwork: *mut $real,
                info: *mut i32,
            );
        }

        impl Grid {
            #[allow(clippy::too_many_arguments)]
            pub(crate) fn $svd(
                &self,
                m: usize,
                n: usize,
                a: &[$scalar],
                desc_a: &[i32; 9],
                u: &mut [$scalar],
                desc_u: &[i32; 9],
                vt: &mut [$scalar],
                desc_vt: &[i32; 9],
            ) -> Result<Vec<$scalar>, i32> {
                self.validate_matrix(desc_a, a.len());
                self.validate_matrix(desc_u, u.len());
                self.validate_matrix(desc_vt, vt.len());
                let (m, n) = (int(m), int(n));
                let k = m.min(n);
                assert!(m <= desc_a[2] && n <= desc_a[3]);
                assert!(m <= desc_u[2] && k <= desc_u[3]);
                assert!(k <= desc_vt[2] && n <= desc_vt[3]);

                let mut a = a.to_vec();
                let mut s = vec![$real_zero; k as usize];
                let mut rwork = vec![$real_zero; 4 * m.max(n) as usize + 1];
                let vectors = b'V' as c_char;
                let one = 1;
                let query = -1;
                let mut work_query = [$scalar_zero];
                let mut info = 0;
                unsafe {
                    $gesvd(
                        &vectors,
                        &vectors,
                        &m,
                        &n,
                        a.as_mut_ptr(),
                        &one,
                        &one,
                        desc_a.as_ptr(),
                        s.as_mut_ptr(),
                        u.as_mut_ptr(),
                        &one,
                        &one,
                        desc_u.as_ptr(),
                        vt.as_mut_ptr(),
                        &one,
                        &one,
                        desc_vt.as_ptr(),
                        work_query.as_mut_ptr(),
                        &query,
                        rwork.as_mut_ptr(),
                        &mut info,
                    );
                }
                result(info)?;

                let lwork = int(work_query[0].re as usize);
                let mut work = vec![$scalar_zero; lwork as usize];
                unsafe {
                    $gesvd(
                        &vectors,
                        &vectors,
                        &m,
                        &n,
                        a.as_mut_ptr(),
                        &one,
                        &one,
                        desc_a.as_ptr(),
                        s.as_mut_ptr(),
                        u.as_mut_ptr(),
                        &one,
                        &one,
                        desc_u.as_ptr(),
                        vt.as_mut_ptr(),
                        &one,
                        &one,
                        desc_vt.as_ptr(),
                        work.as_mut_ptr(),
                        &lwork,
                        rwork.as_mut_ptr(),
                        &mut info,
                    );
                }
                result(info).map(|()| {
                    s.into_iter()
                        .map(|value| Complex::new(value, $real_zero))
                        .collect()
                })
            }
        }
    };
}

real_svd_family!(f32, psgesvd_, svd_f32, 0.0, |value: f32| value as usize);
complex_svd_family!(
    Complex<f32>,
    f32,
    pcgesvd_,
    svd_c32,
    Complex::new(0.0, 0.0),
    0.0
);
complex_svd_family!(
    Complex<f64>,
    f64,
    pzgesvd_,
    svd_c64,
    Complex::new(0.0, 0.0),
    0.0
);

macro_rules! real_eigh_family {
    ($scalar:ty, $syevx:ident, $eigh:ident, $zero:expr, $query_len:expr) => {
        unsafe extern "C" {
            fn $syevx(
                job_z: *const c_char,
                range: *const c_char,
                uplo: *const c_char,
                n: *const i32,
                a: *mut $scalar,
                ia: *const i32,
                ja: *const i32,
                desc_a: *const i32,
                vl: *const $scalar,
                vu: *const $scalar,
                il: *const i32,
                iu: *const i32,
                abstol: *const $scalar,
                m: *mut i32,
                nz: *mut i32,
                w: *mut $scalar,
                orfac: *const $scalar,
                z: *mut $scalar,
                iz: *const i32,
                jz: *const i32,
                desc_z: *const i32,
                work: *mut $scalar,
                lwork: *const i32,
                iwork: *mut i32,
                liwork: *const i32,
                ifail: *mut i32,
                iclustr: *mut i32,
                gap: *mut $scalar,
                info: *mut i32,
            );
        }

        impl Grid {
            pub(crate) fn $eigh(
                &self,
                n: usize,
                a: &[$scalar],
                desc: &[i32; 9],
                vectors: &mut [$scalar],
            ) -> Result<Vec<$scalar>, i32> {
                self.validate_matrix(desc, a.len());
                self.validate_matrix(desc, vectors.len());
                let n = int(n);
                assert!(n <= desc[2] && n <= desc[3]);

                let job_z = b'V' as c_char;
                let range = b'A' as c_char;
                let uplo = b'U' as c_char;
                let one = 1;
                let zero_i = 0;
                let zero: $scalar = $zero;
                let query = -1;
                let (mut m, mut nz) = (0, 0);
                let mut work_query = [$zero];
                let mut iwork_query = [0];
                let mut info = 0;
                unsafe {
                    $syevx(
                        &job_z,
                        &range,
                        &uplo,
                        &n,
                        std::ptr::null_mut(),
                        &one,
                        &one,
                        desc.as_ptr(),
                        &zero,
                        &zero,
                        &zero_i,
                        &zero_i,
                        &zero,
                        &mut m,
                        &mut nz,
                        std::ptr::null_mut(),
                        &zero,
                        std::ptr::null_mut(),
                        &one,
                        &one,
                        desc.as_ptr(),
                        work_query.as_mut_ptr(),
                        &query,
                        iwork_query.as_mut_ptr(),
                        &query,
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                        &mut info,
                    );
                }
                result(info)?;

                let lwork = int(($query_len)(work_query[0]));
                let liwork = iwork_query[0];
                let mut work = vec![$zero; lwork as usize];
                let mut iwork = vec![0; usize::try_from(liwork).unwrap()];
                let mut ifail = vec![0; n as usize];
                let processes = self.rows.checked_mul(self.cols).unwrap();
                let mut iclustr = vec![0; 2usize.checked_mul(processes).unwrap()];
                let mut gap = vec![$zero; processes];
                let mut matrix = a.to_vec();
                let mut eigenvalues = vec![$zero; n as usize];
                unsafe {
                    $syevx(
                        &job_z,
                        &range,
                        &uplo,
                        &n,
                        matrix.as_mut_ptr(),
                        &one,
                        &one,
                        desc.as_ptr(),
                        &zero,
                        &zero,
                        &zero_i,
                        &zero_i,
                        &zero,
                        &mut m,
                        &mut nz,
                        eigenvalues.as_mut_ptr(),
                        &zero,
                        vectors.as_mut_ptr(),
                        &one,
                        &one,
                        desc.as_ptr(),
                        work.as_mut_ptr(),
                        &lwork,
                        iwork.as_mut_ptr(),
                        &liwork,
                        ifail.as_mut_ptr(),
                        iclustr.as_mut_ptr(),
                        gap.as_mut_ptr(),
                        &mut info,
                    );
                }
                result(info)?;
                assert_eq!(m, n);
                assert_eq!(nz, n);
                Ok(eigenvalues)
            }
        }
    };
}

macro_rules! complex_eigh_family {
    (
        $scalar:ty,
        $real:ty,
        $heevx:ident,
        $eigh:ident,
        $scalar_zero:expr,
        $real_zero:expr
    ) => {
        unsafe extern "C" {
            fn $heevx(
                job_z: *const c_char,
                range: *const c_char,
                uplo: *const c_char,
                n: *const i32,
                a: *mut $scalar,
                ia: *const i32,
                ja: *const i32,
                desc_a: *const i32,
                vl: *const $real,
                vu: *const $real,
                il: *const i32,
                iu: *const i32,
                abstol: *const $real,
                m: *mut i32,
                nz: *mut i32,
                w: *mut $real,
                orfac: *const $real,
                z: *mut $scalar,
                iz: *const i32,
                jz: *const i32,
                desc_z: *const i32,
                work: *mut $scalar,
                lwork: *const i32,
                rwork: *mut $real,
                lrwork: *const i32,
                iwork: *mut i32,
                liwork: *const i32,
                ifail: *mut i32,
                iclustr: *mut i32,
                gap: *mut $real,
                info: *mut i32,
            );
        }

        impl Grid {
            pub(crate) fn $eigh(
                &self,
                n: usize,
                a: &[$scalar],
                desc: &[i32; 9],
                vectors: &mut [$scalar],
            ) -> Result<Vec<$real>, i32> {
                self.validate_matrix(desc, a.len());
                self.validate_matrix(desc, vectors.len());
                let n = int(n);
                assert!(n <= desc[2] && n <= desc[3]);

                let job_z = b'V' as c_char;
                let range = b'A' as c_char;
                let uplo = b'U' as c_char;
                let one = 1;
                let zero_i = 0;
                let zero: $real = $real_zero;
                let query = -1;
                let (mut m, mut nz) = (0, 0);
                let mut work_query = [$scalar_zero];
                let mut rwork_query = [$real_zero];
                let mut iwork_query = [0];
                let mut info = 0;
                unsafe {
                    $heevx(
                        &job_z,
                        &range,
                        &uplo,
                        &n,
                        std::ptr::null_mut(),
                        &one,
                        &one,
                        desc.as_ptr(),
                        &zero,
                        &zero,
                        &zero_i,
                        &zero_i,
                        &zero,
                        &mut m,
                        &mut nz,
                        std::ptr::null_mut(),
                        &zero,
                        std::ptr::null_mut(),
                        &one,
                        &one,
                        desc.as_ptr(),
                        work_query.as_mut_ptr(),
                        &query,
                        rwork_query.as_mut_ptr(),
                        &query,
                        iwork_query.as_mut_ptr(),
                        &query,
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                        &mut info,
                    );
                }
                result(info)?;

                let lwork = int(work_query[0].re as usize);
                // Unlike the pinned pheevx forwarding typo, LRWORK describes
                // the actual real buffer, independently of complex LWORK.
                let lrwork = int(rwork_query[0] as usize);
                let liwork = iwork_query[0];
                let mut work = vec![$scalar_zero; lwork as usize];
                let mut rwork = vec![$real_zero; lrwork as usize];
                let mut iwork = vec![0; usize::try_from(liwork).unwrap()];
                let mut ifail = vec![0; n as usize];
                let processes = self.rows.checked_mul(self.cols).unwrap();
                let mut iclustr = vec![0; 2usize.checked_mul(processes).unwrap()];
                let mut gap = vec![$real_zero; processes];
                let mut matrix = a.to_vec();
                let mut eigenvalues = vec![$real_zero; n as usize];
                unsafe {
                    $heevx(
                        &job_z,
                        &range,
                        &uplo,
                        &n,
                        matrix.as_mut_ptr(),
                        &one,
                        &one,
                        desc.as_ptr(),
                        &zero,
                        &zero,
                        &zero_i,
                        &zero_i,
                        &zero,
                        &mut m,
                        &mut nz,
                        eigenvalues.as_mut_ptr(),
                        &zero,
                        vectors.as_mut_ptr(),
                        &one,
                        &one,
                        desc.as_ptr(),
                        work.as_mut_ptr(),
                        &lwork,
                        rwork.as_mut_ptr(),
                        &lrwork,
                        iwork.as_mut_ptr(),
                        &liwork,
                        ifail.as_mut_ptr(),
                        iclustr.as_mut_ptr(),
                        gap.as_mut_ptr(),
                        &mut info,
                    );
                }
                result(info)?;
                assert_eq!(m, n);
                assert_eq!(nz, n);
                Ok(eigenvalues)
            }
        }
    };
}

real_eigh_family!(f32, pssyevx_, eigh_f32, 0.0, |value: f32| value as usize);
complex_eigh_family!(
    Complex<f32>,
    f32,
    pcheevx_,
    eigh_c32,
    Complex::new(0.0, 0.0),
    0.0
);
complex_eigh_family!(
    Complex<f64>,
    f64,
    pzheevx_,
    eigh_c64,
    Complex::new(0.0, 0.0),
    0.0
);

fn int(value: usize) -> i32 {
    value.try_into().unwrap()
}

fn result(info: i32) -> Result<(), i32> {
    if info == 0 { Ok(()) } else { Err(info) }
}

fn numroc(n: usize, block: usize, process: usize, source: usize, processes: usize) -> usize {
    let full_blocks = n / block;
    let remainder = n % block;
    let distance = (process + processes - source) % processes;
    let mut local_blocks = full_blocks / processes;
    if distance < full_blocks % processes {
        local_blocks += 1;
    }
    local_blocks * block + usize::from(distance == full_blocks % processes) * remainder
}

impl Grid {
    pub(crate) fn new<C: Communicator + ?Sized>(comm: &C, rows: usize, cols: usize) -> Self {
        assert!(rows > 0 && cols > 0);
        let processes = rows.checked_mul(cols).unwrap();
        let size = comm.size();
        let rank = comm.rank();
        assert_eq!(processes, usize::try_from(size).unwrap());

        let system_context = unsafe { Csys2blacs_handle(comm.as_raw()) };
        let mut context = system_context;
        let order = b'C' as c_char;
        unsafe {
            Cblacs_gridinit(&mut context, &order, int(rows), int(cols));
        }

        let (mut actual_rows, mut actual_cols, mut row, mut col) = (0, 0, 0, 0);
        unsafe {
            Cblacs_gridinfo(
                context,
                &mut actual_rows,
                &mut actual_cols,
                &mut row,
                &mut col,
            );
        }
        assert_eq!((actual_rows, actual_cols), (int(rows), int(cols)));
        let (row, col) = (usize::try_from(row).unwrap(), usize::try_from(col).unwrap());
        assert_eq!(usize::try_from(rank).unwrap(), row + col * rows);

        Self {
            context,
            system_context,
            rows,
            cols,
            row,
            col,
            _single_thread: PhantomData,
        }
    }

    pub(crate) fn descriptor(
        &self,
        m: usize,
        n: usize,
        block_rows: usize,
        block_cols: usize,
        lld: usize,
    ) -> Result<[i32; 9], i32> {
        let (m, n) = (int(m), int(n));
        let (block_rows, block_cols) = (int(block_rows), int(block_cols));
        let lld = int(lld);
        let source = 0;
        let mut desc = [0; 9];
        let mut info = 0;
        unsafe {
            descinit_(
                desc.as_mut_ptr(),
                &m,
                &n,
                &block_rows,
                &block_cols,
                &source,
                &source,
                &self.context,
                &lld,
                &mut info,
            );
        }
        result(info).map(|()| desc)
    }

    pub(crate) fn cholesky(
        &self,
        n: usize,
        a: &mut [f64],
        desc: &[i32; 9],
        lower: bool,
    ) -> Result<(), i32> {
        self.validate_matrix(desc, a.len());
        let n = int(n);
        assert!(n <= desc[2] && n <= desc[3]);
        let uplo = if lower { b'L' } else { b'U' } as c_char;
        let one = 1;
        let mut info = 0;
        unsafe {
            pdpotrf_(
                &uplo,
                &n,
                a.as_mut_ptr(),
                &one,
                &one,
                desc.as_ptr(),
                &mut info,
            );
        }
        result(info)
    }

    pub(crate) fn solve_spd(
        &self,
        n: usize,
        nrhs: usize,
        a: &mut [f64],
        desc_a: &[i32; 9],
        b: &mut [f64],
        desc_b: &[i32; 9],
    ) -> Result<(), i32> {
        self.validate_matrix(desc_a, a.len());
        self.validate_matrix(desc_b, b.len());
        let (n, nrhs) = (int(n), int(nrhs));
        assert!(n <= desc_a[2] && n <= desc_a[3]);
        assert!(n <= desc_b[2] && nrhs <= desc_b[3]);

        let uplo = b'L' as c_char;
        let one = 1;
        let mut info = 0;
        unsafe {
            pdposv_(
                &uplo,
                &n,
                &nrhs,
                a.as_mut_ptr(),
                &one,
                &one,
                desc_a.as_ptr(),
                b.as_mut_ptr(),
                &one,
                &one,
                desc_b.as_ptr(),
                &mut info,
            );
        }
        result(info)
    }

    pub(crate) fn eigh(
        &self,
        n: usize,
        a: &[f64],
        desc: &[i32; 9],
        vectors: &mut [f64],
    ) -> Result<Vec<f64>, i32> {
        self.validate_matrix(desc, a.len());
        self.validate_matrix(desc, vectors.len());
        let n = int(n);
        assert!(n <= desc[2] && n <= desc[3]);

        let job_z = b'V' as c_char;
        let range = b'A' as c_char;
        let uplo = b'U' as c_char;
        let one = 1;
        let zero_i = 0;
        let zero = 0.0;
        let query = -1;
        let (mut m, mut nz) = (0, 0);
        let mut work_query = [0.0];
        let mut iwork_query = [0];
        let mut info = 0;
        unsafe {
            pdsyevx_(
                &job_z,
                &range,
                &uplo,
                &n,
                std::ptr::null_mut(),
                &one,
                &one,
                desc.as_ptr(),
                &zero,
                &zero,
                &zero_i,
                &zero_i,
                &zero,
                &mut m,
                &mut nz,
                std::ptr::null_mut(),
                &zero,
                std::ptr::null_mut(),
                &one,
                &one,
                desc.as_ptr(),
                work_query.as_mut_ptr(),
                &query,
                iwork_query.as_mut_ptr(),
                &query,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &mut info,
            );
        }
        result(info)?;

        let lwork = int(work_query[0] as usize);
        let liwork = iwork_query[0];
        let mut work = vec![0.0; lwork as usize];
        let mut iwork = vec![0; usize::try_from(liwork).unwrap()];
        let mut ifail = vec![0; n as usize];
        let processes = self.rows.checked_mul(self.cols).unwrap();
        let mut iclustr = vec![0; 2usize.checked_mul(processes).unwrap()];
        let mut gap = vec![0.0; processes];
        let mut matrix = a.to_vec();
        let mut eigenvalues = vec![0.0; n as usize];
        unsafe {
            pdsyevx_(
                &job_z,
                &range,
                &uplo,
                &n,
                matrix.as_mut_ptr(),
                &one,
                &one,
                desc.as_ptr(),
                &zero,
                &zero,
                &zero_i,
                &zero_i,
                &zero,
                &mut m,
                &mut nz,
                eigenvalues.as_mut_ptr(),
                &zero,
                vectors.as_mut_ptr(),
                &one,
                &one,
                desc.as_ptr(),
                work.as_mut_ptr(),
                &lwork,
                iwork.as_mut_ptr(),
                &liwork,
                ifail.as_mut_ptr(),
                iclustr.as_mut_ptr(),
                gap.as_mut_ptr(),
                &mut info,
            );
        }
        result(info)?;
        assert_eq!(m, n);
        assert_eq!(nz, n);
        Ok(eigenvalues)
    }

    pub(crate) fn qr(
        &self,
        m: usize,
        n: usize,
        a: &[f64],
        desc: &[i32; 9],
    ) -> Result<(Vec<f64>, Vec<f64>), i32> {
        self.validate_matrix(desc, a.len());
        let (m, n) = (int(m), int(n));
        assert!(m <= desc[2] && n <= desc[3]);

        let k = m.min(n);
        let tau_len = numroc(
            k as usize,
            desc[5] as usize,
            self.col,
            desc[7] as usize,
            self.cols,
        );
        let mut tau = vec![0.0; tau_len];
        let mut q = a.to_vec();
        let one = 1;
        let query = -1;
        let mut work_query = [0.0];
        let mut info = 0;
        unsafe {
            pdgeqrf_(
                &m,
                &n,
                q.as_mut_ptr(),
                &one,
                &one,
                desc.as_ptr(),
                tau.as_mut_ptr(),
                work_query.as_mut_ptr(),
                &query,
                &mut info,
            );
        }
        result(info)?;

        let lwork = int(work_query[0] as usize);
        let mut work = vec![0.0; lwork as usize];
        unsafe {
            pdgeqrf_(
                &m,
                &n,
                q.as_mut_ptr(),
                &one,
                &one,
                desc.as_ptr(),
                tau.as_mut_ptr(),
                work.as_mut_ptr(),
                &lwork,
                &mut info,
            );
        }
        result(info)?;

        let r = q.clone();
        work_query[0] = 0.0;
        unsafe {
            pdorgqr_(
                &m,
                &k,
                &k,
                q.as_mut_ptr(),
                &one,
                &one,
                desc.as_ptr(),
                tau.as_ptr(),
                work_query.as_mut_ptr(),
                &query,
                &mut info,
            );
        }
        result(info)?;

        let lwork = int(work_query[0] as usize);
        let mut work = vec![0.0; lwork as usize];
        unsafe {
            pdorgqr_(
                &m,
                &k,
                &k,
                q.as_mut_ptr(),
                &one,
                &one,
                desc.as_ptr(),
                tau.as_ptr(),
                work.as_mut_ptr(),
                &lwork,
                &mut info,
            );
        }
        result(info).map(|()| (q, r))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn svd(
        &self,
        m: usize,
        n: usize,
        a: &[f64],
        desc_a: &[i32; 9],
        u: &mut [f64],
        desc_u: &[i32; 9],
        vt: &mut [f64],
        desc_vt: &[i32; 9],
    ) -> Result<Vec<f64>, i32> {
        self.validate_matrix(desc_a, a.len());
        self.validate_matrix(desc_u, u.len());
        self.validate_matrix(desc_vt, vt.len());
        let (m, n) = (int(m), int(n));
        let k = m.min(n);
        assert!(m <= desc_a[2] && n <= desc_a[3]);
        assert!(m <= desc_u[2] && k <= desc_u[3]);
        assert!(k <= desc_vt[2] && n <= desc_vt[3]);

        let mut a = a.to_vec();
        let mut s = vec![0.0; k as usize];
        let vectors = b'V' as c_char;
        let one = 1;
        let query = -1;
        let mut work_query = [0.0];
        let mut info = 0;
        unsafe {
            pdgesvd_(
                &vectors,
                &vectors,
                &m,
                &n,
                a.as_mut_ptr(),
                &one,
                &one,
                desc_a.as_ptr(),
                s.as_mut_ptr(),
                u.as_mut_ptr(),
                &one,
                &one,
                desc_u.as_ptr(),
                vt.as_mut_ptr(),
                &one,
                &one,
                desc_vt.as_ptr(),
                work_query.as_mut_ptr(),
                &query,
                &mut info,
            );
        }
        result(info)?;

        let lwork = int(work_query[0] as usize);
        let mut work = vec![0.0; lwork as usize];
        unsafe {
            pdgesvd_(
                &vectors,
                &vectors,
                &m,
                &n,
                a.as_mut_ptr(),
                &one,
                &one,
                desc_a.as_ptr(),
                s.as_mut_ptr(),
                u.as_mut_ptr(),
                &one,
                &one,
                desc_u.as_ptr(),
                vt.as_mut_ptr(),
                &one,
                &one,
                desc_vt.as_ptr(),
                work.as_mut_ptr(),
                &lwork,
                &mut info,
            );
        }
        result(info).map(|()| s)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn solve_tri(
        &self,
        m: usize,
        n: usize,
        factor: &[f64],
        desc_factor: &[i32; 9],
        b: &mut [f64],
        desc_b: &[i32; 9],
        lower: bool,
        from_left: bool,
        transpose: bool,
    ) {
        self.validate_matrix(desc_factor, factor.len());
        self.validate_matrix(desc_b, b.len());
        let factor_order = if from_left { m } else { n };
        assert!(int(factor_order) <= desc_factor[2] && int(factor_order) <= desc_factor[3]);
        assert!(int(m) <= desc_b[2] && int(n) <= desc_b[3]);

        let side = if from_left { b'L' } else { b'R' } as c_char;
        let uplo = if lower { b'L' } else { b'U' } as c_char;
        let transpose = if transpose { b'T' } else { b'N' } as c_char;
        let diagonal = b'N' as c_char;
        let (m, n) = (int(m), int(n));
        let one = 1;
        let alpha = 1.0;
        unsafe {
            pdtrsm_(
                &side,
                &uplo,
                &transpose,
                &diagonal,
                &m,
                &n,
                &alpha,
                factor.as_ptr(),
                &one,
                &one,
                desc_factor.as_ptr(),
                b.as_mut_ptr(),
                &one,
                &one,
                desc_b.as_ptr(),
            );
        }
    }

    pub(crate) fn close(self) {
        unsafe {
            Cblacs_gridexit(self.context);
            Cfree_blacs_system_handle(self.system_context);
        }
    }

    fn validate_matrix(&self, desc: &[i32; 9], length: usize) {
        assert_eq!(desc[0], 1);
        assert_eq!(desc[1], self.context);
        assert!(desc[2] >= 0 && desc[3] >= 0);
        assert!(desc[4] > 0 && desc[5] > 0);
        assert!(desc[6] >= 0 && desc[6] < int(self.rows));
        assert!(desc[7] >= 0 && desc[7] < int(self.cols));
        assert!(desc[8] > 0);

        let local_rows = numroc(
            desc[2] as usize,
            desc[4] as usize,
            self.row,
            desc[6] as usize,
            self.rows,
        );
        let local_cols = numroc(
            desc[3] as usize,
            desc[5] as usize,
            self.col,
            desc[7] as usize,
            self.cols,
        );
        let lld = desc[8] as usize;
        assert!(lld >= local_rows.max(1));
        assert!(length >= lld.checked_mul(local_cols).unwrap());
    }
}
