//! Independently written bindings to the ScaLAPACK/BLACS LP64 interfaces.
//!
//! The operation selection and one-based whole-matrix calls follow the pinned
//! CTF reference at f69cbb46e23bc2f39cda5722ce096f56301dab4f
//! (`src/interface/matrix.cxx` and `src/shared/lapack_symbs.cxx`).
use mpi_sys as sys;
use std::{ffi::c_char, marker::PhantomData, rc::Rc};

#[link(name = "scalapack-openmpi")]
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
    pub(crate) fn new(comm: sys::MPI_Comm, rows: usize, cols: usize) -> Self {
        assert!(rows > 0 && cols > 0);
        let processes = rows.checked_mul(cols).unwrap();
        let mut size = 0;
        let mut rank = 0;
        unsafe {
            assert_eq!(
                sys::MPI_Comm_size(comm, &mut size),
                0,
                "MPI_Comm_size failed"
            );
            assert_eq!(
                sys::MPI_Comm_rank(comm, &mut rank),
                0,
                "MPI_Comm_rank failed"
            );
        }
        assert_eq!(processes, usize::try_from(size).unwrap());

        let system_context = unsafe { Csys2blacs_handle(comm) };
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
