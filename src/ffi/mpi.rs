//! The only module allowed to use MPI's raw handles and native entry points.
use mpi_sys as sys;
use std::{marker::PhantomData, rc::Rc};

pub(crate) struct Runtime { _single_thread: PhantomData<Rc<()>> }
pub(crate) struct Comm { raw: sys::MPI_Comm, owned: bool, _single_thread: PhantomData<Rc<()>> }

fn check(code: i32) { assert_eq!(code, 0, "MPI error {code}"); }

impl Runtime {
    pub(crate) fn initialize() -> Self {
        unsafe {
            let mut initialized = 0;
            check(sys::MPI_Initialized(&mut initialized));
            assert_eq!(initialized, 0, "MPI already initialized");
            check(sys::MPI_Init(std::ptr::null_mut(), std::ptr::null_mut()));
        }
        Self { _single_thread: PhantomData }
    }
    pub(crate) fn world(&self) -> Comm {
        Comm { raw: unsafe { sys::RSMPI_COMM_WORLD }, owned: false, _single_thread: PhantomData }
    }
    pub(crate) fn finalize(self) { unsafe { check(sys::MPI_Finalize()); } }
}

impl Comm {
    pub(crate) fn rank(&self) -> usize {
        let mut rank = 0;
        unsafe { check(sys::MPI_Comm_rank(self.raw, &mut rank)); }
        rank as usize
    }
    pub(crate) fn size(&self) -> usize {
        let mut size = 0;
        unsafe { check(sys::MPI_Comm_size(self.raw, &mut size)); }
        size as usize
    }
    pub(crate) fn split(&self, color: Option<i32>, key: i32) -> Option<Self> {
        unsafe {
            let mut raw = sys::RSMPI_COMM_NULL;
            check(sys::MPI_Comm_split(self.raw, color.unwrap_or(sys::RSMPI_UNDEFINED), key, &mut raw));
            (raw != sys::RSMPI_COMM_NULL).then_some(Self { raw, owned: true, _single_thread: PhantomData })
        }
    }
    pub(crate) fn split_shared(&self) -> Self {
        unsafe {
            let mut raw = sys::RSMPI_COMM_NULL;
            check(sys::MPI_Comm_split_type(self.raw, sys::RSMPI_COMM_TYPE_SHARED, self.rank() as i32, sys::RSMPI_INFO_NULL, &mut raw));
            Self { raw, owned: true, _single_thread: PhantomData }
        }
    }
    pub(crate) fn close(mut self) {
        if self.owned { unsafe { check(sys::MPI_Comm_free(&mut self.raw)); } }
    }
    pub(crate) fn barrier(&self) { unsafe { check(sys::MPI_Barrier(self.raw)); } }
    pub(crate) fn sum_f64(&self, values: &mut [f64]) {
        let input = values.to_vec();
        if values.is_empty() {
            // Distinct valid addresses for zero-count MPI calls.
            let input = 0.0f64; let mut output = 0.0f64;
            unsafe { check(sys::MPI_Allreduce((&input as *const f64).cast(), (&mut output as *mut f64).cast(),
                0,sys::RSMPI_DOUBLE,sys::RSMPI_SUM,self.raw)); }
        } else {
            unsafe { check(sys::MPI_Allreduce(input.as_ptr().cast(),values.as_mut_ptr().cast(),
                values.len().try_into().unwrap(),sys::RSMPI_DOUBLE,sys::RSMPI_SUM,self.raw)); }
        }
    }
    pub(crate) fn send_receive(&self, send: &[u8], destination: usize, source: usize, recv: &mut [u8]) {
        assert!(destination < self.size() && source < self.size());
        unsafe { check(sys::MPI_Sendrecv(send.as_ptr().cast(), send.len().try_into().unwrap(), sys::RSMPI_UINT8_T,
            destination as i32, 9, recv.as_mut_ptr().cast(), recv.len().try_into().unwrap(), sys::RSMPI_UINT8_T,
            source as i32, 9, self.raw, sys::RSMPI_STATUS_IGNORE)); }
    }
    pub(crate) fn broadcast(&self, root: usize, buffer: &mut [u8]) {
        assert!(root < self.size());
        unsafe { check(sys::MPI_Bcast(buffer.as_mut_ptr().cast(), buffer.len().try_into().unwrap(), sys::RSMPI_UINT8_T, root as i32, self.raw)); }
    }
    /// Rank-bucket exchange: Alltoall counts, then Alltoallv payloads.
    pub(crate) fn exchange(&self, buckets: &[Vec<u8>]) -> Vec<Vec<u8>> {
        let n = self.size();
        assert_eq!(buckets.len(), n);
        let send_counts: Vec<i32> = buckets.iter().map(|b| b.len().try_into().unwrap()).collect();
        let mut recv_counts = vec![0i32; n];
        unsafe { check(sys::MPI_Alltoall(send_counts.as_ptr().cast(), 1, sys::RSMPI_INT32_T,
            recv_counts.as_mut_ptr().cast(), 1, sys::RSMPI_INT32_T, self.raw)); }
        fn offsets(counts: &[i32]) -> (Vec<i32>, usize) {
            let mut next = 0i32;
            let offsets = counts.iter().map(|&c| { let p = next; next = next.checked_add(c).unwrap(); p }).collect();
            (offsets, next as usize)
        }
        let (send_offsets, _) = offsets(&send_counts);
        let (recv_offsets, total) = offsets(&recv_counts);
        let mut send: Vec<u8> = buckets.iter().flatten().copied().collect();
        // Distinct allocations even for zero counts: empty Vecs share a dangling
        // pointer, which MPI implementations can reject as aliased buffers.
        if send.is_empty() { send.push(0); }
        let mut recv = vec![0u8; total.max(1)];
        unsafe { check(sys::MPI_Alltoallv(send.as_ptr().cast(), send_counts.as_ptr(), send_offsets.as_ptr(), sys::RSMPI_UINT8_T,
            recv.as_mut_ptr().cast(), recv_counts.as_ptr(), recv_offsets.as_ptr(), sys::RSMPI_UINT8_T, self.raw)); }
        recv_offsets.iter().zip(recv_counts).map(|(&p, n)| recv[p as usize..(p+n) as usize].to_vec()).collect()
    }
}
