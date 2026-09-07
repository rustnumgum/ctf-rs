//! The only module allowed to use MPI's raw handles and native entry points.
use crate::algebra::{Monoid, Wire};
use mpi_sys as sys;
#[path = "allgather.rs"]
mod allgather;
#[path = "gather.rs"]
mod gather;
#[path = "binary_io.rs"]
mod binary_io;
#[path = "mpi_io.rs"]
mod mpi_io;
#[cfg(feature = "native-linalg")]
#[path = "factor_mpi.rs"]
mod factor_mpi;
use std::{marker::PhantomData, rc::Rc};
std::thread_local! {
    static ACTIVE_ALGEBRA: std::cell::Cell<*const std::ffi::c_void> = const {std::cell::Cell::new(std::ptr::null())};
}

// MPI_Init uses SINGLE; callback state is borrowed only for the blocking call.
unsafe extern "C" fn monoid_add<A: Monoid>(
    input: *mut std::ffi::c_void,
    output: *mut std::ffi::c_void,
    count: *mut i32,
    _datatype: *mut sys::MPI_Datatype,
) where
    A::Element: Wire,
{
    ACTIVE_ALGEBRA.with(|active| {
        let algebra = unsafe {&*active.get().cast::<A>()};
        let count = unsafe {*count} as usize;
        let left =
            unsafe { std::slice::from_raw_parts(input.cast::<u8>(), count * A::Element::WIDTH) };
        let right = unsafe {
            std::slice::from_raw_parts_mut(output.cast::<u8>(), count * A::Element::WIDTH)
        };
        for (a, b) in left
            .chunks_exact(A::Element::WIDTH)
            .zip(right.chunks_exact_mut(A::Element::WIDTH))
        {
            let value = algebra.add(&A::Element::decode(a),&A::Element::decode(b));
            let mut bytes = Vec::with_capacity(A::Element::WIDTH);
            value.encode(&mut bytes);
            b.copy_from_slice(&bytes);
        }
    });
}

pub(crate) struct Runtime {
    _single_thread: PhantomData<Rc<()>>,
}
pub(crate) struct Comm {
    raw: sys::MPI_Comm,
    owned: bool,
    _single_thread: PhantomData<Rc<()>>,
}

#[derive(Clone, Copy)]
pub(crate) struct Transfer {
    pub peer: usize,
    pub displacement: usize,
    pub count: usize,
}

fn check(code: i32) {
    assert_eq!(code, 0, "MPI error {code}");
}

impl Runtime {
    pub(crate) fn initialize() -> Self {
        unsafe {
            let mut initialized = 0;
            check(sys::MPI_Initialized(&mut initialized));
            assert_eq!(initialized, 0, "MPI already initialized");
            check(sys::MPI_Init(std::ptr::null_mut(), std::ptr::null_mut()));
        }
        Self {
            _single_thread: PhantomData,
        }
    }
    pub(crate) fn world(&self) -> Comm {
        Comm {
            raw: unsafe { sys::RSMPI_COMM_WORLD },
            owned: false,
            _single_thread: PhantomData,
        }
    }
    pub(crate) fn finalize(self) {
        unsafe {
            check(sys::MPI_Finalize());
        }
    }
}

impl Comm {
    /// DGTOG root-to-root exchange: post every receive, post every send, then
    /// wait for all requests. Displacements and counts are measured in bytes.
    pub(crate) fn redistribute_ror(
        &self,
        send_buffer: &[u8],
        sends: &[Transfer],
        receive_buffer: &mut [u8],
        receives: &[Transfer],
    ) {
        for transfer in sends.iter().chain(receives) {
            assert!(transfer.peer < self.size());
        }
        for transfer in sends {
            assert!(transfer.displacement + transfer.count <= send_buffer.len());
        }
        for transfer in receives.iter() {
            assert!(transfer.displacement + transfer.count <= receive_buffer.len());
        }

        let request_count = sends.iter().filter(|transfer| transfer.count != 0).count()
            + receives.iter().filter(|transfer| transfer.count != 0).count();
        let mut requests = Vec::with_capacity(request_count);
        for transfer in receives.iter().filter(|transfer| transfer.count != 0) {
            let mut request = unsafe { sys::RSMPI_REQUEST_NULL };
            unsafe {
                check(sys::MPI_Irecv(
                    receive_buffer.as_mut_ptr().add(transfer.displacement).cast(),
                    transfer.count.try_into().unwrap(),
                    sys::RSMPI_UINT8_T,
                    transfer.peer as i32,
                    777,
                    self.raw,
                    &mut request,
                ));
            }
            requests.push(request);
        }
        for transfer in sends.iter().filter(|transfer| transfer.count != 0) {
            let mut request = unsafe { sys::RSMPI_REQUEST_NULL };
            unsafe {
                check(sys::MPI_Isend(
                    send_buffer.as_ptr().add(transfer.displacement).cast(),
                    transfer.count.try_into().unwrap(),
                    sys::RSMPI_UINT8_T,
                    transfer.peer as i32,
                    777,
                    self.raw,
                    &mut request,
                ));
            }
            requests.push(request);
        }
        if !requests.is_empty() {
            let mut statuses: Vec<sys::MPI_Status> = (0..requests.len())
                .map(|_| unsafe { std::mem::zeroed() })
                .collect();
            unsafe {
                check(sys::MPI_Waitall(
                    requests.len().try_into().unwrap(),
                    requests.as_mut_ptr(),
                    statuses.as_mut_ptr(),
                ));
            }
        }
    }

    pub(crate) fn all_reduce_monoid<A: Monoid>(
        &self,
        algebra: &A,
        values: &mut [A::Element],
        commutative: bool,
    ) where
        A::Element: Wire,
    {
        self.reduce_monoid(algebra, values, commutative, None);
    }
    pub(crate) fn reduce_monoid<A: Monoid>(
        &self,
        algebra: &A,
        values: &mut [A::Element],
        commutative: bool,
        root: Option<usize>,
    ) where
        A::Element: Wire,
    {
        if let Some(root) = root {
            assert!(root < self.size());
        }
        assert!(A::Element::WIDTH > 0);
        let mut input = Vec::with_capacity(values.len()*A::Element::WIDTH);
        for value in values.iter() {
            value.encode(&mut input);
        }
        assert_eq!(
            input.len(),
            values.len() * A::Element::WIDTH,
            "Wire encoding must match its fixed width before MPI reads the buffer"
        );
        let mut output = vec![0u8;input.len().max(1)];
        if input.is_empty() {
            input.push(0);
        }
        ACTIVE_ALGEBRA.with(|active| {
            assert!(
                active.get().is_null(),
                "nested MPI user reduction is not supported by MPI callbacks"
            );
            active.set((algebra as *const A).cast());
            unsafe {
                let mut datatype = sys::RSMPI_DATATYPE_NULL;
                check(sys::MPI_Type_contiguous(
                    A::Element::WIDTH.try_into().unwrap(),
                    sys::RSMPI_UINT8_T,
                    &mut datatype,
                ));
                check(sys::MPI_Type_commit(&mut datatype));
                let mut operation = std::mem::zeroed();
                check(sys::MPI_Op_create(
                    Some(monoid_add::<A>),
                    i32::from(commutative),
                    &mut operation,
                ));
                if let Some(root)=root {
                    check(sys::MPI_Reduce(
                        input.as_ptr().cast(),
                        output.as_mut_ptr().cast(),
                        values.len().try_into().unwrap(),
                        datatype,
                        operation,
                        root as i32,
                        self.raw,
                    ));
                } else {
                    check(sys::MPI_Allreduce(
                        input.as_ptr().cast(),
                        output.as_mut_ptr().cast(),
                        values.len().try_into().unwrap(),
                        datatype,
                        operation,
                        self.raw,
                    ));
                }
                check(sys::MPI_Op_free(&mut operation));
                check(sys::MPI_Type_free(&mut datatype));
            }
            active.set(std::ptr::null());
        });
        if root.is_none() || root==Some(self.rank()) {
            for (value, bytes) in values
                .iter_mut()
                .zip(output.chunks_exact(A::Element::WIDTH))
            {
                *value = A::Element::decode(bytes);
            }
        }
    }
    pub(crate) fn rank(&self) -> usize {
        let mut rank = 0;
        unsafe {
            check(sys::MPI_Comm_rank(self.raw, &mut rank));
        }
        rank as usize
    }
    pub(crate) fn size(&self) -> usize {
        let mut size = 0;
        unsafe {
            check(sys::MPI_Comm_size(self.raw, &mut size));
        }
        size as usize
    }
    pub(crate) fn split(&self, color: Option<i32>, key: i32) -> Option<Self> {
        unsafe {
            let mut raw = sys::RSMPI_COMM_NULL;
            check(sys::MPI_Comm_split(
                self.raw,
                color.unwrap_or(sys::RSMPI_UNDEFINED),
                key,
                &mut raw,
            ));
            (raw != sys::RSMPI_COMM_NULL).then_some(Self {
                raw,
                owned: true,
                _single_thread: PhantomData,
            })
        }
    }
    pub(crate) fn split_shared(&self) -> Self {
        unsafe {
            let mut raw = sys::RSMPI_COMM_NULL;
            check(sys::MPI_Comm_split_type(
                self.raw,
                sys::RSMPI_COMM_TYPE_SHARED,
                self.rank() as i32,
                sys::RSMPI_INFO_NULL,
                &mut raw,
            ));
            Self {
                raw,
                owned: true,
                _single_thread: PhantomData,
            }
        }
    }
    pub(crate) fn close(mut self) {
        if self.owned {
            unsafe {
                check(sys::MPI_Comm_free(&mut self.raw));
            }
        }
    }
    pub(crate) fn barrier(&self) {
        unsafe {
            check(sys::MPI_Barrier(self.raw));
        }
    }
    pub(crate) fn reduce_f64(&self, root: usize, values: &mut [f64]) {
        assert!(root < self.size());
        let mut input = values.to_vec();
        if input.is_empty() {
            input.push(0.);
        }
        let mut output = vec![0.;values.len().max(1)];
        unsafe {
            check(sys::MPI_Reduce(
                input.as_ptr().cast(),
                output.as_mut_ptr().cast(),
                values.len().try_into().unwrap(),
                sys::RSMPI_DOUBLE,
                sys::RSMPI_SUM,
                root as i32,
                self.raw,
            ));
        }
        if self.rank() == root {
            values.copy_from_slice(&output[..values.len()]);
        }
    }
    pub(crate) fn sum_f64(&self, values: &mut [f64]) {
        let input = values.to_vec();
        if values.is_empty() {
            // Distinct valid addresses for zero-count MPI calls.
            let input = 0.0f64;
            let mut output = 0.0f64;
            unsafe {
                check(sys::MPI_Allreduce(
                    (&input as *const f64).cast(),
                    (&mut output as *mut f64).cast(),
                    0,
                    sys::RSMPI_DOUBLE,
                    sys::RSMPI_SUM,
                    self.raw,
                ));
            }
        } else {
            unsafe {
                check(sys::MPI_Allreduce(
                    input.as_ptr().cast(),
                    values.as_mut_ptr().cast(),
                    values.len().try_into().unwrap(),
                    sys::RSMPI_DOUBLE,
                    sys::RSMPI_SUM,
                    self.raw,
                ));
            }
        }
    }
    pub(crate) fn all_gather_f64(&self, values:&[f64])->Vec<f64> {
        let count=values.len().try_into().unwrap();
        let mut output=vec![0.;(values.len()*self.size()).max(1)];
        let empty = 0.0f64;
        let input = if values.is_empty() {
            &empty as *const f64
        } else {
            values.as_ptr()
        };
        unsafe {
            check(sys::MPI_Allgather(
                input.cast(),
                count,
                sys::RSMPI_DOUBLE,
                output.as_mut_ptr().cast(),
                count,
                sys::RSMPI_DOUBLE,
                self.raw,
            ));
        }
        output.truncate(values.len() * self.size());
        output
    }
    pub(crate) fn all_gather_i32(&self,value:i32)->Vec<i32> {
        let mut output=vec![0;self.size()];
        unsafe {
            check(sys::MPI_Allgather(
                (&value as *const i32).cast(),
                1,
                sys::RSMPI_INT32_T,
                output.as_mut_ptr().cast(),
                1,
                sys::RSMPI_INT32_T,
                self.raw,
            ));
        }
        output
    }
    #[cfg(feature = "native-scalapack")]
    pub(crate) fn scalapack_grid(&self,rows:usize,cols:usize)->super::scalapack::Grid {
        assert_eq!(rows * cols, self.size());
        super::scalapack::Grid::new(self.raw, rows, cols)
    }
    pub(crate) fn gather_plan_cost(&self,seconds:f64,memory:i64)->(Vec<f64>,Vec<i64>) {
        let mut times = vec![0.; self.size()];
        let mut bytes = vec![0i64; self.size()];
        unsafe {
            check(sys::MPI_Gather(
                (&seconds as *const f64).cast(),
                1,
                sys::RSMPI_DOUBLE,
                times.as_mut_ptr().cast(),
                1,
                sys::RSMPI_DOUBLE,
                0,
                self.raw,
            ));
            check(sys::MPI_Gather(
                (&memory as *const i64).cast(),
                1,
                sys::RSMPI_INT64_T,
                bytes.as_mut_ptr().cast(),
                1,
                sys::RSMPI_INT64_T,
                0,
                self.raw,
            ));
        }
        (times,bytes)
    }
    pub(crate) fn send_receive(
        &self,
        send: &[u8],
        destination: usize,
        source: usize,
        recv: &mut [u8],
    ) {
        assert!(destination < self.size() && source < self.size());
        unsafe {
            check(sys::MPI_Sendrecv(
                send.as_ptr().cast(),
                send.len().try_into().unwrap(),
                sys::RSMPI_UINT8_T,
                destination as i32,
                9,
                recv.as_mut_ptr().cast(),
                recv.len().try_into().unwrap(),
                sys::RSMPI_UINT8_T,
                source as i32,
                9,
                self.raw,
                sys::RSMPI_STATUS_IGNORE,
            ));
        }
    }
    pub(crate) fn replace_wire<T: Wire>(
        &self,
        values: &mut [T],
        destination: usize,
        source: usize,
        tag: i32,
    ) {
        assert!(destination < self.size() && source < self.size());
        assert!(T::WIDTH > 0);
        let mut bytes = Vec::with_capacity(values.len() * T::WIDTH);
        for value in values.iter() {
            value.encode(&mut bytes);
        }
        assert_eq!(bytes.len(), values.len() * T::WIDTH);
        unsafe {
            check(sys::MPI_Sendrecv_replace(
                bytes.as_mut_ptr().cast(),
                bytes.len().try_into().unwrap(),
                sys::RSMPI_UINT8_T,
                destination as i32,
                tag,
                source as i32,
                tag,
                self.raw,
                sys::RSMPI_STATUS_IGNORE,
            ));
        }
        for (value, encoded) in values.iter_mut().zip(bytes.chunks_exact(T::WIDTH)) {
            *value = T::decode(encoded);
        }
    }
    pub(crate) fn broadcast(&self, root: usize, buffer: &mut [u8]) {
        assert!(root < self.size());
        let mut empty = 0u8;
        let pointer = if buffer.is_empty() {
            &mut empty as *mut u8
        } else {
            buffer.as_mut_ptr()
        };
        unsafe {
            check(sys::MPI_Bcast(
                pointer.cast(),
                buffer.len().try_into().unwrap(),
                sys::RSMPI_UINT8_T,
                root as i32,
                self.raw,
            ));
        }
    }
    /// Rank-bucket exchange: Alltoall counts, then Alltoallv payloads.
    pub(crate) fn exchange(&self, buckets: &[Vec<u8>]) -> Vec<Vec<u8>> {
        let n = self.size();
        assert_eq!(buckets.len(), n);
        let send_counts: Vec<i32> = buckets
            .iter()
            .map(|b| b.len().try_into().unwrap())
            .collect();
        let mut recv_counts = vec![0i32; n];
        unsafe {
            check(sys::MPI_Alltoall(
                send_counts.as_ptr().cast(),
                1,
                sys::RSMPI_INT32_T,
                recv_counts.as_mut_ptr().cast(),
                1,
                sys::RSMPI_INT32_T,
                self.raw,
            ));
        }
        fn offsets(counts: &[i32]) -> (Vec<i32>, usize) {
            let mut next = 0i32;
            let offsets = counts
                .iter()
                .map(|&c| {
                    let p = next;
                    next = next.checked_add(c).unwrap();
                    p
                })
                .collect();
            (offsets, next as usize)
        }
        let (send_offsets, _) = offsets(&send_counts);
        let (recv_offsets, total) = offsets(&recv_counts);
        let mut send: Vec<u8> = buckets.iter().flatten().copied().collect();
        // Distinct allocations even for zero counts: empty Vecs share a dangling
        // pointer, which MPI implementations can reject as aliased buffers.
        if send.is_empty() {
            send.push(0);
        }
        let mut recv = vec![0u8; total.max(1)];
        unsafe {
            check(sys::MPI_Alltoallv(
                send.as_ptr().cast(),
                send_counts.as_ptr(),
                send_offsets.as_ptr(),
                sys::RSMPI_UINT8_T,
                recv.as_mut_ptr().cast(),
                recv_counts.as_ptr(),
                recv_offsets.as_ptr(),
                sys::RSMPI_UINT8_T,
                self.raw,
            ));
        }
        recv_offsets
            .iter()
            .zip(recv_counts)
            .map(|(&p, n)| recv[p as usize..(p + n) as usize].to_vec())
            .collect()
    }
}
