//! The only module allowed to use MPI's raw handles and native entry points.
use crate::algebra::{Monoid, Wire};
use ::mpi::{
    collective::{CommunicatorCollectives, Root, SystemOperation, UnsafeUserOperation},
    datatype::{Equivalence, MutView, Partition, PartitionMut, UserDatatype, View},
    ffi as sys,
    point_to_point::{send_receive_into_with_tags, send_receive_replace_into_with_tags},
    topology::{Color, Communicator, SimpleCommunicator},
};
#[path = "allgather.rs"]
mod allgather;
#[path = "binary_io.rs"]
mod binary_io;
#[cfg(feature = "native-linalg")]
#[path = "factor_mpi.rs"]
mod factor_mpi;
#[path = "gather.rs"]
mod gather;
#[path = "mpi_io.rs"]
mod mpi_io;
use std::{marker::PhantomData, mem::ManuallyDrop, rc::Rc};

std::thread_local! {
    static ACTIVE_ALGEBRA: std::cell::Cell<*const std::ffi::c_void> = const {std::cell::Cell::new(std::ptr::null())};
}

// Callback state is borrowed only for the blocking collective call.
unsafe extern "C" fn monoid_add<A: Monoid>(
    input: *mut std::ffi::c_void,
    output: *mut std::ffi::c_void,
    count: *mut i32,
    _datatype: *mut sys::MPI_Datatype,
) where
    A::Element: Wire,
{
    ACTIVE_ALGEBRA.with(|active| {
        let algebra = unsafe { &*active.get().cast::<A>() };
        let count = unsafe { *count } as usize;
        let left =
            unsafe { std::slice::from_raw_parts(input.cast::<u8>(), count * A::Element::WIDTH) };
        let right = unsafe {
            std::slice::from_raw_parts_mut(output.cast::<u8>(), count * A::Element::WIDTH)
        };
        for (a, b) in left
            .chunks_exact(A::Element::WIDTH)
            .zip(right.chunks_exact_mut(A::Element::WIDTH))
        {
            let value = algebra.add(&A::Element::decode(a), &A::Element::decode(b));
            let mut bytes = Vec::with_capacity(A::Element::WIDTH);
            value.encode(&mut bytes);
            b.copy_from_slice(&bytes);
        }
    });
}

enum CommunicatorStorage<'a> {
    Borrowed(&'a dyn Communicator),
    World(SimpleCommunicator),
    Split(ManuallyDrop<SimpleCommunicator>),
}

pub(crate) struct Comm<'a> {
    communicator: CommunicatorStorage<'a>,
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

impl<'a> Comm<'a> {
    fn split_communicator(communicator: SimpleCommunicator) -> Self {
        Self {
            communicator: CommunicatorStorage::Split(ManuallyDrop::new(communicator)),
            _single_thread: PhantomData,
        }
    }

    pub(crate) fn world() -> Self {
        Self {
            communicator: CommunicatorStorage::World(SimpleCommunicator::world()),
            _single_thread: PhantomData,
        }
    }

    pub(crate) fn from_communicator(communicator: &'a impl Communicator) -> Self {
        Self {
            communicator: CommunicatorStorage::Borrowed(communicator),
            _single_thread: PhantomData,
        }
    }

    fn communicator(&self) -> &dyn Communicator {
        match &self.communicator {
            CommunicatorStorage::Borrowed(communicator) => *communicator,
            CommunicatorStorage::World(communicator) => communicator,
            CommunicatorStorage::Split(communicator) => &**communicator,
        }
    }

    pub(crate) fn raw(&self) -> sys::MPI_Comm {
        self.communicator().as_raw()
    }

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
            + receives
                .iter()
                .filter(|transfer| transfer.count != 0)
                .count();
        let mut requests = Vec::with_capacity(request_count);
        for transfer in receives.iter().filter(|transfer| transfer.count != 0) {
            let mut request = unsafe { sys::RSMPI_REQUEST_NULL };
            unsafe {
                check(sys::MPI_Irecv(
                    receive_buffer
                        .as_mut_ptr()
                        .add(transfer.displacement)
                        .cast(),
                    transfer.count.try_into().unwrap(),
                    sys::RSMPI_UINT8_T,
                    transfer.peer as i32,
                    777,
                    self.raw(),
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
                    self.raw(),
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
        let mut input = Vec::with_capacity(values.len() * A::Element::WIDTH);
        for value in values.iter() {
            value.encode(&mut input);
        }
        assert_eq!(
            input.len(),
            values.len() * A::Element::WIDTH,
            "Wire encoding must match its fixed width before MPI reads the buffer"
        );
        let mut output = vec![0u8; input.len().max(1)];
        if input.is_empty() {
            input.push(0);
        }
        let datatype = UserDatatype::contiguous(
            A::Element::WIDTH.try_into().unwrap(),
            &u8::equivalent_datatype(),
        );
        let input = unsafe {
            View::with_count_and_datatype(&input[..], values.len().try_into().unwrap(), &datatype)
        };
        let mut output_view = unsafe {
            MutView::with_count_and_datatype(
                &mut output[..],
                values.len().try_into().unwrap(),
                &datatype,
            )
        };
        ACTIVE_ALGEBRA.with(|active| {
            assert!(
                active.get().is_null(),
                "nested MPI user reduction is not supported by MPI callbacks"
            );
            active.set((algebra as *const A).cast());
            let operation = unsafe {
                if commutative {
                    UnsafeUserOperation::commutative(monoid_add::<A>)
                } else {
                    UnsafeUserOperation::associative(monoid_add::<A>)
                }
            };
            if let Some(root) = root {
                let root_process = self.communicator().process_at_rank(root as i32);
                if root == self.rank() {
                    root_process.reduce_into_root(&input, &mut output_view, &operation);
                } else {
                    root_process.reduce_into(&input, &operation);
                }
            } else {
                self.communicator()
                    .all_reduce_into(&input, &mut output_view, &operation);
            }
            active.set(std::ptr::null());
        });
        drop(output_view);
        if root.is_none() || root == Some(self.rank()) {
            for (value, bytes) in values
                .iter_mut()
                .zip(output.chunks_exact(A::Element::WIDTH))
            {
                *value = A::Element::decode(bytes);
            }
        }
    }

    pub(crate) fn rank(&self) -> usize {
        self.communicator().rank() as usize
    }

    pub(crate) fn size(&self) -> usize {
        self.communicator().size() as usize
    }

    pub(crate) fn split(&self, color: Option<i32>, key: i32) -> Option<Self> {
        let color = color.map_or_else(Color::undefined, Color::with_value);
        self.communicator()
            .split_by_color_with_key(color, key)
            .map(Self::split_communicator)
    }

    pub(crate) fn split_shared(&self) -> Self {
        Self::split_communicator(self.communicator().split_shared(self.rank() as i32))
    }

    pub(crate) fn close(mut self) {
        if let CommunicatorStorage::Split(communicator) = &mut self.communicator {
            unsafe {
                ManuallyDrop::drop(communicator);
            }
        }
    }

    pub(crate) fn barrier(&self) {
        self.communicator().barrier();
    }

    pub(crate) fn reduce_f64(&self, root: usize, values: &mut [f64]) {
        assert!(root < self.size());
        let mut input = values.to_vec();
        if input.is_empty() {
            input.push(0.0);
        }
        let mut output = vec![0.0; values.len().max(1)];
        let datatype = f64::equivalent_datatype();
        let input = unsafe {
            View::with_count_and_datatype(&input[..], values.len().try_into().unwrap(), &datatype)
        };
        let mut output_view = unsafe {
            MutView::with_count_and_datatype(
                &mut output[..],
                values.len().try_into().unwrap(),
                &datatype,
            )
        };
        let root_process = self.communicator().process_at_rank(root as i32);
        if root == self.rank() {
            root_process.reduce_into_root(&input, &mut output_view, SystemOperation::sum());
            drop(output_view);
            values.copy_from_slice(&output[..values.len()]);
        } else {
            root_process.reduce_into(&input, SystemOperation::sum());
        }
    }

    pub(crate) fn sum_f64(&self, values: &mut [f64]) {
        let mut input = values.to_vec();
        if input.is_empty() {
            input.push(0.0);
        }
        if values.is_empty() {
            let mut output = [0.0];
            let datatype = f64::equivalent_datatype();
            let input = unsafe { View::with_count_and_datatype(&input[..], 0, &datatype) };
            let mut output =
                unsafe { MutView::with_count_and_datatype(&mut output[..], 0, &datatype) };
            self.communicator()
                .all_reduce_into(&input, &mut output, SystemOperation::sum());
        } else {
            self.communicator()
                .all_reduce_into(&input[..], values, SystemOperation::sum());
        }
    }

    pub(crate) fn all_gather_f64(&self, values: &[f64]) -> Vec<f64> {
        let mut input = values.to_vec();
        if input.is_empty() {
            input.push(0.0);
        }
        let output_len = values.len() * self.size();
        let mut output = vec![0.0; output_len.max(1)];
        if values.is_empty() {
            let datatype = f64::equivalent_datatype();
            let input = unsafe { View::with_count_and_datatype(&input[..], 0, &datatype) };
            let mut output_view =
                unsafe { MutView::with_count_and_datatype(&mut output[..], 0, &datatype) };
            self.communicator()
                .all_gather_into(&input, &mut output_view);
        } else {
            self.communicator()
                .all_gather_into(values, &mut output[..output_len]);
        }
        output.truncate(output_len);
        output
    }

    pub(crate) fn all_gather_i32(&self, value: i32) -> Vec<i32> {
        let mut output = vec![0; self.size()];
        self.communicator().all_gather_into(&value, &mut output[..]);
        output
    }

    #[cfg(feature = "native-scalapack")]
    pub(crate) fn scalapack_grid(&self, rows: usize, cols: usize) -> super::scalapack::Grid {
        assert_eq!(rows * cols, self.size());
        super::scalapack::Grid::new(self.communicator(), rows, cols)
    }

    pub(crate) fn gather_plan_cost(&self, seconds: f64, memory: i64) -> (Vec<f64>, Vec<i64>) {
        let mut times = vec![0.0; self.size()];
        let mut bytes = vec![0i64; self.size()];
        let root = self.communicator().process_at_rank(0);
        if self.rank() == 0 {
            root.gather_into_root(&seconds, &mut times[..]);
            root.gather_into_root(&memory, &mut bytes[..]);
        } else {
            root.gather_into(&seconds);
            root.gather_into(&memory);
        }
        (times, bytes)
    }

    pub(crate) fn send_receive(
        &self,
        send: &[u8],
        destination: usize,
        source: usize,
        recv: &mut [u8],
    ) {
        assert!(destination < self.size() && source < self.size());
        let send_len = send.len();
        let empty_send = [0u8];
        let send = if send.is_empty() {
            &empty_send[..]
        } else {
            send
        };
        let recv_len = recv.len();
        let mut empty_recv = [0u8];
        let recv = if recv.is_empty() {
            &mut empty_recv[..]
        } else {
            recv
        };
        send_receive_into_with_tags(
            &send[..send_len],
            &self.communicator().process_at_rank(destination as i32),
            9,
            &mut recv[..recv_len],
            &self.communicator().process_at_rank(source as i32),
            9,
        );
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
        let byte_len = bytes.len();
        if bytes.is_empty() {
            bytes.push(0);
        }
        send_receive_replace_into_with_tags(
            &mut bytes[..byte_len],
            &self.communicator().process_at_rank(destination as i32),
            tag,
            &self.communicator().process_at_rank(source as i32),
            tag,
        );
        for (value, encoded) in values.iter_mut().zip(bytes.chunks_exact(T::WIDTH)) {
            *value = T::decode(encoded);
        }
    }

    pub(crate) fn broadcast(&self, root: usize, buffer: &mut [u8]) {
        assert!(root < self.size());
        let len = buffer.len();
        let mut empty = [0u8];
        let buffer = if buffer.is_empty() {
            &mut empty[..]
        } else {
            buffer
        };
        self.communicator()
            .process_at_rank(root as i32)
            .broadcast_into(&mut buffer[..len]);
    }

    /// Rank-bucket exchange: Alltoall counts, then Alltoallv payloads.
    pub(crate) fn exchange(&self, buckets: &[Vec<u8>]) -> Vec<Vec<u8>> {
        let n = self.size();
        assert_eq!(buckets.len(), n);
        let send_counts: Vec<i32> = buckets
            .iter()
            .map(|bucket| bucket.len().try_into().unwrap())
            .collect();
        let mut recv_counts = vec![0i32; n];
        self.communicator()
            .all_to_all_into(&send_counts[..], &mut recv_counts[..]);

        fn offsets(counts: &[i32]) -> (Vec<i32>, usize) {
            let mut next = 0i32;
            let offsets = counts
                .iter()
                .map(|&count| {
                    let offset = next;
                    next = next.checked_add(count).unwrap();
                    offset
                })
                .collect();
            (offsets, next as usize)
        }

        let (send_offsets, send_total) = offsets(&send_counts);
        let (recv_offsets, recv_total) = offsets(&recv_counts);
        let mut send: Vec<u8> = buckets.iter().flatten().copied().collect();
        if send.is_empty() {
            send.push(0);
        }
        let mut recv = vec![0u8; recv_total.max(1)];
        let send = Partition::new(&send[..send_total], &send_counts[..], &send_offsets[..]);
        let mut recv_partition =
            PartitionMut::new(&mut recv[..recv_total], &recv_counts[..], &recv_offsets[..]);
        self.communicator()
            .all_to_all_varcount_into(&send, &mut recv_partition);

        recv_offsets
            .iter()
            .zip(recv_counts)
            .map(|(&offset, count)| recv[offset as usize..(offset + count) as usize].to_vec())
            .collect()
    }
}
