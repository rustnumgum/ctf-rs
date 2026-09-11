//! Host-owned MPI lifecycle. Context never communicates on Drop.
use crate::{
    algebra::{Monoid, Wire},
    ffi::mpi,
};

use ::mpi::{Threading, environment::Universe, topology::Communicator};

#[must_use = "call close explicitly for communicators created by collective splits"]
pub struct Context<'u> {
    pub(crate) inner: mpi::Comm<'u>,
    _universe: &'u Universe,
}
impl<'u> Context<'u> {
    /// Borrow the host's MPI universe without taking ownership of MPI_COMM_WORLD.
    pub fn world(universe: &'u Universe) -> Self {
        Self::check_thread();
        Self {
            inner: mpi::Comm::world(),
            _universe: universe,
        }
    }

    /// Borrow a host communicator; its owner remains responsible for freeing it.
    pub fn from_communicator(universe: &'u Universe, communicator: &'u impl Communicator) -> Self {
        Self::check_thread();
        Self {
            inner: mpi::Comm::from_communicator(communicator),
            _universe: universe,
        }
    }

    fn check_thread() {
        let provided = ::mpi::environment::threading_support();
        assert!(
            provided >= Threading::Funneled,
            "ctf requires at least MPI_THREAD_FUNNELED; provided {provided:?}"
        );
        let mut is_main = 0;
        let code = unsafe { ::mpi::ffi::MPI_Is_thread_main(&mut is_main) };
        assert_eq!(
            code, 0,
            "MPI_Is_thread_main error {code}; provided {provided:?}"
        );
        assert_ne!(
            is_main, 0,
            "ctf requires the MPI main thread; provided {provided:?}"
        );
    }
    pub fn rank(&self) -> usize {
        self.inner.rank()
    }
    pub fn size(&self) -> usize {
        self.inner.size()
    }
    pub fn barrier(&self) {
        self.inner.barrier();
    }
    /// Native MPI user operation over explicitly serialized elements. Set
    /// commutative=false when rank order matters. Algebra callbacks must not
    /// invoke MPI or panic. Datatype/operator cleanup is explicit before return.
    pub fn all_reduce_monoid<A: Monoid>(
        &self,
        algebra: &A,
        values: &mut [A::Element],
        commutative: bool,
    ) where
        A::Element: Wire,
    {
        self.inner.all_reduce_monoid(algebra, values, commutative);
    }
    /// Native MPI user reduction to a specified root; nonroots retain contributions.
    pub fn reduce_monoid<A: Monoid>(
        &self,
        algebra: &A,
        values: &mut [A::Element],
        commutative: bool,
        root: usize,
    ) where
        A::Element: Wire,
    {
        self.inner
            .reduce_monoid(algebra, values, commutative, Some(root));
    }
    /// Native MPI sum of one local f64 block. Collective; no global tensor gather.
    pub fn sum_f64(&self, values: &mut [f64]) {
        self.inner.sum_f64(values);
    }
    /// Reduce one local block to root; other ranks retain their local contributions.
    pub fn reduce_f64(&self, root: usize, values: &mut [f64]) {
        self.inner.reduce_f64(root, values);
    }
    #[must_use = "a split communicator leaks unless the caller calls close on it"]
    pub fn split(&self, color: Option<i32>, key: i32) -> Option<Context<'_>> {
        self.inner.split(color, key).map(|inner| Context {
            inner,
            _universe: self._universe,
        })
    }
    #[must_use = "a split communicator leaks unless the caller calls close on it"]
    pub fn split_shared(&self) -> Context<'_> {
        Context {
            inner: self.inner.split_shared(),
            _universe: self._universe,
        }
    }
    pub fn close(self) {
        self.inner.close();
    }
    pub fn broadcast<T: Wire>(&self, root: usize, values: &mut [T]) {
        let mut bytes = Vec::with_capacity(values.len() * T::WIDTH);
        for v in values.iter() {
            v.encode(&mut bytes);
        }
        self.inner.broadcast(root, &mut bytes);
        for (v, b) in values.iter_mut().zip(bytes.chunks_exact(T::WIDTH)) {
            *v = T::decode(b);
        }
    }
    /// Explicit deterministic rank-ordered custom-monoid reduction, then broadcast.
    pub fn all_reduce<A: Monoid>(&self, algebra: &A, value: &A::Element) -> A::Element
    where
        A::Element: Wire,
    {
        let mut buckets = vec![Vec::new(); self.size()];
        value.encode(&mut buckets[0]);
        let received = self.inner.exchange(&buckets);
        let mut result = algebra.zero();
        if self.rank() == 0 {
            for bytes in received {
                result = algebra.add(&result, &A::Element::decode(&bytes));
            }
        }
        self.broadcast(0, std::slice::from_mut(&mut result));
        result
    }
}
