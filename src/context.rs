//! Explicit MPI lifecycle. Neither Runtime nor Context implements communicating Drop.
use crate::{algebra::{Monoid, Wire}, ffi::mpi};

#[must_use = "call finalize collectively when all contexts have been closed"]
pub struct Runtime { inner: mpi::Runtime }
impl Runtime {
    pub fn initialize() -> Self { Self { inner: mpi::Runtime::initialize() } }
    pub fn world(&self) -> Context<'_> { Context { inner: self.inner.world(), _runtime: self } }
    pub fn finalize(self) { self.inner.finalize(); }
}

#[must_use = "call close explicitly for communicators created by collective splits"]
pub struct Context<'runtime> {
    pub(crate) inner: mpi::Comm,
    _runtime: &'runtime Runtime,
}
impl Context<'_> {
    pub fn rank(&self) -> usize { self.inner.rank() }
    pub fn size(&self) -> usize { self.inner.size() }
    pub fn barrier(&self) { self.inner.barrier(); }
    /// Native MPI sum of one local f64 block. Collective; no global tensor gather.
    pub fn sum_f64(&self, values: &mut [f64]) { self.inner.sum_f64(values); }
    pub fn split(&self, color: Option<i32>, key: i32) -> Option<Context<'_>> {
        self.inner.split(color, key).map(|inner| Context { inner, _runtime: self._runtime })
    }
    pub fn split_shared(&self) -> Context<'_> {
        Context { inner: self.inner.split_shared(), _runtime: self._runtime }
    }
    pub fn close(self) { self.inner.close(); }
    pub fn broadcast<T: Wire>(&self, root: usize, values: &mut [T]) {
        let mut bytes = Vec::with_capacity(values.len() * T::WIDTH);
        for v in values.iter() { v.encode(&mut bytes); }
        self.inner.broadcast(root, &mut bytes);
        for (v, b) in values.iter_mut().zip(bytes.chunks_exact(T::WIDTH)) { *v = T::decode(b); }
    }
    /// Explicit deterministic rank-ordered custom-monoid reduction, then broadcast.
    pub fn all_reduce<A: Monoid>(&self, algebra: &A, value: &A::Element) -> A::Element
    where A::Element: Wire {
        let mut buckets = vec![Vec::new(); self.size()];
        value.encode(&mut buckets[0]);
        let received = self.inner.exchange(&buckets);
        let mut result = algebra.zero();
        if self.rank() == 0 {
            for bytes in received { result = algebra.add(&result, &A::Element::decode(&bytes)); }
        }
        self.broadcast(0, std::slice::from_mut(&mut result));
        result
    }
}
