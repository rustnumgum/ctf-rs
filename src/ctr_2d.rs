// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
// Adapted from contraction/ctr_2d_general.cxx::run and find_bsizes.
use crate::{
    algebra::{Semiring, Wire},
    context::Context,
};

#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub struct Layers {
    pub count: usize,
    pub index: usize,
}

#[derive(Clone,Copy)]
pub struct Panel<'c,'r> {
    pub comm: Option<&'c Context<'r>>,
    /// Number of noncontiguous strips per panel (ctr_lda).
    pub outer: usize,
    /// Contiguous elements per strip; zero denotes a stationary whole operand.
    pub inner: usize,
}
impl Panel<'_, '_> {
    fn size(&self) -> usize {
        self.outer * self.inner
    }
    fn pack<T: Clone>(&self, data: &[T], index: usize, stride: usize, work: &mut [T]) {
        for strip in 0..self.outer {
            let start = (strip*stride+index)*self.inner;
            work[strip * self.inner..(strip + 1) * self.inner]
                .clone_from_slice(&data[start..start + self.inner]);
        }
    }
    fn operand<'a, T: Wire + Clone>(
        &self,
        data: &'a [T],
        step: usize,
        edge: usize,
        work: &'a mut [T],
    ) -> &'a [T] {
        if let Some(comm) = self.comm {
            assert_eq!(edge%comm.size(),0);
            let owner = step%comm.size();
            if comm.rank() == owner {
                self.pack(data, step / comm.size(), edge / comm.size(), work);
            }
            comm.broadcast(owner,work);
            work
        } else if self.inner == 0 {
            data
        } else if self.outer == 1 {
            &data[step * self.inner..(step + 1) * self.inner]
        } else {
            self.pack(data, step, edge, work);
            work
    }
    }
    fn scatter<A: Semiring>(
        &self,
        algebra: &A,
        work: &[A::Element],
        data: &mut [A::Element],
        index: usize,
        stride: usize,
        beta: &A::Element,
    ) {
        for strip in 0..self.outer {
            let start = (strip*stride+index)*self.inner;
            for i in 0..self.inner {
                let previous = if *beta == algebra.zero() {
                    algebra.zero()
                } else {
                    algebra.multiply(beta, &data[start + i])
                };
                data[start + i] = algebra.add(&work[strip * self.inner + i], &previous);
            }
        }
    }
}

/// Execute one upstream 2D communication level. The child receives local panels,
/// its beta and the remaining replication-layer coordinates. At most two operands
/// move at one level. Communicators are borrowed and never freed by this routine.
pub fn execute<A: Semiring>(
    algebra: &A,
    edge: usize,
    layers: Layers,
    a_plan: Panel<'_, '_>,
    b_plan: Panel<'_, '_>,
    c_plan: Panel<'_, '_>,
    a: &[A::Element],
    b: &[A::Element],
    c: &mut [A::Element],
    beta: A::Element,
    mut child: impl FnMut(
        &[A::Element],
        &[A::Element],
        &mut [A::Element],
        A::Element,
        Layers,
    ),
) where
    A::Element: Wire,
{
    assert!(edge > 0 && layers.count > 0 && layers.index < layers.count);
    assert!(!(a_plan.comm.is_some() && b_plan.comm.is_some() && c_plan.comm.is_some()));
    let (count,index,next) = if edge >= layers.count && edge%layers.count == 0 {
        (layers.count,layers.index,Layers {count:1,index:0})
    } else if edge < layers.count && layers.count%edge == 0 {
        (
            edge,
            layers.index % edge,
            Layers {
                count: layers.count / edge,
                index: layers.index / edge,
            },
        )
    } else {
        (1, 0, layers)
    };
    let mut work_a = vec![algebra.zero(); a_plan.size()];
    let mut work_b = vec![algebra.zero(); b_plan.size()];
    let mut work_c = vec![algebra.zero(); c_plan.size()];
    let mut child_beta = beta.clone();
    for step in (index..edge).step_by(count) {
        let op_a = a_plan.operand(a,step,edge,&mut work_a);
        let op_b = b_plan.operand(b,step,edge,&mut work_b);
        if let Some(comm) = c_plan.comm {
            assert_eq!(edge%comm.size(),0);
            child(op_a, op_b, &mut work_c, algebra.zero(), next);
            let owner = step%comm.size();
            comm.reduce_monoid(algebra, &mut work_c, false, owner);
            if comm.rank() == owner {
                c_plan.scatter(
                    algebra,
                    &work_c,
                    c,
                    step / comm.size(),
                    edge / comm.size(),
                    &beta,
                );
            }
        } else if c_plan.inner == 0 {
            child(op_a, op_b, c, child_beta.clone(), next);
            child_beta = algebra.one();
        } else if c_plan.outer == 1 {
            child(
                op_a,
                op_b,
                &mut c[step * c_plan.inner..(step + 1) * c_plan.inner],
                beta.clone(),
                next,
            );
        } else {
            child(op_a, op_b, &mut work_c, algebra.zero(), next);
            c_plan.scatter(algebra, &work_c, c, step, edge, &beta);
        }
    }
}
