//! Complete arbitrary-phase cyclic reshuffle workspaces.
//!
//! The pinned `glb_cyclic_reshuffle` returns immediately after packing while
//! its all-to-all and unpack stages are behind `#if 0`.  This implementation
//! intentionally completes count, pack, exchange, and scaled unpack.  It sends
//! only canonical values and never gathers a global tensor.

use crate::{
    algebra::{Monoid, Semiring, Wire},
    context::Context,
    mapping::Distribution,
    pad::CanonicalPadding,
    symmetric_distribution::SymmetricDistribution,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GlobalReshufflePlan {
    pub rank: usize,
    pub send_offsets: Vec<Vec<usize>>,
    pub receive_offsets: Vec<Vec<usize>>,
    pub send_counts: Vec<usize>,
    pub receive_counts: Vec<usize>,
    pub send_displacements: Vec<usize>,
    pub receive_displacements: Vec<usize>,
    source_len: usize,
    destination_padding: CanonicalPadding,
}

impl GlobalReshufflePlan {
    pub fn dense(old: &Distribution, new: &Distribution, rank: usize) -> Self {
        assert_eq!(old.shape, new.shape);
        assert_eq!(old.topology.size(), new.topology.size());
        let source = dense_pairs(old, rank);
        let destination = dense_pairs(new, rank);
        Self::from_pairs(old, new, rank, old.local_len(), source, destination,
            CanonicalPadding::new(new.local_len(), (0..new.local_len())
                .filter(|&offset| new.global_key(rank, offset).is_some()).collect()))
    }

    pub fn symmetric(
        old: &SymmetricDistribution,
        new: &SymmetricDistribution,
        rank: usize,
    ) -> Self {
        assert_eq!(old.distribution().shape, new.distribution().shape);
        assert_eq!(old.links(), new.links());
        assert_eq!(old.distribution().topology.size(), new.distribution().topology.size());
        Self::from_pairs(
            old.distribution(),
            new.distribution(),
            rank,
            old.local_len(),
            old.local_pairs(rank),
            new.local_pairs(rank),
            new.padding(rank),
        )
    }

    fn from_pairs(
        old: &Distribution,
        new: &Distribution,
        rank: usize,
        source_len: usize,
        mut source: Vec<(usize, usize)>,
        mut destination: Vec<(usize, usize)>,
        destination_padding: CanonicalPadding,
    ) -> Self {
        let size = old.topology.size();
        assert!(rank < size);
        source.sort_by_key(|&(_, key)| key);
        destination.sort_by_key(|&(_, key)| key);
        let mut send_offsets = vec![Vec::new(); size];
        for (offset, key) in source {
            if old.owner(key) == rank {
                for (destination, offsets) in send_offsets.iter_mut().enumerate() {
                    if new.owns(destination, key) { offsets.push(offset); }
                }
            }
        }
        let mut receive_offsets = vec![Vec::new(); size];
        for (offset, key) in destination {
            receive_offsets[old.owner(key)].push(offset);
        }
        let send_counts = send_offsets.iter().map(Vec::len).collect::<Vec<_>>();
        let receive_counts = receive_offsets.iter().map(Vec::len).collect::<Vec<_>>();
        let send_displacements = displacements(&send_counts);
        let receive_displacements = displacements(&receive_counts);
        Self { rank, send_offsets, receive_offsets, send_counts, receive_counts,
            send_displacements, receive_displacements, source_len,
            destination_padding }
    }

    pub fn pack<T: Wire>(&self, source: &[T]) -> Vec<Vec<u8>> {
        assert_eq!(source.len(), self.source_len);
        self.send_offsets.iter().map(|offsets| {
            let mut bytes=Vec::with_capacity(offsets.len()*T::WIDTH);
            for &offset in offsets { source[offset].encode(&mut bytes); }
            bytes
        }).collect()
    }

    pub fn unpack_scaled<A: Semiring>(
        &self, algebra: &A, received: &[Vec<u8>], destination: &mut [A::Element],
        alpha: &A::Element, beta: &A::Element,
    ) where A::Element: Wire {
        assert_eq!(received.len(), self.receive_offsets.len());
        assert_eq!(destination.len(), self.destination_padding.allocated_len());
        self.destination_padding.zero_padding(algebra, destination);
        for ((bytes, offsets), &count) in received.iter().zip(&self.receive_offsets).zip(&self.receive_counts) {
            assert_eq!(bytes.len(), count*A::Element::WIDTH);
            for (encoded, &offset) in bytes.chunks_exact(A::Element::WIDTH).zip(offsets) {
                let incoming=A::Element::decode(encoded);
                destination[offset]=algebra.add(&algebra.multiply(beta,&destination[offset]),
                    &algebra.multiply(alpha,&incoming));
            }
        }
    }

    pub fn execute_into<A: Semiring>(
        &self, context: &Context<'_>, algebra: &A, source: &[A::Element],
        destination: &mut [A::Element], alpha: &A::Element, beta: &A::Element,
    ) where A::Element: Wire {
        assert_eq!(context.rank(),self.rank);assert_eq!(context.size(),self.send_offsets.len());
        let received=context.inner.exchange(&self.pack(source));
        self.unpack_scaled(algebra,&received,destination,alpha,beta);
    }

    pub fn unpack<A: Monoid>(
        &self,
        algebra: &A,
        received: &[Vec<u8>],
        destination: &mut [A::Element],
    ) where
        A::Element: Wire,
    {
        assert_eq!(received.len(), self.receive_offsets.len());
        assert_eq!(destination.len(), self.destination_padding.allocated_len());
        for value in destination.iter_mut() { *value=algebra.zero(); }
        for ((bytes, offsets), &count) in received.iter().zip(&self.receive_offsets).zip(&self.receive_counts) {
            assert_eq!(bytes.len(), count*A::Element::WIDTH);
            for (encoded, &offset) in bytes.chunks_exact(A::Element::WIDTH).zip(offsets) {
                destination[offset]=A::Element::decode(encoded);
            }
        }
    }

    pub fn execute<A: Monoid>(
        &self,
        context: &Context<'_>,
        algebra: &A,
        source: &[A::Element],
        destination: &mut [A::Element],
    ) where
        A::Element: Wire,
    {
        assert_eq!(context.rank(),self.rank);assert_eq!(context.size(),self.send_offsets.len());
        let received=context.inner.exchange(&self.pack(source));
        self.unpack(algebra,&received,destination);
    }
}

fn dense_pairs(distribution:&Distribution,rank:usize)->Vec<(usize,usize)>{
    (0..distribution.local_len()).filter_map(|offset|
        distribution.global_key(rank,offset).map(|key|(offset,key))).collect()
}
fn displacements(counts:&[usize])->Vec<usize>{let mut total=0;counts.iter().map(|&count|{let d=total;total+=count;d}).collect()}

#[cfg(test)]
mod tests {
    use super::GlobalReshufflePlan;
    use crate::{algebra::Arithmetic,mapping::{Distribution,Mapping,Topology}};

    #[test]
    fn pinned_early_return_defect_is_not_preserved_count_pack_and_unpack_complete(){
        let topology=Topology::new(vec![2]);let old=Distribution::cyclic(vec![5],2);
        let new=Distribution::new(vec![5],topology,vec![Mapping::Unmapped]);
        let plans:Vec<_>=(0..2).map(|rank|GlobalReshufflePlan::dense(&old,&new,rank)).collect();
        assert_eq!(plans[0].send_counts,[3,3]);assert_eq!(plans[1].send_counts,[2,2]);
        assert_eq!(plans[0].receive_counts,[3,2]);assert_eq!(plans[0].receive_displacements,[0,3]);
        let source=[vec![10i64,12,14],vec![11,13,99]];
        let packed:Vec<_>=plans.iter().zip(&source).map(|(p,x)|p.pack(x)).collect();
        for destination_rank in 0..2 {
            let received=(0..2).map(|source_rank|packed[source_rank][destination_rank].clone()).collect::<Vec<_>>();
            let mut destination=vec![7i64;5];
            plans[destination_rank].unpack_scaled(&Arithmetic::<i64>::new(),&received,&mut destination,&2,&3);
            assert_eq!(destination,[41,43,45,47,49]);
        }
    }
}
