// Adapted from cc4s CTF tensor/algstrct.cxx csr_reduce at
// f69cbb46e23bc2f39cda5722ce096f56301dab4f.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! Partitioned sparse-matrix reduction without whole-matrix root gathering.

use crate::{
    algebra::{Monoid, Wire},
    context::Context,
    sparse_formats::{Ccsr, Coo, Csr},
};

fn encode<T: Wire + Clone>(coordinates: &Coo<T>) -> Vec<u8> {
    let (rows, columns) = coordinates.shape();
    let mut bytes = Vec::with_capacity(24 + coordinates.entries().len() * (16 + T::WIDTH));
    u64::try_from(rows).unwrap().encode(&mut bytes);
    u64::try_from(columns).unwrap().encode(&mut bytes);
    u64::try_from(coordinates.entries().len()).unwrap().encode(&mut bytes);
    for (row, column, value) in coordinates.entries() {
        u64::try_from(*row).unwrap().encode(&mut bytes);
        u64::try_from(*column).unwrap().encode(&mut bytes);
        value.encode(&mut bytes);
    }
    bytes
}

fn decode<T: Wire + Clone>(bytes: &[u8]) -> Coo<T> {
    assert!(bytes.len() >= 24);
    let rows = usize::try_from(u64::decode(&bytes[..8])).unwrap();
    let columns = usize::try_from(u64::decode(&bytes[8..16])).unwrap();
    let count = usize::try_from(u64::decode(&bytes[16..24])).unwrap();
    let width = 16 + T::WIDTH;
    assert_eq!(bytes.len(), 24 + count * width);
    let entries = bytes[24..]
        .chunks_exact(width)
        .map(|entry| {
            (
                usize::try_from(u64::decode(&entry[..8])).unwrap(),
                usize::try_from(u64::decode(&entry[8..16])).unwrap(),
                T::decode(&entry[16..]),
            )
        })
        .collect();
    Coo::new(rows, columns, entries)
}

macro_rules! define_reduce {
    ($matrix:ident, $from_coo:ident, $compressed_rows:expr) => {
        impl<T: Wire + Clone> $matrix<T> {
            pub fn reduce<A: Monoid<Element = T>>(
                &self,
                context: &Context<'_>,
                root: usize,
                algebra: &A,
            ) -> Option<Self> {
                let processes = context.size();
                assert!(root < processes);
                if processes == 1 {
                    return Some(self.clone());
                }

                let group_size = if $compressed_rows {
                    processes
                } else {
                    (2..).find(|divisor| processes % divisor == 0).unwrap()
                };
                let rank = context.rank();
                let within = rank % group_size;
                let group_index = rank / group_size;
                let group = context
                    .split(Some(group_index as i32), within as i32)
                    .unwrap();
                let column = context
                    .split(Some(within as i32), group_index as i32)
                    .unwrap();

                let partitions = self.partition(group_size);
                let mut outgoing = vec![Vec::new(); group_size];
                for (destination, partition) in partitions.iter().enumerate() {
                    if destination != within {
                        outgoing[destination] = encode(&partition.to_coo());
                    }
                }
                let incoming = group.inner.exchange(&outgoing);
                let mut summands: Vec<Option<Self>> = partitions
                    .into_iter()
                    .enumerate()
                    .map(|(source, partition)| {
                        Some(if source == within {
                            partition
                        } else {
                            decode(&incoming[source]).$from_coo()
                        })
                    })
                    .collect();

                let mut stride = 1;
                while stride < group_size {
                    for left in (0..group_size - stride).step_by(2 * stride) {
                        let first = summands[left].take().unwrap();
                        let second = summands[left + stride].take().unwrap();
                        summands[left] = Some(first.add(&second, algebra));
                    }
                    stride *= 2;
                }
                let local = summands[0].take().unwrap();
                let reduced = local.reduce(&column, root / group_size, algebra);

                let root_group = root / group_size;
                let root_within = root % group_size;
                let result = if group_index == root_group {
                    let bytes = encode(&reduced.as_ref().unwrap().to_coo());
                    let gathered = group.inner.gather_bytes(root_within, &bytes);
                    if within == root_within {
                        let parts: Vec<Self> = gathered
                            .unwrap()
                            .iter()
                            .map(|bytes| decode(bytes).$from_coo())
                            .collect();
                        Some(Self::assemble(&parts))
                    } else {
                        None
                    }
                } else {
                    None
                };

                group.close();
                column.close();
                result
            }
        }
    };
}

define_reduce!(Csr, to_csr, false);
define_reduce!(Ccsr, to_ccsr, true);
