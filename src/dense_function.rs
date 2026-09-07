// Adapted from cc4s CTF contraction/{sym_seq_ctr,ctr_comm}.cxx at
// f69cbb46e23bc2f39cda5722ce096f56301dab4f.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! Explicitly mapped distributed dense custom-function contraction.

use crate::{
    algebra::{Semiring, Wire},
    context::Context,
    diagonal::Projection,
    mapping::{Distribution, Mapping, Topology},
    tensor::Tensor,
};

fn projections(shapes: [&[usize]; 3], indices: [&str; 3]) -> [Projection; 3] {
    let mut dimensions = [None; 256];
    for operand in 0..3 {
        assert!(indices[operand].is_ascii());
        assert_eq!(shapes[operand].len(), indices[operand].len());
        for (&dimension, label) in shapes[operand].iter().zip(indices[operand].bytes()) {
            if let Some(previous) = dimensions[label as usize] {
                assert_eq!(previous, dimension);
            } else {
                dimensions[label as usize] = Some(dimension);
            }
        }
    }
    std::array::from_fn(|operand| Projection::new(shapes[operand], indices[operand]))
}

fn validate_physical(topology: &Topology, physical_labels: &str, indices: [&str; 3]) {
    assert!(physical_labels.is_ascii());
    assert_eq!(physical_labels.len(), topology.dimensions.len());
    for label in physical_labels.bytes() {
        let operands = indices.iter().filter(|labels| labels.as_bytes().contains(&label)).count();
        assert!(operands >= 2, "single-operand labels cannot be physically mapped");
    }
}

fn append_physical(map: &mut Mapping, topology: &Topology, axis: usize) {
    match map {
        Mapping::Unmapped => map.augment_physical(topology, axis),
        Mapping::Physical { child, .. } => append_physical(child, topology, axis),
        Mapping::Virtual { .. } => unreachable!(),
    }
}

fn mapped_distribution(
    shape: &[usize],
    indices: &str,
    topology: &Topology,
    physical_labels: &str,
) -> Distribution {
    let mut mappings = vec![Mapping::Unmapped; shape.len()];
    for (physical_axis, label) in physical_labels.bytes().enumerate() {
        if let Some(tensor_axis) = indices.bytes().position(|candidate| candidate == label) {
            append_physical(&mut mappings[tensor_axis], topology, physical_axis);
        }
    }
    Distribution::new(shape.to_vec(), topology.clone(), mappings)
}

fn actual_shape(distribution: &Distribution, rank: usize) -> Vec<usize> {
    let coordinates = distribution.topology.coordinates(rank);
    distribution.shape.iter().zip(&distribution.mappings).map(|(&length, mapping)| {
        let phase = mapping.physical_phase();
        let residue = mapping.physical_rank(&coordinates);
        if residue < length { (length - 1 - residue) / phase + 1 } else { 0 }
    }).collect()
}

fn packed_roots<A: Semiring>(tensor: &Tensor<'_, '_, A>, target: &Distribution)
    -> Vec<A::Element> where A::Element: Wire {
    let rank = tensor.context().rank();
    let keys: Vec<_> = (0..target.local_len())
        .filter_map(|offset| target.global_key(rank,offset)).collect();
    let requests: Vec<_> = keys.iter().enumerate()
        .filter(|(_,key)|target.owner(**key)==rank).map(|(offset,&key)|(offset,key)).collect();
    let read = tensor.read(&requests.iter().map(|&(_,key)|key).collect::<Vec<_>>());
    let mut values = vec![tensor.algebra().zero();keys.len()];
    for ((offset,_),value) in requests.into_iter().zip(read) {values[offset]=value;}
    values
}

impl<'c, 'r, A: Semiring + Clone> Tensor<'c, 'r, A>
where
    A::Element: Wire,
{
    /// Contract dense tensors with an arbitrary bivariate function on an
    /// explicit physical-label mapping. Labels absent from an operand create
    /// source-style broadcast or reduction fibers. Local buffers are repacked
    /// to their true rank extents before evaluating the function, so cyclic
    /// padding is never presented as an input value.
    pub fn contract_function_on(
        &mut self,
        indices_c: &str,
        a: &Self,
        indices_a: &str,
        b: &Self,
        indices_b: &str,
        topology: Topology,
        physical_labels: &str,
        alpha: A::Element,
        beta: A::Element,
        function: impl Fn(&A::Element, &A::Element) -> A::Element,
    ) {
        assert!(std::ptr::eq(self.context(), a.context())
            && std::ptr::eq(self.context(), b.context()));
        assert_eq!(topology.size(), self.context().size());
        let projections = projections(
            [&a.distribution().shape, &b.distribution().shape, &self.distribution().shape],
            [indices_a, indices_b, indices_c],
        );
        validate_physical(&topology, physical_labels,
            [&projections[0].labels, &projections[1].labels, &projections[2].labels]);
        if projections.iter().any(Projection::repeated) {
            let output_repeated = projections[2].repeated();
            let (aa, ia) = a.extract_diagonal(indices_a);
            let (bb, ib) = b.extract_diagonal(indices_b);
            let (mut cc, ic) = self.extract_diagonal(indices_c);
            cc.contract_function_on(&ic, &aa, &ia, &bb, &ib, topology,
                physical_labels, alpha, beta, function);
            if output_repeated { self.replace_diagonal(indices_c, &cc); }
            else { *self = cc; }
            return;
        }

        let distributions = [
            mapped_distribution(&a.distribution().shape, indices_a, &topology, physical_labels),
            mapped_distribution(&b.distribution().shape, indices_b, &topology, physical_labels),
            mapped_distribution(&self.distribution().shape, indices_c, &topology, physical_labels),
        ];
        let mut adata = packed_roots(a,&distributions[0]);
        let mut bdata = packed_roots(b,&distributions[1]);
        let mut cdata = packed_roots(self,&distributions[2]);
        let shapes = [
            actual_shape(&distributions[0], self.context().rank()),
            actual_shape(&distributions[1], self.context().rank()),
            actual_shape(&distributions[2], self.context().rank()),
        ];
        assert_eq!(adata.len(), shapes[0].iter().product());
        assert_eq!(bdata.len(), shapes[1].iter().product());
        assert_eq!(cdata.len(), shapes[2].iter().product());

        let mut comms: [Vec<Context<'_>>; 3] = std::array::from_fn(|_| Vec::new());
        let labels = [indices_a, indices_b, indices_c];
        for (axis, label) in physical_labels.bytes().enumerate() {
            for operand in 0..3 {
                if !labels[operand].as_bytes().contains(&label) {
                    comms[operand].push(topology.fiber(self.context(), axis));
                }
            }
        }
        for comm in &comms[0] { comm.broadcast(0, &mut adata); }
        for comm in &comms[1] { comm.broadcast(0, &mut bdata); }
        let root = comms[2].iter().all(|comm| comm.rank() == 0);
        let child_beta = if root { beta } else { self.algebra().zero() };
        crate::contraction::sequential_function(
            self.algebra(),
            &shapes[0], indices_a, &adata,
            &shapes[1], indices_b, &bdata,
            &shapes[2], indices_c, &mut cdata,
            &alpha, &child_beta, &function,
        );
        for comm in &comms[2] {
            comm.reduce_monoid(self.algebra(), &mut cdata, false, 0);
        }
        for group in comms {
            for comm in group { comm.close(); }
        }

        let rank = self.context().rank();
        let pairs: Vec<_> = (0..distributions[2].local_len())
            .filter_map(|offset|distributions[2].global_key(rank,offset)).zip(cdata)
            .filter(|(key,_)|distributions[2].owner(*key)==rank).collect();
        let zero = self.algebra().zero();
        self.transform(|_,value|*value=zero.clone());
        self.write_add(&pairs);
    }
}
