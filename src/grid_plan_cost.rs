// Adapted from cc4s contraction/{contraction,ctr_comm,ctr_tsr}.cxx.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! Cost-tree extraction for the aligned dense execution represented by GridPlan.
//! Redistribution into/out of the plan and tensor folding are separate costs.

use crate::{
    mapping::Mapping,
    plan_cost::{Collective, Tree},
    planning::GridPlan,
};

/// Pinned unfolded dense candidate estimate. This is source model memory, not
/// a Rust allocator peak. Folding is excluded; mapped_cost also uses this result
/// type for unfolded 2D-panel trees.
#[derive(Clone,Debug)]
pub struct UnfoldedEstimate {
    pub inner: crate::plan_cost::Estimate,
    pub redistribution_seconds: [f64;3],
    pub redistributed_input_bytes: usize,
    pub redistribution_temporary_bytes: usize,
    pub seconds: f64,
    pub memory_bytes: usize,
}

fn mark_physical_axes(mapping: &Mapping, used: &mut [bool]) {
    match mapping {
        Mapping::Unmapped => {}
        Mapping::Physical { axis, child, .. } => {
            used[*axis] = true;
            mark_physical_axes(child, used);
        }
        Mapping::Virtual { child, .. } => mark_physical_axes(child, used),
    }
}

impl GridPlan {
    /// Source detail_estimate_mem_and_time for this unfolded aligned path.
    /// C redistribution time is counted twice; only changed A/B contribute
    /// additional resident input storage. No folding cost is fabricated.
    pub fn estimate_unfolded(&self,models:&crate::cost::Models,element_bytes:usize,
        nodes_per_axis:&[usize],custom_reduce:bool)->UnfoldedEstimate {
        let inner=self.cost_tree(element_bytes,nodes_per_axis,custom_reduce).estimate(models,1);
        let old=self.signature().distributions();let mapped=self.mapped_distributions();
        let mut redistribution_seconds=[0.;3];
        let mut redistributed_input_bytes=0;let mut redistribution_temporary_bytes=0;
        for operand in 0..3 {
            if crate::redist_cost::same_mapping(&old[operand],&mapped[operand]) {continue;}
            let cost=crate::redist_cost::dense(&old[operand],&mapped[operand],element_bytes,models);
            redistribution_seconds[operand]=cost.seconds*if operand==2{2.}else{1.};
            redistribution_temporary_bytes+=cost.temporary_bytes;
            if operand<2 {redistributed_input_bytes+=mapped[operand].local_len()*element_bytes;}
        }
        let seconds=inner.seconds+redistribution_seconds.iter().sum::<f64>();
        let memory_bytes=redistributed_input_bytes+redistribution_temporary_bytes.max(inner.working_bytes);
        UnfoldedEstimate{inner,redistribution_seconds,redistributed_input_bytes,
            redistribution_temporary_bytes,seconds,memory_bytes}
    }

    /// Build the source replication/virtual/local tree for this unfolded dense
    /// aligned plan. `nodes_per_axis[i]` is the source `dim_comm[i].comm_nodes`.
    /// `custom_reduce` selects the native or custom reduction timing model; it
    /// does not change the standard local contraction model.
    pub fn cost_tree(
        &self,
        element_bytes: usize,
        nodes_per_axis: &[usize],
        custom_reduce: bool,
    ) -> Tree {
        let signature = self.signature();
        let indices = signature.indices();
        let mapped = self.mapped_distributions();
        let topology = signature.topology();
        assert_eq!(nodes_per_axis.len(), topology.dimensions.len());

        let blocks = mapped.each_ref().map(|distribution| distribution.block_shape());
        let operand_bytes = blocks
            .each_ref()
            .map(|shape| shape.iter().product::<usize>() * element_bytes);

        let labels = indices
            .iter()
            .flatten()
            .copied()
            .max()
            .map_or(0, |label| label + 1);
        let mut extents = vec![None; labels];
        let mut phases = vec![None; labels];
        for operand in 0..3 {
            for (axis, &label) in indices[operand].iter().enumerate() {
                let extent = blocks[operand][axis];
                let mapping = &mapped[operand].mappings[axis];
                let phase = mapping.phase() / mapping.physical_phase();
                if let Some(previous) = extents[label] {
                    assert_eq!(previous, extent);
                    assert_eq!(phases[label], Some(phase));
                } else {
                    extents[label] = Some(extent);
                    phases[label] = Some(phase);
                }
            }
        }
        let extents: Vec<usize> = extents.into_iter().map(Option::unwrap).collect();
        let phases: Vec<usize> = phases.into_iter().map(Option::unwrap).collect();

        let mut tree = Tree::Local {
            custom: false,
            folded: false,
            operand_bytes,
            flops: 2.0 * extents.iter().product::<usize>() as f64,
        };
        if phases.iter().product::<usize>() > 1 {
            tree = Tree::Virtual {
                phases,
                orders: std::array::from_fn(|operand| indices[operand].len()),
                child: Box::new(tree),
            };
        }

        let mut physical: [Vec<bool>; 3] =
            std::array::from_fn(|_| vec![false; topology.dimensions.len()]);
        for operand in 0..3 {
            for mapping in &mapped[operand].mappings {
                mark_physical_axes(mapping, &mut physical[operand]);
            }
        }
        let local_bytes = mapped
            .each_ref()
            .map(|distribution| distribution.local_len() * element_bytes);
        let mut inputs: [Vec<Collective>; 2] = std::array::from_fn(|_| Vec::new());
        let mut output = Vec::new();
        for axis in 0..topology.dimensions.len() {
            if !(physical[0][axis] || physical[1][axis] || physical[2][axis]) {
                continue;
            }
            for operand in 0..2 {
                if !physical[operand][axis] {
                    inputs[operand].push(Collective {
                        ranks: topology.dimensions[axis],
                        nodes: nodes_per_axis[axis],
                        bytes: local_bytes[operand],
                    });
                }
            }
            if !physical[2][axis] {
                output.push(Collective {
                    ranks: topology.dimensions[axis],
                    nodes: nodes_per_axis[axis],
                    bytes: local_bytes[2],
                });
            }
        }
        if !inputs[0].is_empty() || !inputs[1].is_empty() || !output.is_empty() {
            tree = Tree::Replicated {
                inputs,
                output,
                custom_reduce,
                child: Box::new(tree),
            };
        }
        tree
    }
}
