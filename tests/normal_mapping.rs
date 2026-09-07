use ctf::{mapping::{Mapping,Topology},normal_mapping::{Problem,Negative}};

fn physical(axis:usize)->Mapping{Mapping::Physical{axis,processes:2,child:Box::new(Mapping::Unmapped)}}

#[test]
fn source_pair_mapping_keeps_two_dimensional_contraction(){
    let problem=Problem::new([&[3,4],&[4,5],&[3,5]],["ik","kj","ij"]).unwrap();
    let topology=Topology::new(vec![2,2]);
    let result=problem.map_to_topology(&topology,0,[None;3]).unwrap();
    // map_ctr_indices assigns the paired k dimensions separately; A-fill then
    // maps i, and C-nonfill maps j. No globally aligned k replacement.
    assert_eq!(result[0].mappings,vec![physical(1),physical(0)]);
    assert_eq!(result[1].mappings,vec![physical(1),physical(0)]);
    assert_eq!(result[2].mappings,vec![physical(1),physical(0)]);
    let retained=problem.map_to_topology(&topology,0,result.each_ref().map(|d|Some(d.mappings.as_slice()))).unwrap();
    assert_eq!(retained,result);
}

#[test]
fn source_weigh_and_singleton_physical_rejections(){
    let topology=Topology::new(vec![2]);
    let common=Problem::new([&[7],&[7],&[7]],["i","i","i"]).unwrap();
    for permutation in 0..6{
        let result=common.map_to_topology(&topology,permutation,[None;3]).unwrap();
        for distribution in result{assert_eq!(distribution.mappings,vec![physical(0)]);}
    }
    let old=[physical(0)];
    assert_eq!(common.map_to_topology(&topology,0,[Some(&old),None,None]),Err(Negative::WeighPhysical));
    let extra=Problem::new([&[7],&[],&[]],["i","",""]).unwrap();
    assert_eq!(extra.map_to_topology(&topology,0,[Some(&old),None,None]),Err(Negative::ExtraPhysical));
}
