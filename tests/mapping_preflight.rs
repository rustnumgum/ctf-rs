use ctf::{mapping::{Distribution,Mapping,Topology},mapping_preflight::check};
fn physical(t:&Topology,axis:usize)->Mapping{let mut m=Mapping::Unmapped;m.augment_physical(t,axis);m}
#[test]
fn source_accepts_two_dimensional_mismatches_with_covering_output_maps(){
    let t=Topology::new(vec![2,2]);let x=physical(&t,0);let y=physical(&t,1);
    let a=Distribution::new(vec![3,4],t.clone(),vec![x.clone(),y.clone()]);
    let b=Distribution::new(vec![4,5],t.clone(),vec![x.clone(),y.clone()]);
    let c=Distribution::new(vec![3,5],t.clone(),vec![x,y]);
    assert!(check([&a,&b,&c],["ik","kj","ij"]));
    let bad=Distribution::new(vec![3,4],t,vec![a.mappings[0].clone(),Mapping::Unmapped]);
    assert!(!check([&bad,&b,&c],["ik","kj","ij"]));
}
#[test]
fn shared_three_way_labels_and_singletons_follow_source_rules(){
    let t=Topology::new(vec![2,2]);
    let a=Distribution::new(vec![3],t.clone(),vec![physical(&t,0)]);
    let b=Distribution::new(vec![3],t.clone(),vec![physical(&t,1)]);
    assert!(!check([&a,&b,&a],["i","i","i"]));
    let shared=Distribution::new(vec![3],t.clone(),vec![Mapping::Unmapped]);
    let extra=Distribution::new(vec![3,2],t.clone(),vec![Mapping::Unmapped,physical(&t,0)]);
    assert!(!check([&extra,&shared,&shared],["ix","i","i"]));
    let extra=Distribution::new(vec![3,2],t,vec![Mapping::Unmapped;2]);
    assert!(check([&extra,&shared,&shared],["ix","i","i"]));
}
