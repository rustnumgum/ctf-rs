use ctf::{mapping::{Mapping,Topology},mapping_preflight,mapping_variants::VariantSpace};
#[test]
fn source_count_choice_zero_and_three_two_dimensional_orientations(){
    let space=VariantSpace::new([&[3,4],&[4,5],&[3,5]],["ik","kj","ij"],Topology::new(vec![2,2])).unwrap();
    assert_eq!(space.len(),6);
    let first=space.decode(0).unwrap();assert_eq!(first.replicated_labels,vec![0,0]);
    assert_eq!(first.distributions[0].mappings[0].physical_phase(),4);
    for orientation in 0..3{
        let variant=space.decode(3+orientation).unwrap();
        assert_eq!(variant.two_dimensional,1);assert_eq!(variant.orientations,vec![orientation as u8]);
        assert!(mapping_preflight::check(variant.distributions.each_ref(),["ik","kj","ij"]));
        if orientation==1{
            assert_eq!(variant.distributions[0].mappings[0].phase(),1);
            assert_eq!(variant.distributions[2].mappings[0].phase(),1);
            assert!(!matches!(variant.distributions[0].mappings[0],Mapping::Physical{..}));
        }
    }
}
#[test]
fn rectangular_two_dimensional_grid_equalizes_shared_phases(){
    let space=VariantSpace::new([&[3,4],&[4,5],&[3,5]],["ik","kj","ij"],Topology::new(vec![2,3])).unwrap();
    let variant=space.decode(3).unwrap();
    assert_eq!(variant.distributions[0].mappings[1].phase(),6);
    assert_eq!(variant.distributions[1].mappings[0].phase(),6);
    assert!(mapping_preflight::check(variant.distributions.each_ref(),["ik","kj","ij"]));
}
